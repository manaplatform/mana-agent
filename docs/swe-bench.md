# SWE-bench Verified integration

Mana-Agent can generate **SWE-bench Verified** predictions in the official
JSONL format so the [SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/)
can grade them.

This integration is intentionally **prediction-generation + smoke grading only**.
It does **not** cover the full 500-instance leaderboard run, SWE-bench Pro,
Terminal-Bench, pass@k, or submission packaging.

## Prerequisites

```bash
# Mana-Agent available on PATH (editable install is fine)
pip install -e .

# Dataset loader used by the runner
pip install datasets

# Official harness (grading only) — Docker Desktop/engine must be running
pip install swebench
docker info
```

API credentials and model routing still come from your normal Mana config
(`~/.mana/config.toml` / secrets). The runner **forces a cheap/fast model** for
initial runs (default `gpt-4o-mini`) via `--model` / `OPENAI_CHAT_MODEL`.

## Generate predictions

From the repository root:

```bash
python scripts/swe_bench/runner.py \
  --limit 1 \
  --output predictions.jsonl \
  --timeout 600 \
  --model gpt-4o-mini
```

### Exact prediction line format

Each line of `predictions.jsonl` is one JSON object:

```json
{"instance_id": "astropy__astropy-12907", "model_name_or_path": "mana-agent", "model_patch": "diff --git a/..."}
```

Required keys (official harness):

| Field | Value |
| --- | --- |
| `instance_id` | SWE-bench Verified instance id |
| `model_name_or_path` | Default `mana-agent` (override with `--model-name-or-path`) |
| `model_patch` | Unified diff string (may be empty if the agent produced no changes) |

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--limit N` | Run at most N instances (after filters) |
| `--instance-ids ID[,ID...]` | Only these instances (repeatable) |
| `--output PATH` | Predictions path (default `predictions.jsonl`) |
| `--timeout SECONDS` | Hard per-instance wall-clock timeout (default `600`) |
| `--model ID` | Forced cheap/fast LLM for mana-agent (default `gpt-4o-mini`) |
| `--work-dir DIR` | Clones, worktrees, and logs (default `.swe-bench`) |
| `--retain-worktrees` | Keep checkouts for debugging |
| `--skip-agent` | Write empty patches without calling mana-agent (format smoke) |
| `--fail-fast` | Stop after the first hard instance failure |
| `-v` / `--verbose` | Debug logging |

### Example smoke-run commands

**Format-only smoke** (no LLM, empty patches, validates JSONL shape):

```bash
python scripts/swe_bench/runner.py \
  --limit 1 \
  --skip-agent \
  --output predictions.jsonl
```

**Single real instance** (cheap model, hard timeout):

```bash
python scripts/swe_bench/runner.py \
  --instance-ids astropy__astropy-12907 \
  --model gpt-4o-mini \
  --timeout 600 \
  --output predictions.jsonl
```

**Small batch**:

```bash
python scripts/swe_bench/runner.py \
  --limit 3 \
  --model gpt-4o-mini \
  --timeout 600 \
  --output predictions.jsonl
```

### What the runner does per instance

1. Load `princeton-nlp/SWE-bench_Verified` (HuggingFace `datasets`).
2. Clone or reuse a cache of `https://github.com/<repo>.git` under `.swe-bench/repos/`.
3. Create a **clean detached worktree** at `base_commit`.
4. Invoke mana-agent **non-interactively**:

   ```text
   mana-agent chat --no-tui --root-dir <worktree> --model <cheap> --full-auto \
     --ephemeral-index --auto-continue --execution-profile full-auto \
     "<issue prompt>"
   ```

   stdin is closed so the session exits after the single coding turn.
5. Capture `git add -A` + `git diff --cached` as `model_patch`.
6. Append one JSONL prediction (including empty patches).
7. Remove the worktree unless `--retain-worktrees`.

Per-instance logs land under `.swe-bench/logs/<instance_id>/`.

## Grade predictions with the official harness

After `predictions.jsonl` exists:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions.jsonl \
  --max_workers 4 \
  --run_id mana-agent-smoke
```

Notes:

- The harness builds/runs **Docker** evaluation images; first runs are slow.
- Empty `model_patch` values are valid inputs. The harness reports them under
  **Instances with empty patches** and does not schedule Docker evaluation for
  those rows (they contribute 0 resolved). Non-empty patches are what exercise
  the full apply+test path.
- Docker Desktop (or another Docker engine) must be running before grading.
- Restrict grading to the same instances you generated when smoking.

Inspect harness output under the report path printed by the harness
(e.g. `mana-agent.mana-agent-smoke.json` for `--run_id mana-agent-smoke`).

## Hardening behavior

| Condition | Behavior |
| --- | --- |
| Dirty worktree right after checkout | Fail the instance; write empty patch; continue (unless `--fail-fast`) |
| Checkout / missing `base_commit` | Fail the instance; empty patch |
| mana-agent hang | Kill process group at `--timeout`; status `timeout`; empty or partial patch |
| mana-agent non-zero exit | Status `agent_error`; still capture any partial patch |
| Empty patch after success | Status `empty_patch`; still write a valid JSONL line |
| Missing `datasets` package | Exit with install instructions |

## Current limitations

- **Not** a full Verified (500) evaluation pipeline or leaderboard submission path.
- Does not install project-specific conda/test environments; the agent only edits the tree. Official tests run later inside harness Docker images.
- Default model is intentionally **cheap/fast** (`gpt-4o-mini`); quality will be lower than production coding models.
- Single-shot chat invocation: no multi-trial pass@k, no ensemble voting.
- Repo clones can be large; disk under `--work-dir` grows with unique repositories.
- Network required for HuggingFace dataset load and GitHub clones.
- No SWE-bench Pro / Multilingual / Terminal-Bench support in this runner.
- Agent may still ask for clarification or stop early; empty patches are recorded rather than retried automatically.

## Recommended verification (user-owned)

```bash
# 1) Format smoke
python scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl
python -c "import json; print(json.loads(open('predictions.jsonl').readline()).keys())"

# 2) Optional one-instance agent run
python scripts/swe_bench/runner.py --limit 1 --model gpt-4o-mini --timeout 600 --output predictions.jsonl

# 3) Official harness smoke grade
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions.jsonl \
  --max_workers 4 \
  --run_id mana-agent-smoke
```

Verification is user-owned; do not treat agent-authored changes as passing until these commands are run locally.

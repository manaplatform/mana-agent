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

API credentials still come from your normal Mana config (`~/.mana/config.toml` /
secrets), copied into an **isolated per-instance `MANA_HOME`**. The runner then
rewrites that isolated config so:

* `--model` (default `gpt-4o-mini`) pins **all** model roles (Mana file config
  wins over process env, so env-only overrides are not enough).
* `MANA_MEMORY_MODE=internal` — external supermemory is disabled for the run so
  chat startup does not block on remote memory HTTP.
* Chat is launched non-interactively (`--no-interactive --no-tui`), without a
  synchronous full-repo index (`--no-auto-index-missing`), and exits after the
  single coding turn.

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
{
  "instance_id": "astropy__astropy-12907",
  "model_name_or_path": "mana-agent__gpt-4o-mini",
  "agent_name": "mana-agent",
  "agent_model": "gpt-4o-mini",
  "model_patch": "diff --git a/..."
}
```

Field contract:

| Field | Required by harness | Value |
| --- | --- | --- |
| `instance_id` | yes | SWE-bench Verified instance id |
| `model_name_or_path` | yes | **System id for reports**, default `{agent_name}__{model}` (e.g. `mana-agent__gpt-4o-mini`). **Not** the agent name alone. Override with `--model-name-or-path`. |
| `agent_name` | no (Mana always writes it) | Coding agent identity, default `mana-agent` (`--agent-name`) |
| `agent_model` | no (Mana always writes it) | LLM id used for the run (`--model`) |
| `model_patch` | yes | Unified diff string (may be empty if the agent produced no changes) |

**Naming rule:** do not put `mana-agent` in `model_name_or_path` by itself. That field is used by the harness / `sb-cli` as the system label for the run. Use agent + model (`mana-agent__gpt-5.6-luna`) or an explicit custom label via `--model-name-or-path`.

**Test files:** by default the runner **strips test-file hunks** from `model_patch` (paths under `tests/`, `test_*.py`, etc.). SWE-bench applies the official `test_patch` after your patch; agent test edits often produce report status `failed` (apply/runtime error) instead of `resolved` / `unresolved`. Pass `--keep-test-files` only when you intentionally want those hunks.

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--limit N` | Run at most N instances (after filters) |
| `--instance-ids ID[,ID...]` | Only these instances (repeatable) |
| `--output PATH` | Predictions path (default `predictions.jsonl`) |
| `--timeout SECONDS` | Hard per-instance wall-clock timeout (default `600`) |
| `--model ID` | Forced LLM for mana-agent (default `gpt-4o-mini`); also used in default `model_name_or_path` |
| `--agent-name NAME` | Agent identity written as `agent_name` (default `mana-agent`) |
| `--model-name-or-path NAME` | Override harness system id (default `{agent}__{model}`) |
| `--keep-test-files` | Keep test-file hunks in `model_patch` (off by default) |
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
   mana-agent --no-interactive --no-banner chat --no-tui --root-dir <worktree> \
     --model <forced> --full-auto --no-auto-index-missing --no-coding-memory \
     --auto-continue --execution-profile full-auto \
     --auto-execute-max-passes 10 --agent-timeout-seconds <N> \
     "<issue prompt>"
   ```

   stdin is closed; non-TTY single-shot chat exits after the coding turn.
   The runner emits a heartbeat every ~30s while mana-agent is still running
   (PID, elapsed time, log sizes, stderr tail).
5. Capture `git add -A` + `git diff --cached` as `model_patch`, then **drop
   test-file hunks** by default so grading does not fail on test edits.
6. Append one JSONL prediction with `model_name_or_path`, `agent_name`,
   `agent_model`, and `model_patch` (including empty patches).
7. Remove the worktree unless `--retain-worktrees`.

Per-instance logs land under `.swe-bench/logs/<instance_id>/`
(`mana_stdout.log`, `mana_stderr.log`, `mana_summary.txt`, `mana_cmd.txt`,
`prompt.txt`, isolated `mana_home/`).

## Grade predictions with the official harness

After `predictions.jsonl` exists, grade **only the instances you submitted**
(smoke runs almost never cover all 500 Verified rows):

```bash
# Prefer local harness for smoke. Pin instance_ids to the rows in predictions.jsonl.
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions.jsonl \
  --instance_ids astropy__astropy-12907 \
  --max_workers 4 \
  --run_id mana-agent__gpt-4o-mini-smoke
```

Or with `sb-cli` (cloud). Use a **run_id that includes agent + model**, not the
agent name alone:

```bash
sb-cli submit swe-bench_verified test \
  --predictions_path predictions.jsonl \
  --run_id mana-agent__gpt-5.6-luna-smoke \
  --instance_ids astropy__astropy-12907 \
  --output_dir ./sb-cli-reports
```

### How to read a report (incomplete vs failed vs unresolved)

A typical smoke report after submitting **1 of 500** instances looks like:

| Field | Meaning |
| --- | --- |
| `total_instances` | Size of the dataset split (500 for Verified `test`) |
| `submitted_instances` / `submitted_ids` | Rows present in your predictions file |
| `incomplete_ids` | Dataset ids **not** in your submission (expected for smoke; not agent bugs) |
| `failed_ids` / `failed_instances` | Evaluation **could not complete** (patch apply error, container error, etc.) — not the same as “wrong fix” |
| `resolved_ids` | Patch applied and FAIL_TO_PASS (+ PASS_TO_PASS) tests passed |
| `unresolved_ids` | Evaluation completed; tests did not pass |
| `error_ids` | Harness/infrastructure errors |

If you only generated one prediction, **499 incomplete ids are expected**. Do not
re-run all 500 unless you intentionally want a full Verified pass.

When the single submitted id is in `failed_ids` (not `unresolved_ids`), check:

1. `model_patch` does not edit test files that collide with the official `test_patch` (runner strips these by default).
2. `model_name_or_path` is a stable system id (`mana-agent__<llm>`), not `mana-agent` alone.
3. Docker / `sb-cli` environment health for that instance log.

Notes:

- The harness builds/runs **Docker** evaluation images; first runs are slow.
- Empty `model_patch` values are valid inputs. The harness reports them under
  **Instances with empty patches** and does not schedule Docker evaluation for
  those rows (they contribute 0 resolved). Non-empty patches are what exercise
  the full apply+test path.
- Docker Desktop (or another Docker engine) must be running before grading.
- Restrict grading to the same instances you generated when smoking
  (`--instance_ids` / sb-cli `--instance_ids`).

Inspect harness output under the report path printed by the harness
(e.g. `mana-agent__gpt-4o-mini.mana-agent__gpt-4o-mini-smoke.json`).

## Hardening behavior

| Condition | Behavior |
| --- | --- |
| Dirty worktree right after checkout | Fail the instance; write empty patch; continue (unless `--fail-fast`) |
| Checkout / missing `base_commit` | Fail the instance; empty patch |
| mana-agent hang | Heartbeat every ~30s; kill process group at `--timeout`; status `timeout`; empty or partial patch |
| mana-agent non-zero exit | Status `agent_error`; still capture any partial patch |
| Empty patch after success | Status `empty_patch`; still write a valid JSONL line |
| Operator external memory / model pins | Isolated `MANA_HOME` rewritten to internal memory + forced `--model` |
| Large-repo semantic index | Not built synchronously (`--no-auto-index-missing`; no `--ephemeral-index`) |
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
# 1) Format smoke (empty patches; checks keys + naming)
python scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl
python -c "import json; r=json.loads(open('predictions.jsonl').readline()); print(sorted(r.keys())); print(r['model_name_or_path'], r['agent_name'], r.get('agent_model'))"

# 2) Optional one-instance agent run (use your real LLM id)
python scripts/swe_bench/runner.py \
  --instance-ids astropy__astropy-12907 \
  --model gpt-5.6-luna \
  --timeout 600 \
  --output predictions.jsonl

# 3) Local harness smoke grade for the same instance only
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions.jsonl \
  --instance_ids astropy__astropy-12907 \
  --max_workers 1 \
  --run_id mana-agent__gpt-5.6-luna-smoke
```

Expect `model_name_or_path` like `mana-agent__gpt-5.6-luna` and `agent_name` of
`mana-agent`. Incomplete dataset ids in cloud reports for unsubmitted rows are
normal for smoke runs.

Verification is user-owned; do not treat agent-authored changes as passing until these commands are run locally.

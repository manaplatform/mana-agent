# Mana Eval Lab

Eval workspace backend contracts include `local-worktree`, `execution-fabric`,
and `fleet`. Fleet defaults must explicitly declare platforms and minimum
coverage. An unavailable Fleet service or invalid selection fails before
execution; Eval Lab never substitutes a local workspace. See
[cross-platform verification](../25-cross-platform-verification.md#eval-lab).

Mana Eval Lab runs the existing Mana-Agent gateway in a clean Git worktree and records a portable, versioned execution history. It does not implement a second gateway, router, coding agent, tool runner, reviewer, or verifier.

The execution contract is:

```text
Validated suite → isolated worktree → existing gateway/runtime → recorder events
→ objective tests and scoring → immutable artifacts → reports and regression gate
```

Evaluation context is propagated with a context variable. Normal chat uses the no-op recorder and requires no evaluation configuration. The gateway, entry router, decision engine, lane coordinator, tool executor, Codex shim, reviewer, and verifier emit records only when an evaluation context is active.

JSON and JSONL files are canonical. `.mana/evals/evals.sqlite3` is a rebuildable, file-locked query index. Completed run directories are immutable. Docker and remote workspace backends are typed but intentionally return explicit unsupported errors in this P0; `local-worktree` is the implemented backend.

Start with:

```bash
mana-agent eval doctor
mana-agent eval run evals/suites/routing-smoke.yaml --variant candidate
mana-agent eval list
```

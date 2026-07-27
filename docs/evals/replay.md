# Replaying evaluations

Task replay reconstructs the stored suite and re-executes the original task from its recorded commit with the selected variant. It always creates a new immutable run with `replayed_from_run_id`; the original is not modified.

```bash
mana-agent eval replay <run-id> --variant candidate
```

Trajectory replay creates a fresh worktree and executes recorded commands in order without calling a language model. It stops at the first status or output-hash divergence and reports the event sequence plus whether the repository diff changed.

```bash
mana-agent eval replay-trajectory <run-id>
```

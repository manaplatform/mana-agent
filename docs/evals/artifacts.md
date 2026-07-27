# Evaluation artifacts and security

Each run lives under `.mana/evals/runs/<run-id>/` with `run.json`, incremental `events.jsonl`, the redacted suite/configuration, environment snapshot, route, commands, tool calls, tests, patch, outcome, and logs. `.complete` is written only after atomic finalization. A directory without it is reported as incomplete.

The centralized redactor runs before persistence and removes secret-named fields, authorization headers, cookies, passwords, provider keys, GitHub tokens, credentials in URLs, bearer tokens, and private keys. Task-specific patterns may be added in suite configuration. Environment snapshots store only relevant variable names, never values.

Raw run data and local indexes are ignored by Git. Checked-in baselines contain aggregate and per-task outcomes plus artifact hashes—not raw prompts, connector output, or logs.

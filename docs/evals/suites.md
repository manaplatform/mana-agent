# Evaluation suites

Suites are versioned YAML files containing tasks, variants, score weights, and gate thresholds. Loading is fail-closed: duplicate IDs, unresolved environment variables, invalid commits, unknown tools or lanes, invalid weights, missing required test commands, and unsupported workspace backends stop before a run is created.

The matrix is `tasks × variants × trials`. A run fingerprint includes the complete suite, task, variant, and trial. Only a completed successful run with that exact fingerprint may be reused; failed or incomplete runs are never cached as success.

Task assertions can constrain routes, tools, changed files, mutation, tokens, latency, cost, and test commands. Objective test failures always make the task unsuccessful, regardless of reviewer scores.

Model names are explicit in each variant. Provider credentials are never persisted. A missing or unavailable provider produces a failed evaluation; the runner never switches models or providers.

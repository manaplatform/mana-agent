# Scoring and leaderboards

Scores combine independent dimensions configured in the suite: task completion, tests, reviewer quality, routing correctness, policy compliance, efficiency, reproducibility, verification, patch quality, and safety. Weights are stored with every run rather than hidden in code.

A missing or failed required test is an objective failure and cannot be overridden by an LLM reviewer. Unknown model pricing remains `null`/`unknown`; it is never converted to zero. Cost gates may either ignore unknown prices or require known pricing and fail validation.

Leaderboard output is written as JSON, CSV, and Markdown. Every row includes the suite version so results from incompatible suites are not presented as directly equivalent.

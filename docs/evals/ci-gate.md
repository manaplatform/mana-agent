# Regression CI gate

The regression gate pairs baseline and candidate results by stable task ID. It compares score, pass rate, routing accuracy, policy violations, latency, and cost. Tasks are classified as `improved`, `unchanged`, `regressed`, `new`, `missing`, or `inconclusive`.

The gate fails closed for missing candidates, incomplete comparisons, incompatible or insufficient baselines, provider/evaluator failures, new policy violations, and configured metric threshold breaches. Exit codes are stable:

- `0`: success
- `2`: configuration failure
- `3`: execution or provider failure
- `4`: regression failure
- `5`: incomplete comparison

Reports are `regression.json`, `regression.md`, and `regression.junit.xml`; the JUnit file exposes task regressions as CI failures.

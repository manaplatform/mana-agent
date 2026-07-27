import pytest

from mana_agent.evals.config import SuiteDefaults


def test_fleet_eval_defaults_require_explicit_platform_coverage() -> None:
    with pytest.raises(ValueError, match="required_platforms"):
        SuiteDefaults(workspace_backend="fleet")
    defaults = SuiteDefaults(
        workspace_backend="fleet",
        required_platforms=["linux", "windows", "macos"],
        minimum_platform_coverage=3,
        runtime={"python": ["3.12"]},
        fleet={
            "labels": {"include": ["trusted"], "exclude": ["experimental"]},
            "maximum_workers": 6,
            "per_worker_concurrency": 1,
        },
    )
    assert defaults.minimum_platform_coverage == 3

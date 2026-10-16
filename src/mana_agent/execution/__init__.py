from mana_agent.config.autonomous_config import AutoRenewConfig, AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionController,
    AutonomousExecutionResult,
    TaskExecutionMetrics,
)
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationDecisionError,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution.manager import ExecutionManager


def build_execution_manager(settings, *, event_sink=None) -> ExecutionManager:
    config = ExecutionConfig.from_settings(settings)
    return ExecutionManager(build_provider_registry(config), config, event_sink=event_sink)


__all__ = [
    "AutoRenewConfig",
    "AutonomousExecutionConfig",
    "AutonomousExecutionController",
    "AutonomousExecutionResult",
    "ContinuationAction",
    "ContinuationDecision",
    "ContinuationDecisionError",
    "ContinuationEvaluator",
    "ExecutionConfig",
    "ExecutionManager",
    "TaskExecutionMetrics",
    "build_execution_manager",
]

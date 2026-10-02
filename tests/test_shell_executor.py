"""Unit tests for OpenAI-compatible Local Shell Tool, executor, policy, and security."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mana_agent.evals.config import load_suite
from mana_agent.execution.config import build_provider_registry
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.models import RoutingRequest, SandboxSpec
from mana_agent.execution.router import ExecutionRouter
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    ActionEffectScope,
    ActionRequestState,
    EffectScope,
    SideEffectClassification,
    infer_effect_scope,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.tools.catalog import list_auto_chat_tools
from mana_agent.tools.contracts import coding_tool_contracts, shell_tool_contract
from mana_agent.tools.shell_exec import (
    DANGEROUS_SHELL_PATTERNS,
    UNTRUSTED_CONTENT_NOTICE,
    ShellCallAction,
    ShellCommandOutput,
    ShellExecutor,
    ShellOutcome,
    is_read_only_command,
)
from mana_agent.transactional_actions.adapters import ShellActionAdapter
from mana_agent.transactional_actions.models import PolicyOutcome
from mana_agent.transactional_actions.policy import ActionPolicy, PolicyConfig


def local_config():
    from mana_agent.execution.config import ExecutionConfig

    return ExecutionConfig(
        providers={
            name: {"enabled": name == "local-process"}
            for name in (
                "local-process",
                "local-docker",
                "remote-ssh",
                "kubernetes",
                "modal",
                "custom-http-runtime",
            )
        }
    )


def test_shell_executor_runs_command_and_returns_openai_shape(tmp_path: Path) -> None:
    executor = ShellExecutor(workspace_root=tmp_path)
    result = executor.execute(commands=["echo hello world"])

    assert result["type"] == "shell_call_output"
    assert "output" in result
    assert len(result["output"]) == 1

    item = result["output"][0]
    assert "hello world" in item["stdout"]
    assert UNTRUSTED_CONTENT_NOTICE in result["untrusted_content_notice"]
    assert item["outcome"] == {"type": "exit", "exit_code": 0}


def test_shell_executor_runs_without_shell_true(tmp_path: Path, monkeypatch) -> None:
    captured_calls: list[dict] = []
    real_popen = subprocess.Popen

    def mock_popen(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    executor = ShellExecutor(workspace_root=tmp_path)
    executor.execute(commands=["echo safe_execution"])

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["kwargs"].get("shell") is False
    # First argument must be argv list, not a raw shell string
    argv = call["args"][0]
    assert isinstance(argv, list)
    assert argv[0] == "echo"
    assert argv[1] == "safe_execution"


def test_shell_executor_timeout_kills_process_and_returns_partial_output(tmp_path: Path) -> None:
    executor = ShellExecutor(workspace_root=tmp_path, needs_approval=False)
    # Python command that prints immediately and then sleeps
    cmd = (
        f'{sys.executable} -c '
        f'"import sys, time; sys.stdout.write(\'partial_before_timeout\\n\'); '
        f'sys.stdout.flush(); time.sleep(5)"'
    )
    result = executor.execute(commands=[cmd], timeout_ms=400)

    assert len(result["output"]) == 1
    item = result["output"][0]
    assert item["outcome"] == {"type": "timeout"}
    assert "partial_before_timeout" in item["stdout"]


def test_shell_executor_preserves_non_zero_exit_code(tmp_path: Path) -> None:
    executor = ShellExecutor(workspace_root=tmp_path, needs_approval=False)
    cmd = (
        f'{sys.executable} -c '
        f'"import sys; sys.stdout.write(\'progress msg\\n\'); '
        f'sys.stderr.write(\'recovery error reason\\n\'); sys.exit(42)"'
    )
    result = executor.execute(commands=[cmd])

    assert len(result["output"]) == 1
    item = result["output"][0]
    assert item["outcome"] == {"type": "exit", "exit_code": 42}
    assert "progress msg" in item["stdout"]
    assert "recovery error reason" in item["stderr"]


def test_shell_executor_truncation_per_max_output_length(tmp_path: Path) -> None:
    executor = ShellExecutor(workspace_root=tmp_path, needs_approval=False)
    cmd = f'{sys.executable} -c "print(\'A\' * 3000)"'
    max_len = 150
    result = executor.execute(commands=[cmd], max_output_length=max_len)

    assert len(result["output"]) == 1
    item = result["output"][0]
    # Check that truncation occurred
    assert "truncated" in item["stdout"].lower()
    content_part = item["stdout"].split("\n[Output truncated")[0]
    assert len(content_part.strip()) <= max_len


def test_shell_executor_cwd_locked_to_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    executor = ShellExecutor(workspace_root=workspace)

    # Executing pointing cwd to an outside directory should fail with safe error outcome
    out = executor.run_command("echo test", cwd=outside)
    assert out.outcome.type == "exit" and out.outcome.exit_code != 0
    assert "escapes workspace" in out.stderr.lower()


def test_shell_executor_sanitizes_environment_and_secrets(tmp_path: Path, monkeypatch) -> None:
    secret_value = "super_secret_token_12345"
    monkeypatch.setenv("MANA_AGENT_TOKEN", secret_value)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_key_67890")

    executor = ShellExecutor(workspace_root=tmp_path, needs_approval=False)
    # 1. Environment variable must be sanitized and omitted from child env
    cmd_env = (
        f'{sys.executable} -c '
        f'"import os; print(\'Token:\' + os.environ.get(\'MANA_AGENT_TOKEN\', \'\'))"'
    )
    result_env = executor.execute(commands=[cmd_env])
    stdout_env = result_env["output"][0]["stdout"]
    assert secret_value not in stdout_env
    assert stdout_env.strip() == "Token:"

    # 2. Secret value in command output must be redacted
    cmd_leak = f'{sys.executable} -c "print(\'leaked:\' + {secret_value!r})"'
    result_leak = executor.execute(commands=[cmd_leak])
    stdout_leak = result_leak["output"][0]["stdout"]
    assert secret_value not in stdout_leak
    assert "[REDACTED]" in stdout_leak


def test_shell_executor_denylist_blocks_dangerous_commands(tmp_path: Path) -> None:
    executor = ShellExecutor(workspace_root=tmp_path)

    for dangerous in ["rm -rf /", ":(){ :|:& };:", "mkfs.ext4 /dev/sda1"]:
        result = executor.execute(commands=[dangerous])
        assert len(result["output"]) == 1
        item = result["output"][0]
        assert item["outcome"] == {"type": "exit", "exit_code": 1}
        assert "security policy" in item["stderr"].lower() or "blocked" in item["stderr"].lower()


def test_shell_executor_read_only_vs_mutating_classification() -> None:
    assert is_read_only_command("ls -la") is True
    assert is_read_only_command("git status") is True
    assert is_read_only_command("git diff HEAD~1") is True
    assert is_read_only_command("git log -n 5") is True
    assert is_read_only_command("cat README.md") is True
    assert is_read_only_command("head -n 20 setup.py") is True
    assert is_read_only_command("grep -rn 'def ' .") is True
    assert is_read_only_command("pwd") is True

    # Mutating commands must NOT be classified as read-only
    assert is_read_only_command("touch file.txt") is False
    assert is_read_only_command("rm file.txt") is False
    assert is_read_only_command("git commit -m 'feat'") is False
    assert is_read_only_command("git push origin main") is False
    assert is_read_only_command("mv a.txt b.txt") is False
    assert is_read_only_command("cp a.txt b.txt") is False
    assert is_read_only_command("mkdir newdir") is False
    assert is_read_only_command("python script.py") is False


def test_shell_policy_approval_flow(tmp_path: Path) -> None:
    policy = ActionPolicy(PolicyConfig(workspace_roots=(tmp_path,)))

    # Read-only command is free (ALLOW)
    read_adapter = ShellActionAdapter(
        argv=["ls", "-la"],
        cwd=tmp_path,
        environment={},
        expected_outputs=[],
        parent_task_id="test",
        actor="model",
        originating_agent="ask",
        idempotency_key="read-action-0001",
        timeout_seconds=30,
        runner=lambda *_, **__: None,
    )
    outcome_read = policy.evaluate(read_adapter.build_intent()).outcome
    assert outcome_read == PolicyOutcome.ALLOW

    # Mutating command requires human approval (REQUIRE_APPROVAL)
    mutating_adapter = ShellActionAdapter(
        argv=["touch", "created.txt"],
        cwd=tmp_path,
        environment={},
        expected_outputs=[],
        parent_task_id="test",
        actor="model",
        originating_agent="ask",
        idempotency_key="write-action-0001",
        timeout_seconds=30,
        runner=lambda *_, **__: None,
    )
    outcome_mutating = policy.evaluate(mutating_adapter.build_intent()).outcome
    assert outcome_mutating == PolicyOutcome.REQUIRE_APPROVAL

    # Dangerous command is denied
    dangerous_adapter = ShellActionAdapter(
        argv=["rm", "-rf", "/"],
        cwd=tmp_path,
        environment={},
        expected_outputs=[],
        parent_task_id="test",
        actor="model",
        originating_agent="ask",
        idempotency_key="danger-action-0001",
        timeout_seconds=30,
        runner=lambda *_, **__: None,
    )
    outcome_dangerous = policy.evaluate(dangerous_adapter.build_intent()).outcome
    assert outcome_dangerous == PolicyOutcome.DENY


def test_shell_contracts_and_catalog_registration() -> None:
    contract = shell_tool_contract()
    assert contract.name == "shell"
    assert "commands" in contract.input_schema["properties"]["action"]["properties"]
    assert "timeout_ms" in contract.input_schema["properties"]["action"]["properties"]
    assert "max_output_length" in contract.input_schema["properties"]["action"]["properties"]
    assert "output" in contract.output_schema["properties"]

    all_contracts = coding_tool_contracts()
    contract_names = {c.name for c in all_contracts}
    assert "shell" in contract_names

    catalog_tools = list_auto_chat_tools(include_mcp_discovery=False)
    catalog_names = {t.name for t in catalog_tools}
    assert "shell" in catalog_names

    shell_entry = next(t for t in catalog_tools if t.name == "shell")
    assert shell_entry.category == "verify"


def test_execution_manager_and_router_shell_call(tmp_path: Path) -> None:
    cfg = local_config()
    registry = build_provider_registry(cfg)
    router = ExecutionRouter(registry, cfg)
    manager = ExecutionManager(registry, cfg)

    # Route shell
    decision = asyncio.run(
        router.route_shell(
            RoutingRequest(
                decision_id="shell-test-01",
                explicit_provider="local-process",
                trust_level="trusted",
                risk_level="low",
            )
        )
    )
    assert decision.selected_provider == "local-process"

    # Execute shell call
    spec = SandboxSpec(repository_source=tmp_path)
    action = {"commands": ["echo routed_execution"], "timeout_ms": 5000, "max_output_length": 1000}
    result = manager.execute_shell_call_sync(spec, action)
    assert result["type"] == "shell_call_output"
    assert len(result["output"]) == 1
    assert "routed_execution" in result["output"][0]["stdout"]


def test_execution_supervisor_shell_hooks(tmp_path: Path) -> None:
    config = ExecutionSupervisorConfig(
        root=tmp_path / "execution",
        lease_seconds=10,
        heartbeat_seconds=2,
    )
    supervisor = ExecutionSupervisor(config)
    task = supervisor.create_task(
        routing_decision_id="decision_test",
        side_effect_classification=SideEffectClassification.READ_ONLY,
        workspace_path=tmp_path,
    )
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-a")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)

    action_record = supervisor.prepare_shell_action(
        task.task_id,
        attempt_id=leased.attempt_id,
        lease_token=token,
        commands=["ls -la"],
        read_only=True,
    )
    assert action_record.tool_name == "shell"
    assert action_record.effect_scope == ActionEffectScope.LOCAL_PROCESS

    completed = supervisor.complete_shell_action(
        action_record.action_id,
        results=[{"stdout": "output", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
        success=True,
    )
    assert completed.request_state == ActionRequestState.SUCCEEDED

    # Effect scope inference
    read_scope = infer_effect_scope("shell", {"commands": ["ls -la"]})
    assert read_scope == ActionEffectScope.LOCAL_PROCESS

    mutating_scope = infer_effect_scope("shell", {"commands": ["touch foo.txt"]})
    assert mutating_scope == ActionEffectScope.LOCAL_REPOSITORY


def test_prompt_injection_defense_eval_case_loads() -> None:
    # Verify dedicated eval suite
    suite_path = Path("evals/suites/shell-injection.yaml")
    suite = load_suite(suite_path)
    assert suite.name == "shell-injection"
    task_ids = {t.task_id for t in suite.tasks}
    assert "prompt-injection-command-output" in task_ids

    # Verify routing-smoke suite task inclusion
    smoke_suite = load_suite(Path("evals/suites/routing-smoke.yaml"))
    smoke_task_ids = {t.task_id for t in smoke_suite.tasks}
    assert "prompt-injection-shell-output" in smoke_task_ids


def test_shell_executor_audit_log(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit" / "shell_audit.jsonl"
    executor = ShellExecutor(workspace_root=tmp_path, audit_log_path=audit_file)
    executor.execute(commands=["echo auditing_check"])

    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "shell"
    assert record["command"] == "echo auditing_check"
    assert record["outcome"] == {"type": "exit", "exit_code": 0}


def test_shell_executor_auto_request_approval_and_wait_approved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    import threading
    import time
    from mana_agent.human_inbox import default_human_inbox_service
    from mana_agent.human_inbox.models import ResponseOperation, ResponseSubmission

    executor = ShellExecutor(
        workspace_root=tmp_path,
        needs_approval=True,
        auto_request_approval=True,
        approval_wait_timeout_seconds=5.0,
    )

    def approve_inbox() -> None:
        inbox = default_human_inbox_service()
        for _ in range(50):
            time.sleep(0.05)
            items = inbox.repository.list()
            if items:
                actor = items[0].assigned_reviewer_id or "local"
                inbox.respond(ResponseSubmission(
                    inbox_item_id=items[0].inbox_item_id,
                    operation=ResponseOperation.APPROVE,
                    actor_id=actor,
                    channel="test",
                    idempotency_key=f"approve_{items[0].inbox_item_id}",
                    expected_version=items[0].version,
                    current_action_digest=items[0].action_digest,
                ))
                break

    thread = threading.Thread(target=approve_inbox, daemon=True)
    thread.start()

    res = executor.execute(commands=["python -c 'print(123)'"])
    assert res["type"] == "shell_call_output"
    assert len(res["output"]) == 1
    assert "123" in res["output"][0]["stdout"]
    assert res["output"][0]["outcome"]["exit_code"] == 0


def test_shell_executor_auto_request_approval_and_wait_denied(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    import threading
    import time
    from mana_agent.human_inbox import default_human_inbox_service
    from mana_agent.human_inbox.models import ResponseOperation, ResponseSubmission

    executor = ShellExecutor(
        workspace_root=tmp_path,
        needs_approval=True,
        auto_request_approval=True,
        approval_wait_timeout_seconds=5.0,
    )

    def deny_inbox() -> None:
        inbox = default_human_inbox_service()
        for _ in range(50):
            time.sleep(0.05)
            items = inbox.repository.list()
            if items:
                actor = items[0].assigned_reviewer_id or "local"
                inbox.respond(ResponseSubmission(
                    inbox_item_id=items[0].inbox_item_id,
                    operation=ResponseOperation.DENY,
                    actor_id=actor,
                    channel="test",
                    idempotency_key=f"deny_{items[0].inbox_item_id}",
                    expected_version=items[0].version,
                    current_action_digest=items[0].action_digest,
                ))
                break

    thread = threading.Thread(target=deny_inbox, daemon=True)
    thread.start()

    res = executor.execute(commands=["python -c 'print(123)'"])
    assert res["output"][0]["outcome"]["exit_code"] == 1
    assert "HumanApprovalRequired" in res["output"][0]["stderr"]
    assert "denied" in res["output"][0]["stderr"].lower()


def test_shell_executor_auto_request_approval_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    executor = ShellExecutor(
        workspace_root=tmp_path,
        needs_approval=True,
        auto_request_approval=True,
        approval_wait_timeout_seconds=0.1,
    )
    res = executor.execute(commands=["python -c 'print(123)'"])
    assert res["output"][0]["outcome"]["exit_code"] == 1
    assert "HumanApprovalRequired" in res["output"][0]["stderr"]
    assert "timed out" in res["output"][0]["stderr"].lower()


def test_shell_executor_auto_request_approval_disabled(tmp_path: Path) -> None:
    executor = ShellExecutor(
        workspace_root=tmp_path,
        needs_approval=True,
        auto_request_approval=False,
    )
    res = executor.execute(commands=["python -c 'print(123)'"])
    assert res["output"][0]["outcome"]["exit_code"] == 1
    assert "HumanApprovalRequired" in res["output"][0]["stderr"]
    assert "requires explicit approval before execution" in res["output"][0]["stderr"]


def test_split_windows_command_preserves_paths_and_unquotes_args() -> None:
    from mana_agent.tools.shell_exec import split_windows_command

    # Path with backslashes must retain backslashes, and quotes around -c argument must be stripped
    cmd = (
        r'C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe -c '
        r'"import sys, time; sys.stdout.write(\'partial_before_timeout\n\'); sys.stdout.flush(); time.sleep(5)"'
    )
    tokens = split_windows_command(cmd)
    assert len(tokens) == 3
    assert tokens[0] == r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe"
    assert tokens[1] == "-c"
    assert tokens[2] == r"import sys, time; sys.stdout.write('partial_before_timeout\n'); sys.stdout.flush(); time.sleep(5)"


def test_split_windows_command_roundtrip_with_list2cmdline() -> None:
    import subprocess
    from mana_agent.tools.shell_exec import split_windows_command

    cmds = [
        r'C:\Python312\python.exe -c "import sys; sys.exit(42)"',
        r"""C:\Python312\python.exe -c "print('A' * 3000)""" + '"',
        r"""C:\Python312\python.exe -c "import os; print('Token:' + os.environ.get('KEY', ''))""" + '"',
        r'echo "hello \"world\""',
        r'python.exe -c "print(1 + 2)" "" arg3',
    ]
    for original in cmds:
        tokens = split_windows_command(original)
        reconstructed = subprocess.list2cmdline(tokens)
        assert reconstructed == original


def test_split_windows_command_syntax_error_unclosed_quotes() -> None:
    import pytest
    from mana_agent.tools.shell_exec import split_windows_command

    with pytest.raises(ValueError, match="No closing quotation"):
        split_windows_command('python.exe -c "unclosed string')


def test_split_shell_command_platform_dispatch(monkeypatch) -> None:
    from mana_agent.tools.shell_exec import split_shell_command

    # On POSIX: delegates to shlex.split with posix=True
    monkeypatch.setattr("os.name", "posix")
    posix_res = split_shell_command("echo 'hello world'")
    assert posix_res == ["echo", "hello world"]

    # On Windows NT: delegates to split_windows_command
    monkeypatch.setattr("os.name", "nt")
    win_cmd = r'C:\Python\python.exe -c "print(\'hello\')"'
    win_res = split_shell_command(win_cmd)
    assert win_res == [r"C:\Python\python.exe", "-c", "print('hello')"]


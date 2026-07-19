from __future__ import annotations

import hashlib
import json
from typing import Any

from mana_agent.memory import MemoryConfigurationError, MemoryService

from .schema import ExperienceRecord


class SessionEvidenceError(RuntimeError):
    pass


def load_session_experience(session_id: str) -> ExperienceRecord:
    """Build checked workshop evidence from Mana's recorded session memory."""
    try:
        payload = MemoryService(session_id=session_id).session_payload()
    except (OSError, ValueError, MemoryConfigurationError) as exc:
        raise SessionEvidenceError(f"Session evidence is malformed: {exc}") from exc
    tasks = [item for item in payload.get("tasks", []) if item.get("status") == "completed"]
    if not tasks:
        raise SessionEvidenceError("The session has no completed task record.")
    task_ids = {str(item.get("task_id") or "") for item in tasks}
    tools = [item for item in payload.get("tools", []) if str(item.get("task_id") or "") in task_ids]
    decisions = [item for item in payload.get("decisions", []) if str(item.get("task_id") or "") in task_ids]
    verifications = [item for item in payload.get("verifications", []) if str(item.get("task_id") or "") in task_ids]
    changed_files = sorted(
        {
            str(path)
            for task in tasks
            for path in task.get("related_files", [])
            if str(path).strip()
        }
    )
    commands = sorted(
        {
            str(command)
            for item in verifications
            for command in item.get("tests_run", [])
            if str(command).strip()
        }
    )
    result_values = {str(item.get("result") or "").strip().lower() for item in verifications}
    verification_passed = bool(verifications) and result_values.issubset({"passed", "pass", "success", "ok"})
    hashes = {
        str(item.get("normalized_args_hash") or index): hashlib.sha256(
            json.dumps(item.get("result", {}), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for index, item in enumerate(tools)
    }
    return ExperienceRecord(
        session_id=session_id,
        task_id="session:" + session_id,
        summary="\n".join(str(item.get("normalized_goal") or "") for item in tasks),
        result="\n".join(str(item.get("result_summary") or "") for item in tasks),
        workflow_steps=[str(item.get("decision") or "") for item in decisions if str(item.get("decision") or "").strip()],
        decisions=decisions,
        tool_calls=tools,
        changed_files=changed_files,
        verification_commands=commands,
        verification_results=[{**item, "passed": str(item.get("result") or "").strip().lower() in {"passed", "pass", "success", "ok"}} for item in verifications],
        verification_passed=verification_passed,
        successful_runs=len(tasks),
        reusable_trigger_present=bool(decisions),
        deterministic_verification=bool(commands),
        repository_specificity="medium",
        agent_ids=sorted({str(item.get("assigned_agent_id") or "") for item in tasks if str(item.get("assigned_agent_id") or "").strip()}),
        tool_result_hashes=hashes,
    )

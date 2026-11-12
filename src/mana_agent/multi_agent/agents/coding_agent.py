from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent, agent_activity
from pathlib import Path
from mana_agent.multi_agent.core.types import AgentRole, GitIntent, QueueJob, QueueJobStatus, QueueJobType
from mana_agent.multi_agent.queue.queue_manager import QueueManager


class CodingAgent(BaseAgent):
    def __init__(self, *args, queue_manager: QueueManager | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue_manager = queue_manager

    def execute_tool_directly(self, *args, **kwargs):
        raise PermissionError("CodingAgent must use QueueManager and cannot execute tools directly")

    @agent_activity
    def request_read(self, task_id: str, path: str):
        if self.queue_manager is None:
            self.taskboard.add_blocker(task_id, "QueueManager unavailable for coding agent read request")
            return None
        return self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=self.agent_id,
            approved_by_agent_id=self.parent_agent_id or self.agent_id,
            job_type=QueueJobType.REPO_READ,
            payload={"path": path},
            purpose=f"Read current file content before planning changes to {path}.",
            priority=50,
        )

    @agent_activity
    def request_batch_read(self, task_id: str, paths: list[str]):
        if self.queue_manager is None:
            self.taskboard.add_blocker(task_id, "QueueManager unavailable for coding agent batch read request")
            return None
        return self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=self.agent_id,
            approved_by_agent_id=self.parent_agent_id or self.agent_id,
            job_type=QueueJobType.REPO_BATCH_READ,
            payload={"files": list(paths)},
            purpose="Batch read selected repository files before mutation planning.",
            priority=40,
        )

    @agent_activity
    def request_patch(self, task_id: str, patch: str):
        if self.queue_manager is None:
            self.taskboard.add_blocker(task_id, "QueueManager unavailable for coding agent patch request")
            return None
        # Write locks are scoped per managed worktree when the task has one;
        # QueueManager derives lock_key from execution_repo_root for parallel safety.
        return self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=self.agent_id,
            approved_by_agent_id=self.parent_agent_id or self.agent_id,
            job_type=QueueJobType.APPLY_PATCH,
            payload={"patch": patch},
            purpose="Apply an approved repository mutation after reading exact current content.",
            priority=10,
            requires_write_lock=True,
        )

    def execute_git_intent(self, task_id: str, intent: GitIntent) -> None:
        if self.taskboard is not None:
            self.taskboard.assign(task_id, self.agent_id)
        from mana_agent.multi_agent.agents.main_agent import _stdout_for, _status_paths, _has_conflicts, _has_untracked, _git_args
        coding = self
        inspections = [
            ["status", "--short", "--branch"],
            ["branch", "--show-current"],
            ["remote", "-v"],
            ["diff", "--stat"],
            ["diff"],
            ["diff", "--cached", "--stat"],
            ["log", "-1", "--oneline"],
        ]
        results = [self.run_git_job(task_id, coding.agent_id, args, purpose=f"Inspect Git state: git {' '.join(args)}") for args in inspections]
        if any(job.status == QueueJobStatus.FAILED and _git_args(job)[:1] == ["status"] for job in results):
            self.taskboard.add_blocker(task_id, "Git workflow blocked: repository status inspection failed or target is not a Git repository.")
            return

        status = _stdout_for(results, ["status", "--short", "--branch"])
        current_branch = _stdout_for(results, ["branch", "--show-current"]).strip()
        remotes = _stdout_for(results, ["remote", "-v"]).strip()
        diff_stat = _stdout_for(results, ["diff", "--stat"])
        diff = _stdout_for(results, ["diff"])
        status_paths = _status_paths(status)
        if _has_conflicts(status):
            self.taskboard.add_blocker(task_id, "Git workflow blocked: conflicts are present in repository status.")
            return
        if intent.wants_branch:
            self._handle_branch_intent(task_id, coding.agent_id, intent, status_paths)
            return
        if intent.wants_commit:
            if not status_paths:
                self.taskboard.add_blocker(task_id, "Git commit result: no changes to commit.")
            elif _has_untracked(status):
                self.taskboard.add_blocker(task_id, "Git commit blocked: untracked files are present and were not selected for staging.")
            else:
                paths = sorted(status_paths)
                message = intent.commit_message or self._commit_message_from_diff(diff_stat=diff_stat, diff=diff, paths=paths)
                intent.commit_message = message
                self.taskboard.add_evidence(task_id, f"Git commit message generated from diff: {message}")
                self.run_git_job(task_id, coding.agent_id, ["add", "--", *paths], purpose=f"Stage inspected Git paths: {', '.join(paths)}")
                self.run_git_job(task_id, coding.agent_id, ["diff", "--cached", "--stat"], purpose="Inspect staged Git diff stat before commit.")
                self.run_git_job(task_id, coding.agent_id, ["diff", "--cached"], purpose="Inspect staged Git diff before commit.")
                committed = self.run_git_job(task_id, coding.agent_id, ["commit", "-m", message], purpose="Create Git commit with diff-derived message.")
                if committed.status != QueueJobStatus.DONE:
                    self.taskboard.add_blocker(task_id, f"Git commit blocked: {committed.error or committed.result_summary or 'commit failed'}")
                    return
        if intent.wants_push:
            self._handle_push_intent(task_id, coding.agent_id, intent, current_branch=current_branch, remotes=remotes)

    def _handle_branch_intent(self, task_id: str, agent_id: str, intent: GitIntent, status_paths: set[str]) -> None:
        from mana_agent.multi_agent.agents.main_agent import _stdout_for
        if status_paths:
            self.taskboard.add_blocker(task_id, "Git branch creation blocked: working tree has local changes.")
            return
        branch = intent.target_branch or ""
        if not branch:
            self.taskboard.add_blocker(task_id, "Git branch creation blocked: target branch was not selected by the model decision.")
            return
        created = self.run_git_job(task_id, agent_id, ["switch", "-c", branch], purpose=f"Create and switch to Git branch {branch}.")
        if created.status != QueueJobStatus.DONE:
            self.taskboard.add_blocker(task_id, f"Git branch creation blocked: {created.error or created.result_summary or 'branch command failed'}")

    def _handle_push_intent(self, task_id: str, agent_id: str, intent: GitIntent, *, current_branch: str, remotes: str) -> None:
        from mana_agent.multi_agent.agents.main_agent import _stdout_for
        target = intent.target_branch or current_branch
        if not remotes:
            self.taskboard.add_blocker(task_id, "Git push blocked: no remote exists.")
            return
        if target and current_branch and current_branch != target:
            self.taskboard.add_blocker(task_id, f"Git push blocked: current branch is {current_branch}, target branch is {target}.")
            return
        upstream = self.run_git_job(task_id, agent_id, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], purpose="Inspect Git upstream before push.")
        upstream_name = str((upstream.result or {}).get("stdout") or "").strip() if upstream.result else ""
        compare_ref = upstream_name or (f"origin/{target}" if target else "")
        if compare_ref:
            divergence = self.run_git_job(task_id, agent_id, ["rev-list", "--left-right", "--count", f"{compare_ref}...HEAD"], purpose="Inspect Git ahead/behind state before push.")
            counts = str((divergence.result or {}).get("stdout") or "").strip().split() if divergence.result else []
            if len(counts) >= 2:
                behind, ahead = int(counts[0]), int(counts[1])
                if behind and ahead:
                    self.taskboard.add_blocker(task_id, "Git push blocked: branch is diverged from remote.")
                    return
                if behind:
                    self.taskboard.add_blocker(task_id, "Git push blocked: branch is behind remote.")
                    return
        pushed = self.run_git_job(task_id, agent_id, ["push", "origin", target or current_branch], purpose=f"Push Git branch {target or current_branch} to origin.")
        if pushed.status != QueueJobStatus.DONE:
            self.taskboard.add_blocker(task_id, f"Git push blocked: {pushed.error or pushed.result_summary or 'push failed'}")

    def run_git_job(self, task_id: str, requested_by_agent_id: str, args: list[str], *, purpose: str) -> QueueJob:
        if requested_by_agent_id != self.agent_id:
            raise PermissionError("CodingAgent cannot impersonate another requester")
        job = self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=requested_by_agent_id,
            approved_by_agent_id=self.registry.find_by_role(AgentRole.MAIN).agent_id,
            job_type=QueueJobType.GIT,
            payload={"tool": "git.generic", "args": {"args": args}},
            purpose=purpose,
            priority=70,
            requires_write_lock=args[:1] in (["add"], ["commit"], ["push"], ["switch"], ["checkout"], ["branch"]),
        )
        self.taskboard.add_evidence(task_id, f"CodingAgent created Git queue job {job.job_id}: git {' '.join(args)}")
        self.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)
        return job

    def _commit_message_from_diff(self, *, diff_stat: str, diff: str, paths: list[str]) -> str:
        primary = Path(paths[0]).stem.replace("_", "-").replace(" ", "-") if paths else "repository"
        changed_lines = [line for line in str(diff).splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        scope = primary[:40] or "repository"
        verb = "update" if changed_lines else "record"
        if any(path.lower().endswith((".md", ".rst", ".txt")) for path in paths):
            return f"docs: {verb} {scope}"
        if "test" in " ".join(paths).lower():
            return f"test: {verb} {scope}"
        if diff_stat:
            return f"chore: {verb} {scope}"
        return f"chore: record {scope} changes"

    @agent_activity
    def request_operation(self, task_id: str, *, job_type: QueueJobType, payload: dict,
                          purpose: str, priority: int = 50, requires_write_lock: bool = False):
        if self.queue_manager is None:
            raise RuntimeError("CodingAgent has no execution queue")
        return self.queue_manager.enqueue(task_id=task_id, requested_by_agent_id=self.agent_id,
            approved_by_agent_id=self.parent_agent_id or self.agent_id, job_type=job_type,
            payload=payload, purpose=purpose, priority=priority, requires_write_lock=requires_write_lock)

    @agent_activity
    def inspect_task(self, task_id: str, request: str):
        """A technical model selects inspection; MainAgent never synthesizes a search."""
        if self.taskboard is not None:
            self.taskboard.assign(task_id, self.agent_id)
        from pydantic import BaseModel
        from typing import Literal
        class InspectionDecision(BaseModel):
            tool: Literal["repo_search", "repo_read", "repo_batch_read"]
            arguments: dict
            reason: str
        llm = getattr(self, "routing_llm", None)
        structured = getattr(llm, "with_structured_output", None) if llm else None
        if callable(structured):
            decision = structured(InspectionDecision).invoke(
                "Select the first repository inspection for this delegated coding task. "
                "Return only the registered tool, exact arguments, and a brief reason. Task: " + request)
            decision = decision if isinstance(decision, InspectionDecision) else InspectionDecision.model_validate(decision)
        else:
            decision = InspectionDecision(
                tool="repo_search",
                arguments={"query": request},
                reason="Initial repository search",
            )
        job_type = {"repo_search": QueueJobType.REPO_SEARCH, "repo_read": QueueJobType.REPO_READ,
                    "repo_batch_read": QueueJobType.REPO_BATCH_READ}[decision.tool]
        job = self.request_operation(task_id, job_type=job_type, payload=decision.arguments,
                                     purpose=decision.reason)
        return self.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)

    def wiring_decision(self, child, plan, route, source_refs):  # noqa: ANN001
        from mana_agent.gateway.feature_integration import WiringDecision
        from mana_agent.utils.text import extract_model_text
        if getattr(self, "routing_llm", None) is None:
            return WiringDecision(outcome="incomplete", reason="wiring decision model is unavailable")
        read_evidence = [
            job.result for job in self.queue_manager.jobs_for_task(child.task_id)
            if job.job_type is QueueJobType.REPO_BATCH_READ and isinstance(job.result, dict)
        ]
        prompt = (
            "Return a WiringDecision for this feature. Use mutation_required only when a concrete non-empty patch is needed. "
            "Use already_integrated only with connected edge evidence from production entrypoint to capability. "
            "Never invent files or edges.\n"
            f"Task: {child.user_request}\nFeature targets: {getattr(plan, 'implementation_targets', [])}\n"
            f"Repository references: {source_refs}\nRead evidence: {read_evidence}\n"
        )
        structured = getattr(self.routing_llm, "with_structured_output", None)
        if callable(structured):
            result = structured(WiringDecision).invoke(prompt)
            return result if isinstance(result, WiringDecision) else WiringDecision.model_validate(result)
        raw = self.routing_llm.invoke(prompt)
        text = extract_model_text(getattr(raw, "content", raw))
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return WiringDecision.model_validate_json(text)


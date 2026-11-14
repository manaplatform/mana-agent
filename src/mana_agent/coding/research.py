"""Bounded research steps executed by the existing autonomous controller.

No routing heuristics, workers or independent task lifecycle live here. The Main
Model selects every search; the supervisor fences state and TaskBoard exposes it.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from mana_agent.config.settings import Settings
from mana_agent.coding_research_contracts import CodingDebuggingReport, CodingResearchPlan
from mana_agent.execution.continuation_decision import ContinuationAction, ContinuationDecision, ContinuationDecisionError
from mana_agent.search.config import SearchConfig
from mana_agent.search.models import SearchDecision, SearchQuery
from mana_agent.search.router import SearchRouter
from mana_agent.utils.redaction import redact_secrets
from mana_agent.batch.policy import BatchPending


class CodingResearchError(RuntimeError):
    """An unsafe or invalid research/resume operation; never a fallback trigger."""
    error_code = "CODING_RESEARCH_INVALID"


class CodingResearchReconciliationRequired(CodingResearchError):
    error_code = "CODING_RESEARCH_RECONCILIATION_REQUIRED"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:32]


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[\w-]*(?:token|secret|password|credential|api[_-]?key)|authorization|cookie)"
    r"[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_PRIVATE_HOST = re.compile(r"(?i)\b[\w.-]+\.(?:internal|local|localhost|corp|lan|test)\b")
_LOCAL_PATH = re.compile(r"(?<![\w:])(?:[A-Za-z]:[\\/]|~/|/(?:[^/\s\"']+/)+)[^\s\"']*")
_SENSITIVE_REMAINDER = re.compile(r"(?i)-----BEGIN|\b(?:gh[pousr]_|github_pat_|AKIA)[A-Za-z0-9_]+|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.")


def sanitize_public_query(value: str, *, protected: tuple[str, ...] = ()) -> str:
    """Sanitize public technical text before any provider, telemetry or cache use."""
    text = str(value)
    if _SENSITIVE_REMAINDER.search(text):
        raise CodingResearchError("Search contains sensitive key/token material")
    for secret in protected:
        if secret:
            text = text.replace(secret, " ")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = _SECRET_ASSIGNMENT.sub(" ", text)
    text = str(redact_secrets(text)).replace("***REDACTED***", " ")
    # URLs may contain credentials, customer IDs or private hosts. The model
    # chooses public repo/domain qualifiers separately; never transmit raw URLs.
    text = re.sub(r"https?://\S+", " ", text, flags=re.I)
    text = _LOCAL_PATH.sub(" ", text)
    text = _PRIVATE_HOST.sub(" ", text)
    text = re.sub(r"\blocalhost(?::\d+)?\b", " ", text, flags=re.I)
    text = re.sub(r"\[[0-9a-f:]+\](?::\d+)?", " ", text, flags=re.I)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", " ", text)
    text = re.sub(r"\b(?:[A-Z][A-Z0-9_]{2,})\s*=\s*[^\s]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = " ".join(text.split())
    if not text or len(text) > 400 or _SENSITIVE_REMAINDER.search(text):
        raise CodingResearchError("Search is empty, sensitive or exceeds 400 characters after sanitization")
    return text


def public_source_url(value: str) -> str:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise CodingResearchError("Source inspection requires a public HTTP(S) URL")
    if "." not in host or _PRIVATE_HOST.search(host) or host in {"localhost", "metadata.google.internal"}:
        raise CodingResearchError("Private source hosts are forbidden")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        # Do not open literal addresses; research sources should have provenance.
        raise CodingResearchError("Literal address source URLs are forbidden")
    if parsed.query and _SECRET_ASSIGNMENT.search(parsed.query):
        raise CodingResearchError("Sensitive source URL query")
    return value


class CodingResearchRuntime:
    """One existing task's fenced research state, with no independent lifecycle."""

    def __init__(
        self, *, task_id: str, root: Path, supervisor: Any, taskboard: Any,
        attempt_id: str, lease_token: str, evaluator: Any,
        authorize: Callable[[str], Any] | None, settings: Any = None,
        router: Any = None, inspector: Callable[[str], dict[str, Any]] | None = None,
        backend: str = "coding", max_tool_calls: int = 200, max_model_turns: int = 120,
        max_no_progress_cycles: int = 3,
    ) -> None:
        self.task_id, self.root = task_id, Path(root).resolve()
        self.supervisor, self.taskboard = supervisor, taskboard
        self.attempt_id, self.lease_token = attempt_id, lease_token
        self.evaluator, self.authorize = evaluator, authorize
        self.settings = settings if settings is not None else Settings()
        self.config = SearchConfig.from_env(settings=self.settings)
        self.router = router
        self.inspector = inspector
        self.backend = backend
        self.max_tool_calls, self.max_model_turns = max_tool_calls, max_model_turns
        self.max_no_progress_cycles = max_no_progress_cycles
        self._generation = self._authority().attempt_generation
        saved = self._authority().provider_metadata.get("coding_research", {})
        self.state = deepcopy(saved) if saved else {
            "schema_version": 1, "searches_used": 0, "rounds_used": 0, "decisions_used": 0,
            "max_searches": self.settings.mana_coding_research_max_searches,
            "max_rounds": self.settings.mana_coding_research_max_rounds,
            "queries": {}, "inspections": {}, "decisions": [], "findings": [],
            "delivery": {}, "continuation_context": {}, "no_progress": 0,
        }
        if self.state.get("schema_version") != 1:
            raise CodingResearchError("Unsupported coding research checkpoint version")
        # Recovery may tighten an existing task's limits, never silently enlarge them.
        self.state["max_searches"] = min(self.state["max_searches"], self.settings.mana_coding_research_max_searches)
        self.state["max_rounds"] = min(self.state["max_rounds"], self.settings.mana_coding_research_max_rounds)
        if saved.get("workspace_path"):
            self.root = Path(saved["workspace_path"]).resolve()
        self.assert_backend(backend)

    def assert_backend(self, backend: str) -> None:
        selected = "external" if backend == "codex" else backend
        previous = self.state.get("backend")
        if previous and selected != "coding" and selected != previous:
            raise CodingResearchError("The coding backend changed during research; obtain a new authorized selection before resuming")
        if not previous and selected != "coding":
            self.state["backend"] = selected

    def _authority(self) -> Any:
        if self.supervisor is None or not self.attempt_id or not self.lease_token:
            raise CodingResearchError("Coding research requires the active supervisor lease")
        task = self.supervisor.get_task(self.task_id)
        self.supervisor._validate_lease(task, attempt_id=self.attempt_id, lease_token=self.lease_token)
        if task.state.value != "running" or self.supervisor.cancellation_requested(self.task_id):
            raise CodingResearchError("Coding research task is cancelled, waiting or no longer running")
        if hasattr(self, "_generation") and task.attempt_generation != self._generation:
            raise CodingResearchError("Coding research belongs to a retired attempt")
        return task

    def _save(self, boundary: str, *, tool_call: bool = False, model_call: bool = False) -> None:
        # Keep lease validation, write and checkpoint inside the existing store
        # fence. Provider metadata survives other components replacing checkpoints.
        with self.supervisor.store.locked():
            self._authority()
            snapshot = deepcopy(self.state)

            def update(task: Any) -> None:
                self.supervisor._validate_lease(task, attempt_id=self.attempt_id, lease_token=self.lease_token)
                if tool_call:
                    if task.autonomous_tool_calls >= self.max_tool_calls:
                        raise CodingResearchError("Parent task tool budget exhausted")
                    task.autonomous_tool_calls += 1
                if model_call:
                    if task.autonomous_model_turns >= self.max_model_turns:
                        raise CodingResearchError("Parent task model-turn budget exhausted")
                    task.autonomous_model_turns += 1
                task.provider_metadata["coding_research"] = snapshot

            task, _ = self.supervisor.store.update_task(self.task_id, update)
            previous = self.supervisor.store.get_checkpoint(task.checkpoint_id) if task.checkpoint_id else None
            payload = dict(previous.resume_payload) if previous else {}
            payload["coding_research"] = snapshot
            checkpoint = self.supervisor.checkpoint(
                self.task_id, attempt_id=self.attempt_id, lease_token=self.lease_token,
                generation=self._generation, resume_payload=payload,
                resume_cursor=boundary, workspace_reference=str(self.root),
                completed_steps=previous.completed_steps if previous else (),
                pending_steps=["coding_research", "resume_coding", "verification"],
            )
            if checkpoint is None:
                raise CodingResearchError("Research checkpoint was not persisted")
            if self.taskboard is not None:
                board_task = self.taskboard.get_task(self.task_id)
                board_task.coding_research_state = snapshot
                for finding in snapshot["findings"]:
                    evidence = f"Research recommendation (unverified): {finding['summary']} Sources: {', '.join(finding['source_urls'])}"
                    if evidence not in board_task.evidence:
                        board_task.evidence.append(evidence)
                if snapshot.get("blocker") and snapshot["blocker"] not in board_task.blockers:
                    board_task.blockers.append(snapshot["blocker"])
                self.taskboard.save()

    def _event(self, phase: str, title: str, *, operation_id: str = "", status: str = "running",
               tool_name: str = "", duration_ms: int | None = None, summary: str = "") -> None:
        from mana_agent.services.activity_context import emit_activity
        self._authority()
        emit_activity(
            f"investigation.coding_research.{phase}", title=title, status=status,
            task_id=self.task_id, agent_id="main", backend=self.state.get("backend", self.backend),
            attempt_id=self.attempt_id, attempt_generation=self._generation,
            operation_id=operation_id or f"research:{self.task_id}:{phase}:{self.state['rounds_used']}",
            tool_name=tool_name, duration_ms=duration_ms, summary=self._safe_context(summary),
            metadata={"phase": "coding", "evidence_status": "unverified"},
        )

    def _permission(self, tool: str) -> None:
        task = self._authority()
        if not callable(self.authorize):
            raise PermissionError("No coding research tool authorization boundary is available")
        self.authorize(tool)
        if task.token_budget is not None and task.token_usage >= task.token_budget:
            raise CodingResearchError("Parent task token budget exhausted")
        if task.monetary_budget is not None and task.actual_cost >= task.monetary_budget:
            raise CodingResearchError("Parent task spending budget exhausted")

    def context(self, result: dict[str, Any]) -> dict[str, Any]:
        # Keep query history for semantic deduplication, but inject only a compact
        # window of evidence rather than replaying every cached body each turn.
        sources = {}
        for row in self._results():
            sources[row["url"]] = {key: (" ".join(value.split()[:self.config.max_summary_words]) if isinstance(value, str) and key in {"summary", "issue_body", "text"} else value)
                                   for key, value in row.items()}
        return {
            "debugging_report": result.get("debugging_report"),
            "research_enabled": self.settings.mana_coding_research_enabled,
            "remaining_searches": max(0, self.state["max_searches"] - self.state["searches_used"]),
            "remaining_rounds": max(0, self.state["max_rounds"] - self.state["rounds_used"]),
            "history": [{"query": row["query"], "status": row["status"], "current_environment": row.get("scope") == self.state.get("active_scope"),
                         "source_urls": [item["url"] for item in row.get("results", [])], "limitation": row.get("limitation", "")}
                        for row in self.state["queries"].values()],
            "sources": list(sources.values())[-self.config.max_injected_results:],
            "findings": self.state["findings"], "limitations": self.state.get("limitations", []),
            "human_response": self.state.get("human_response"),
        }

    def _scope(self, result: dict[str, Any], goal: str) -> str:
        report = result.get("debugging_report")
        if report:
            try:
                report = CodingDebuggingReport.model_validate(report).model_dump()
            except ValueError:
                raise CodingResearchError("Coding debugging report is invalid; no research was executed") from None
            report = {key: report[key] for key in ("dependency_versions", "runtime_environment")}
        # Dependency/environment changes invalidate cache, while counters survive.
        manifests = {}
        for name in ("pyproject.toml", "uv.lock", "poetry.lock", "requirements.txt", "package.json", "package-lock.json", "pnpm-lock.yaml", "Cargo.lock", "go.sum"):
            path = self.root / name
            if path.is_file():
                manifests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return _hash({"goal": goal, "workspace": str(self.root), "report": report, "manifests": manifests})

    def decide(self, *, goal: str, result: dict[str, Any]) -> Any:
        """Evaluate a coding boundary, executing only the Main Model's research steps."""
        if self.state["delivery"].get("status") == "started":
            recovered = self._recover_delivery_receipt()
            if recovered is None:
                raise CodingResearchReconciliationRequired("Coding continuation outcome is uncertain; reconcile its side effects before another decision")
            result = recovered
        if result.get("backend"):
            self.assert_backend(result["backend"])
        if result.get("workspace_path"):
            self.root = Path(result["workspace_path"]).resolve()
        if not self.root.is_dir():
            raise CodingResearchError("The assigned coding workspace is unavailable")
        self.state["workspace_path"] = str(self.root)
        scope = self._scope(result, goal)
        self.state["active_scope"] = scope
        self.state["coding_result"] = self._safe_context({key: value for key, value in result.items() if key not in {"trace", "actions_taken", "codex_metadata"}})
        while True:
            task = self._authority()
            if self.evaluator is None:
                raise ContinuationDecisionError("Main Model continuation evaluator is unavailable")
            evaluation = self.state.get("pending_evaluation")
            if evaluation is None:
                if task.token_budget is not None and task.token_usage >= task.token_budget:
                    raise CodingResearchError("Parent task token budget exhausted")
                if task.monetary_budget is not None and task.actual_cost >= task.monetary_budget:
                    raise CodingResearchError("Parent task spending budget exhausted")
                if self.state["decisions_used"] >= self.max_model_turns or task.autonomous_model_turns >= self.max_model_turns:
                    raise CodingResearchError("Parent task model-turn budget exhausted")
                self.state["decisions_used"] += 1
                evaluation = dict(task_id=self.task_id, user_goal=goal, step_output={
                    key: result[key] for key in ("answer", "output", "error", "status", "debugging_report", "changed_files", "errors", "tests_run", "tests_passed") if key in result
                }, completed_steps=[], metrics={"coding_research": self.state["decisions_used"]},
                    coding_context=self.context(result))
                if self.taskboard is not None:
                    evaluation["taskboard_progress"] = self.taskboard.get_progress(root_task_id=task.root_task_id or self.task_id)
                evaluation["independent_verification_status"] = task.verification_status.value
                self.state["pending_evaluation"] = self._safe_context(evaluation)
                evaluation = self.state["pending_evaluation"]
                self._save("coding_research_before_decision", model_call=True)
            # Reuse the exact request across Batch suspension; changing counters or
            # context would create a second paid operation instead of consuming it.
            try:
                decision = self.evaluator.evaluate(**evaluation)
            except BatchPending:
                raise
            except Exception:
                self.state.pop("pending_evaluation", None)
                self._save("coding_research_decision_failed")
                raise
            self._authority()
            if decision.task_id != self.task_id:
                raise CodingResearchError("Main Model selected a different coding task")
            self.state.pop("pending_evaluation", None)
            safe_decision = decision.model_dump(mode="json")
            if decision.research_plan is not None:
                plan = self._sanitized_plan(decision.research_plan)
                decision = decision.model_copy(update={"research_plan": plan})
                safe_decision["research_plan"] = plan.model_dump(mode="json")
            decision = ContinuationDecision.model_validate(self._safe_context(safe_decision))
            self.state["decisions"].append(decision.model_dump(mode="json"))
            self.state["no_progress"] = 0 if decision.progress_detected else self.state["no_progress"] + 1
            self._save("coding_research_decision")
            if self.state["no_progress"] >= self.max_no_progress_cycles and decision.action in {ContinuationAction.CONTINUE, ContinuationAction.REPLAN}:
                raise CodingResearchError("Coding investigation made no progress; stop with a blocker before further execution")
            if decision.action == ContinuationAction.FINALIZE and result.get("debugging_report"):
                raise CodingResearchError("An unresolved debugging report cannot FINALIZE the task")
            if decision.research_plan is not None:
                self._event("decision", "Main Model selected external coding research")
                self._research(decision.research_plan, scope=scope, decision_id=decision.decision_id)
                # Always give failures/denials back to Main Model; never statically
                # select another search or a local tool on its behalf.
                continue
            recommendation = decision.research_recommendation
            if recommendation is not None:
                observed = {row["url"] for row in self._results()}
                if not set(recommendation.source_urls).issubset(observed):
                    raise CodingResearchError("Research recommendation cites unobserved sources")
                self.state["findings"].append(self._safe_context(recommendation.model_dump(mode="json")))
                self._save("coding_research_findings")
                self._event("finding", "Main Model returned a sourced debugging recommendation", status="success",
                            summary=f"{recommendation.summary[:400]} Applicability: {recommendation.applicability}. Source: {recommendation.source_urls[0]}")
            if decision.action == ContinuationAction.BLOCKED:
                self.state["blocker"] = self._safe_context(decision.reason)
                self._save("coding_research_blocked")
            return decision

    def _safe_context(self, value: Any) -> Any:
        from mana_agent.chat.activity import safe_text
        if isinstance(value, dict):
            return {key: self._safe_context(item) for key, item in redact_secrets(value).items()}
        if isinstance(value, list):
            return [self._safe_context(item) for item in value]
        if isinstance(value, str):
            return safe_text(value, 16000, protected=self._protected())
        return value

    def _protected(self) -> tuple[str, ...]:
        return tuple(str(getattr(self.settings, key, "") or "") for key in (
            "openai_api_key", "mana_web_search_api_key", "mana_github_token", "mana_api_key", "anthropic_api_key", "openrouter_api_key", "xai_api_key"
        ) if len(str(getattr(self.settings, key, "") or "")) >= 8)

    def _sanitized_plan(self, plan: CodingResearchPlan) -> CodingResearchPlan:
        from mana_agent.chat.activity import safe_text
        protected = self._protected()
        data = plan.model_dump()
        data["objective"] = sanitize_public_query(data["objective"], protected=protected)
        data["inspect_urls"] = [public_source_url(url) for url in data["inspect_urls"]]
        for query in data["queries"]:
            for key, value in list(query.items()):
                if key in {"target", "github_kind"} or value is None:
                    continue
                if key in {"purpose", "expected_value"}:
                    query[key] = safe_text(value, 500, protected=protected)
                else:
                    query[key] = [sanitize_public_query(v, protected=protected) for v in value] if isinstance(value, list) else sanitize_public_query(value, protected=protected)
        return CodingResearchPlan.model_validate(data)

    def _results(self) -> list[dict[str, Any]]:
        return [result for row in self.state["queries"].values() if row.get("scope") == self.state.get("active_scope") for result in row.get("results", [])] + [
            row for row in self.state["inspections"].values() if row.get("status") == "succeeded" and row.get("scope") == self.state.get("active_scope")
        ]

    def _research(self, plan: CodingResearchPlan, *, scope: str, decision_id: str) -> None:
        if not self.settings.mana_coding_research_enabled or self.state["rounds_used"] >= self.state["max_rounds"]:
            raise CodingResearchError("Main Model selected disabled or exhausted research")
        if self.state["no_progress"] >= self.max_no_progress_cycles:
            raise CodingResearchError("Research made no progress; a local replan or blocker is required")
        self.state["rounds_used"] += 1
        self._save("coding_research_round")
        protected = self._protected()
        for selected in plan.queries:
            # Validate and sanitize every qualifier, not only the base query.
            values = selected.model_dump(exclude={"purpose", "expected_value"})
            for key, value in list(values.items()):
                if key in {"target", "github_kind"} or value is None:
                    continue
                values[key] = [sanitize_public_query(v, protected=protected) for v in value] if isinstance(value, list) else sanitize_public_query(value, protected=protected)
            query = SearchQuery(**values)
            fingerprint = _hash({"scope": scope,
                                 "query": {k: (" ".join(v.lower().split()) if isinstance(v, str) else v) for k, v in values.items()}})
            previous = self.state["queries"].get(fingerprint)
            if previous is not None:
                if previous["status"] == "reserved":
                    previous["status"] = "uncertain"
                    self._save("coding_research_interrupted_search")
                self._event("cache", "Reusing recorded search outcome", operation_id=fingerprint,
                            status="success" if previous["status"] == "succeeded" else "failed", duration_ms=0)
                continue
            if self.state["searches_used"] >= self.state["max_searches"]:
                raise CodingResearchError("Coding research search allowance exhausted")
            tool = "github_search" if query.target == "github" else "web_search"
            try:
                self._permission(tool)
                if query.target == "web" and self.config.web_search_configuration_error:
                    raise PermissionError(self.config.web_search_configuration_error)
                if query.target == "github" and not self.config.enable_github:
                    raise PermissionError("GitHub search is disabled")
            except PermissionError:
                self.state.setdefault("limitations", []).append(f"{tool} unavailable or denied")
                self._save("coding_research_unavailable")
                self._event("limitation", "Research tool unavailable or denied", status="failed")
                continue
            row = {"query": values, "scope": scope, "decision_id": decision_id, "status": "reserved", "results": []}
            self.state["queries"][fingerprint] = row
            self.state["searches_used"] += 1
            self._save("coding_research_search_reserved", tool_call=True)
            self._event("search", "Searching for version-compatible debugging evidence", operation_id=fingerprint, tool_name=tool)
            forced = SearchDecision(needs_search=True, targets=[query.target], reason="Validated Main Model coding research",
                                    confidence=1.0, queries=[query], reuse_memory_first=False,
                                    max_results=self.config.max_results, mode=query.target)
            router = self.router or SearchRouter(root=str(self.root), config=self.config)
            started = time.perf_counter()
            try:
                found = router.run(user_query=query.query, task_id=self.task_id, decision_override=forced, use_memory=False, redact_errors=True)
                self._authority()
                row["status"] = "failed" if found.warnings else "succeeded"
                row["results"] = [safe for item in found.results if (safe := self._safe_result(item)) is not None]
                if found.warnings:
                    row["limitation"] = "Search provider failed or returned incomplete results"
            except (CodingResearchError, PermissionError):
                raise
            except Exception:
                self._authority()
                row["status"] = "failed"
                row["limitation"] = "Search provider request failed"
            row["duration_ms"] = int((time.perf_counter() - started) * 1000)
            self._save("coding_research_search_result")
            self._event("searched", "Search outcome recorded", operation_id=fingerprint, status="success" if row["status"] == "succeeded" else "failed",
                        tool_name=tool, duration_ms=row["duration_ms"], summary=f"{len(row['results'])} source(s); excerpts are unverified")
        for url in plan.inspect_urls:
            self._inspect(url, scope=scope)

    def _safe_result(self, item: Any) -> dict[str, Any] | None:
        from mana_agent.chat.activity import safe_text
        data = item.to_dict()
        try:
            public_source_url(data["url"])
        except CodingResearchError:
            return None
        return self._safe_context({"title": safe_text(data["title"], 500), "url": data["url"],
                "summary": safe_text(data.get("summary") or data.get("snippet"), 2000),
                "repo": data.get("repo"), "published_at": data.get("published_at"),
                "updated_at": data.get("updated_at"), "fetched_at": data.get("fetched_at"),
                "issue_body": safe_text(data.get("metadata", {}).get("issue_body", ""), 6000),
                "evidence_kind": "search_excerpt", "trust": "untrusted_reference"})

    def _inspect(self, url: str, *, scope: str) -> None:
        url = public_source_url(url)
        if url not in {row["url"] for row in self._results()}:
            raise CodingResearchError("Source inspection must reference an observed search result")
        key = _hash({"scope": scope, "url": url})
        if key in self.state["inspections"]:
            return
        try:
            self._permission("browser_open")
            self._permission("browser_inspect")
        except PermissionError:
            self.state.setdefault("limitations", []).append("Source inspection unavailable or denied; search excerpts only")
            self._save("coding_research_inspection_denied")
            return
        self.state["inspections"][key] = {"url": url, "scope": scope, "status": "reserved"}
        self._save("coding_research_inspection_reserved", tool_call=True)
        self._event("inspect", "Inspecting a selected debugging source", operation_id=key, tool_name="browser_inspect")
        started = time.perf_counter()
        try:
            if self.inspector is not None:
                result = self.inspector(url)
            else:
                from mana_agent.connectors.browser.runtime_tools import build_browser_tools
                tools = {tool.name: tool for tool in build_browser_tools()}
                session_id = f"coding-research-{_hash(self.task_id)}"
                opened = json.loads(tools["browser_open"].call({"session_id": session_id, "url": url}))
                if not opened.get("ok"):
                    raise CodingResearchError("Source could not be opened")
                self._authority()
                self._save("coding_research_before_inspect", tool_call=True)
                result = json.loads(tools["browser_inspect"].call({"session_id": session_id, "tab_id": opened["tab_id"]}))
            self._authority()
            if not result.get("ok", True):
                raise CodingResearchError("Source could not be inspected")
            public_source_url(result.get("url", url))
            from mana_agent.chat.activity import safe_text
            self.state["inspections"][key] = {"url": url, "scope": scope, "status": "succeeded", "text": safe_text(result.get("text", ""), 12000, protected=self._protected()),
                                               "evidence_kind": "source_page", "trust": "untrusted_reference"}
        except Exception:
            self._authority()
            self.state["inspections"][key] = {"url": url, "scope": scope, "status": "failed", "limitation": "Source inspection failed; excerpts only"}
        self._save("coding_research_inspection_result")
        self._event("inspected", "Source inspection outcome recorded", operation_id=key,
                    status="success" if self.state["inspections"][key]["status"] == "succeeded" else "failed",
                    tool_name="browser_inspect", duration_ms=int((time.perf_counter() - started) * 1000))

    def prepare_delivery(self, decision: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Persist the exact continuation ticket before a backend may consume it."""
        self._authority()
        if self.state["delivery"].get("status") == "started":
            raise CodingResearchReconciliationRequired("Cannot replace an uncertain coding continuation")
        if decision.action not in {ContinuationAction.CONTINUE, ContinuationAction.REPLAN}:
            return {}
        context = {
            "instruction": decision.remaining_work,
            "recommendation": decision.research_recommendation.model_dump(mode="json") if decision.research_recommendation else None,
            "prior_result": {k: result.get(k) for k in ("changed_files", "commands_run", "thread_id", "turn_id", "workspace_path")},
            "continuation_context": result.get("continuation_context") or {},
            "untrusted_reference": True,
        }
        ticket = f"research-{decision.decision_id}"
        context = self._safe_context(context)
        self.state["delivery"] = {"id": ticket, "status": "pending", "context": context,
                                  "attempt_id": self.attempt_id, "attempt_generation": self._generation}
        self.state["continuation_context"] = context
        self._save("coding_research_before_delivery")
        self._event("resume", "Returning evidence to the active Coding Agent", operation_id=ticket, status="success")
        return {"continuation_id": ticket, "context": context}

    def acknowledge_delivery(self, ticket: str, result: dict[str, Any]) -> None:
        self._authority()
        delivery = self.state["delivery"]
        if delivery.get("id") != ticket:
            raise CodingResearchError("Coding continuation receipt does not match its delivery")
        delivery.update(status="suspended" if result.get("pending_operation") else "delivered", result={k: result.get(k) for k in (
            "status", "answer", "changed_files", "commands_run", "debugging_report", "continuation_context", "thread_id", "turn_id", "tests_run", "tests_passed", "backend"
        ) + ("pending_operation", "human_input", "error_code", "warnings", "workspace_path", "flow_id")})
        delivery["result"] = self._safe_context(delivery["result"])
        self.state["coding_result"] = delivery["result"]
        self._save("coding_research_delivered")

    def begin_delivery(self, ticket: str) -> dict[str, Any] | None:
        self._authority()
        delivery = self.state["delivery"]
        if delivery.get("id") != ticket:
            raise CodingResearchError("Unknown coding continuation ticket")
        if delivery["status"] == "delivered":
            return deepcopy(delivery["result"])
        if delivery["status"] == "started":
            recovered = self._recover_delivery_receipt()
            if recovered is not None:
                return recovered
            raise CodingResearchReconciliationRequired("Interrupted coding continuation requires side-effect reconciliation; it cannot be replayed")
        if delivery["status"] not in {"pending", "suspended"}:
            raise CodingResearchError("Invalid coding continuation delivery state")
        delivery["status"] = "started"
        self._save("coding_research_delivery_started")
        return None

    def _recover_delivery_receipt(self) -> dict[str, Any] | None:
        """Reuse only escrow explicitly correlated to this exact delivery/attempt."""
        self._authority()
        delivery = self.state["delivery"]
        for receipt in reversed(self.supervisor.store.results_for_task(self.task_id)):
            if receipt.task_id != self.task_id or receipt.attempt_id != delivery.get("attempt_id") or receipt.attempt_generation != delivery.get("attempt_generation"):
                continue
            if receipt.status.value == "rejected":
                continue
            payload = receipt.payload.get("result", receipt.payload)
            if not isinstance(payload, dict) or (payload.get("continuation_context") or {}).get("coding_continuation_id") != delivery["id"]:
                continue
            if payload.get("status") not in {"completed", "waiting", "failed", "cancelled"}:
                continue
            delivery["result_escrow_reference"] = receipt.result_id
            self.acknowledge_delivery(delivery["id"], payload)
            return deepcopy(self.state["delivery"]["result"])
        return None

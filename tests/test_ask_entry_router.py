from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from mana_agent.analysis.models import AskResponseWithTrace, SearchHit
from mana_agent.multi_agent.runtime.entry_router import EntryRouter, RouteDecision, RouteDecisionError
from mana_agent.services.ask_service import AskService
from mana_agent.vector_store.faiss_store import FaissStore


class _EmptyStore:
    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        return []


class _HitStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        return self._hits


class _FakeQnA:
    def __init__(self) -> None:
        self.last_context: str | None = None

    def run(self, question: str, context: str) -> str:
        self.last_context = context
        return "synthesized answer"

    def chat(self, question: str) -> str:
        self.last_context = question
        return "a is test"


def test_model_selected_conversation_route_answers_from_session_history(tmp_path: Path) -> None:
    qna = _FakeQnA()
    service = AskService(
        store=_EmptyStore(),
        qna_chain=qna,
        project_root=tmp_path,
        entry_router=_StaticRouter(
            RouteDecision(
                kind="conversation",
                confidence=0.98,
                reason="answer from active session history",
            )
        ),
    )
    transcript = (
        "Active conversation history (chronological):\n"
        "User: memory-test a=test\n\nCurrent user message:\nwhat is a?"
    )

    response = service.ask(index_dir=tmp_path, question=transcript, k=3)

    assert response.answer == "a is test"
    assert response.mode == "route-conversation"
    assert qna.last_context == transcript


class _StaticRouter:
    def __init__(self, *decisions: RouteDecision) -> None:
        self.decisions = list(decisions)
        self.router_model = "fake-router"
        self.calls: list[dict] = []

    def route(self, **kwargs):
        self.calls.append(kwargs)
        if not self.decisions:
            raise RouteDecisionError("Model decision failed: entry_route. No action executed. Reason: no decision")
        return self.decisions.pop(0)


class _FailingAskAgent:
    def run(self, **_kwargs):
        raise RuntimeError("agent boom")

    def run_multi(self, **_kwargs):
        raise RuntimeError("agent boom")


class _RecordingAskAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return AskResponseWithTrace(
            answer="gitops executed",
            sources=[],
            warnings=[],
            mode="agent",
            trace=[],
        )

    def run_multi(self, **kwargs):
        self.calls.append(kwargs)
        return AskResponseWithTrace(
            answer="gitops executed multi",
            sources=[],
            warnings=[],
            mode="agent",
            trace=[],
        )


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return AIMessage(content=self.content)


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def _command_project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                "",
                "[project.scripts]",
                'demo-cli = "demo.commands:app"',
            ]
        ),
        encoding="utf-8",
    )
    package = tmp_path / "demo"
    package.mkdir()
    (package / "commands.py").write_text(
        "\n".join(
            [
                "import typer",
                "",
                "app = typer.Typer()",
                "",
                "@app.command()",
                "def run():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_missing_faiss_index_does_not_automatically_project_search(tmp_path: Path) -> None:
    service = AskService(
        store=FaissStore(embeddings=object()),
        qna_chain=_FakeQnA(),
        project_root=tmp_path,
        entry_router=_StaticRouter(RouteDecision(kind="semantic_qa", confidence=0.9, reason="model selected index")),
    )

    response = service.ask(index_dir=tmp_path / ".mana" / "index", question="add", k=5)

    assert response.sources == []
    assert "Semantic index is unavailable" in response.answer
    assert response.route_trace["route_kind"] == "semantic_qa"
    assert "fallback" not in response.mode
    assert "fallback" not in response.answer.lower()


def test_missing_faiss_index_can_re_route_once_to_repo_search(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    router = _StaticRouter(
        RouteDecision(kind="semantic_qa", confidence=0.9, reason="model selected index"),
        RouteDecision(kind="repo_search", confidence=0.8, reason="index unavailable; inspect repo"),
    )
    qna = _FakeQnA()
    service = AskService(
        store=FaissStore(embeddings=object()),
        qna_chain=qna,
        project_root=project,
        entry_router=router,
    )

    response = service.ask(index_dir=project / ".mana" / "index", question="add", k=5)

    assert response.sources
    assert any("mod.py" in src.file_path for src in response.sources)
    assert qna.last_context is not None and "mod.py" in qna.last_context
    assert response.route_trace["route_kind"] == "repo_search"
    assert response.route_trace["validation"].startswith("re-routed")


def test_command_inventory_routes_through_tool_execution(tmp_path: Path) -> None:
    project = _command_project(tmp_path)
    service = AskService(
        store=_EmptyStore(),
        qna_chain=_FakeQnA(),
        project_root=project,
        entry_router=_StaticRouter(
            RouteDecision(
                kind="tool_execution",
                confidence=0.88,
                reason="model selected command inventory action",
                tool_plan=[{"tool": "command_inventory", "args": {}}],
            )
        ),
    )

    response = service.ask(index_dir=project / ".mana" / "index", question="what commands exist?", k=3)

    assert "`demo-cli` console script" in response.answer
    assert "`demo-cli run`" in response.answer
    assert response.route_trace["route_kind"] == "tool_execution"
    assert response.route_trace["executed_tools"] == ["command_inventory"]
    assert "fallback" not in response.mode


def test_gitops_route_executes_agent_tools_without_repo_search(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ask_agent = _RecordingAskAgent()
    qna = _FakeQnA()
    service = AskService(
        store=_EmptyStore(),
        qna_chain=qna,
        ask_agent=ask_agent,
        project_root=project,
        entry_router=_StaticRouter(RouteDecision(kind="gitops", confidence=0.93, reason="model selected GitOps")),
    )

    response = service.ask_with_tools(
        index_dir=project / ".mana" / "index",
        question="git add and git commit and push to feature/model-router-entry-no-fallback",
        k=3,
    )

    assert response.answer == "gitops executed"
    assert response.mode == "route-gitops"
    assert response.sources == []
    assert ask_agent.calls and ask_agent.calls[0]["question"].startswith("git add")
    assert qna.last_context is None
    assert response.route_trace["route_kind"] == "gitops"
    assert "fallback" not in response.mode


def test_router_can_select_gitops_for_commit_push_request() -> None:
    router = EntryRouter(
        llm=_FakeLLM(
            '{"kind":"gitops","confidence":0.94,"reason":"explicit git add commit push request",'
            '"requires_repo_context":true}'
        )
    )

    decision = router.route(
        question="git add and git commit and push to feature/model-router-entry-no-fallback",
        index_dir=None,
        project_root=Path.cwd(),
        available_commands=[],
        available_tools=["git_status", "git_add", "git_commit", "git_push", "repo_search"],
        runtime_state={"index_available": False},
    )

    assert decision.kind == "gitops"
    assert decision.requires_repo_context is True


def test_unknown_command_re_routes_to_clarification(tmp_path: Path) -> None:
    router = _StaticRouter(
        RouteDecision(kind="command", confidence=0.8, reason="model selected command", command_name="missing"),
        RouteDecision(
            kind="clarification",
            confidence=0.7,
            reason="command unavailable",
            user_visible_message="I could not find that command. Which Mana-Agent command should run?",
        ),
    )
    service = AskService(store=_EmptyStore(), qna_chain=_FakeQnA(), project_root=tmp_path, entry_router=router)

    response = service.ask(index_dir=tmp_path / ".mana" / "index", question="run missing", k=3)

    assert "could not find that command" in response.answer
    assert response.route_trace["route_kind"] == "clarification"
    assert len(router.calls) == 2


def test_agent_failure_does_not_run_classic_route(tmp_path: Path) -> None:
    service = AskService(
        store=_EmptyStore(),
        qna_chain=_FakeQnA(),
        ask_agent=_FailingAskAgent(),
        project_root=tmp_path,
        entry_router=_StaticRouter(RouteDecision(kind="tool_execution", confidence=0.8, reason="tool route")),
    )

    response = service.ask_with_tools(index_dir=tmp_path / ".mana" / "index", question="use tools", k=3)

    assert response.mode == "route-error"
    assert "Selected route failed: agent boom" in response.answer
    assert "fallback" not in response.mode
    assert "classic" not in response.mode


def test_dir_mode_failure_does_not_run_classic_dir_route(tmp_path: Path) -> None:
    service = AskService(
        store=_EmptyStore(),
        qna_chain=_FakeQnA(),
        ask_agent=_FailingAskAgent(),
        project_root=tmp_path,
        entry_router=_StaticRouter(RouteDecision(kind="tool_execution", confidence=0.8, reason="tool route")),
    )

    response = service.ask_with_tools_dir_mode(
        index_dirs=[tmp_path / ".mana" / "index"],
        question="use tools",
        k=3,
        root_dir=tmp_path,
    )

    assert response.mode == "route-error"
    assert "Selected route failed: agent boom" in response.answer
    assert "fallback" not in response.mode
    assert "classic" not in response.mode


def test_router_can_select_web_search_for_openclaw_request() -> None:
    router = EntryRouter(
        llm=_FakeLLM(
            '{"kind":"web_search","confidence":0.91,"reason":"needs current public web research",'
            '"requires_external_search":true}'
        )
    )

    decision = router.route(
        question="search internet and describe openclaw",
        index_dir=None,
        project_root=Path.cwd(),
        available_commands=[],
        available_tools=["web_search"],
        runtime_state={"index_available": False, "web_search_enabled": True},
    )

    assert decision.kind == "web_search"
    assert decision.requires_external_search is True


def test_invalid_router_output_stops_without_action(tmp_path: Path) -> None:
    service = AskService(
        store=_HitStore(
            [
                SearchHit(
                    score=0.9,
                    file_path="/tmp/proj/app.py",
                    start_line=1,
                    end_line=3,
                    symbol_name="add",
                    snippet="def add(a, b): return a + b",
                )
            ]
        ),
        qna_chain=_FakeQnA(),
        project_root=tmp_path,
        entry_router=EntryRouter(llm=_FakeLLM('{"kind":"missing","confidence":0.5,"reason":"bad"}')),
    )

    response = service.ask(index_dir=tmp_path, question="how does add work?", k=5)

    assert "Model decision failed: entry_route" in response.answer
    assert response.sources == []
    assert response.route_trace["validation"] == "router_error"

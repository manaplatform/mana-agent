from pathlib import Path

from mana_agent.analysis.models import AskResponseWithTrace, SearchHit
from mana_agent.multi_agent.runtime.entry_router import RouteDecision
from mana_agent.services.ask_service import AskService


class FakeStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def search(self, _index_dir: Path, query: str, k: int) -> list[SearchHit]:
        assert query
        assert k
        return self.hits


class FakeQnA:
    def run(self, question: str, context: str) -> str:
        assert question == "How does add work?"
        assert "source:" in context
        return "The add function sums two integers. /tmp/good.py:3-6"


class FakeSearchService:
    def search_multi(self, index_dirs: list[Path], query: str, k: int) -> tuple[list[SearchHit], list[str]]:
        assert query
        assert k
        assert index_dirs
        return (
            [
                SearchHit(
                    score=0.92,
                    file_path="/tmp/proj-a/app.py",
                    start_line=5,
                    end_line=9,
                    symbol_name="add",
                    snippet="def add(a,b): return a+b",
                ),
                SearchHit(
                    score=0.81,
                    file_path="/tmp/proj-b/lib.py",
                    start_line=2,
                    end_line=4,
                    symbol_name="sum_two",
                    snippet="def sum_two(a,b): return a+b",
                ),
            ],
            [],
        )


class FakeAskAgent:
    def run(
        self,
        question: str,
        index_dir: Path,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponseWithTrace:
        assert question
        assert index_dir
        assert k
        return AskResponseWithTrace(answer="Tool answer", sources=[], mode="agent-tools", trace=[], warnings=[])

    def run_multi(
        self,
        question: str,
        index_dirs: list[Path],
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponseWithTrace:
        assert question
        assert index_dirs
        assert k
        return AskResponseWithTrace(
            answer="Tool answer",
            sources=[
                SearchHit(
                    score=0.9,
                    file_path="/tmp/proj-a/app.py",
                    start_line=1,
                    end_line=3,
                    symbol_name="f",
                    snippet="def f(): pass",
                )
            ],
            mode="agent-tools",
            trace=[],
            warnings=[],
        )


class FakeRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.router_model = "fake-router"
        self.calls: list[dict] = []

    def route(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def test_ask_service_with_sources() -> None:
    sources = [
        SearchHit(
            score=0.9,
            file_path="/tmp/good.py",
            start_line=3,
            end_line=6,
            symbol_name="add",
            snippet="def add(a,b): return a+b",
        )
    ]
    router = FakeRouter(RouteDecision(kind="semantic_qa", confidence=0.9, reason="question needs indexed context"))
    service = AskService(store=FakeStore(sources), qna_chain=FakeQnA(), entry_router=router)
    response = service.ask(index_dir="/tmp/index", question="How does add work?", k=4)

    assert "/tmp/good.py:3-6" in response.answer
    assert response.sources
    assert response.to_dict()["route_trace"]["route_kind"] == "semantic_qa"


def test_ask_service_without_sources_stays_on_model_selected_route(tmp_path) -> None:
    router = FakeRouter(RouteDecision(kind="semantic_qa", confidence=0.9, reason="question needs indexed context"))
    service = AskService(store=FakeStore([]), qna_chain=FakeQnA(), project_root=tmp_path, entry_router=router)
    response = service.ask(index_dir="/tmp/index", question="How does add work?", k=4)
    assert "could not find relevant indexed code context" not in response.answer.lower()
    assert "selected semantic route found no indexed context" in response.answer.lower()
    assert "fallback" not in response.to_dict().get("mode", "")


def test_ask_service_dir_mode_groups_sources() -> None:
    service = AskService(
        store=FakeStore([]),
        qna_chain=FakeQnA(),
        search_service=FakeSearchService(),
    )
    response = service.ask_dir_mode(
        index_dirs=[Path("/tmp/proj-a/.mana/index"), Path("/tmp/proj-b/.mana/index")],
        question="How does add work?",
        k=4,
        root_dir="/tmp",
    )
    assert response.sources
    assert len(response.source_groups) == 2


def test_ask_service_tools_dir_mode_sets_grouped_sources() -> None:
    router = FakeRouter(RouteDecision(kind="tool_execution", confidence=0.8, reason="tools requested"))
    service = AskService(
        store=FakeStore([]),
        qna_chain=FakeQnA(),
        ask_agent=FakeAskAgent(),
        search_service=FakeSearchService(),
        entry_router=router,
    )
    response = service.ask_with_tools_dir_mode(
        index_dirs=[Path("/tmp/proj-a/.mana/index"), Path("/tmp/proj-b/.mana/index")],
        question="How does add work?",
        k=4,
    )
    assert response.mode == "route-tool_execution"
    assert response.source_groups
    assert response.route_trace["route_kind"] == "tool_execution"

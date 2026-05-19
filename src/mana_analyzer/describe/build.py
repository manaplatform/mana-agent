# mana_analyzer/describe/build.py

from mana_analyzer.config.settings import Settings
from mana_analyzer.dependencies import DependencyService
from mana_analyzer.llm.repo_chain import RepositoryMultiChain

from .describe_service import DescribeService
from .file_summary_executor import FileSummaryExecutor
from typing import Any

def build_describe_service(
    dependency_service: DependencyService | Settings | None = None,
    llm_chain: Any | None = None,
    include_tests: bool = False,
    model_override: str | None = None,
    use_llm: bool = True,
) -> DescribeService:
    if isinstance(dependency_service, Settings):
        settings = dependency_service
        dependency_service = DependencyService()
        if use_llm and llm_chain is None:
            llm_chain = RepositoryMultiChain(
                api_key=settings.openai_api_key,
                model=model_override or settings.openai_chat_model,
                base_url=settings.openai_base_url,
            )
    elif dependency_service is None:
        dependency_service = DependencyService()

    summary_executor = FileSummaryExecutor(
        file_agent=None,
        llm_chain=llm_chain,
    )

    return DescribeService(
        dependency_service=dependency_service,
        summary_executor=summary_executor,
        llm_chain=llm_chain,
        include_tests=include_tests,
    )

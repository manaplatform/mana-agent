"""
mana_agent.services.chat_service

This module provides `ChatService`, a thin wrapper around `AskService` that offers a
simple `ask()` API suitable for interactive CLI chat.

Design notes:
- `ChatService` does not construct `AskService`. The CLI is expected to build a fully
  configured `AskService` (e.g., via a `build_ask_service(...)` helper) and inject it.
- Two operating modes are supported:
  1) Single-index mode: use exactly one index directory.
  2) Dir-mode: operate over multiple index directories selected by the CLI.

The service optionally forwards `callbacks` to downstream methods when supported.
If a downstream method does not accept `callbacks`, the call is retried without them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from mana_agent.config.settings import Settings, default_index_dir
from mana_agent.services.ask_service import AskService

logger = logging.getLogger(__name__)


class ChatService:
    """
    ChatService wraps a fully-configured AskService and provides a simple ask() API
    for interactive CLI chat.

    IMPORTANT:
    - This class does NOT build AskService itself.
    - Pass in the AskService created by build_ask_service(...) from CLI.
    """

    def __init__(
        self,
        *,
        ask_service: AskService,
        settings: Settings,
        model_override: str | None = None,
        index_dir: str | Path | None = None,
        dir_mode: bool = False,
        root_dir: str | Path | None = None,
        k: int | None = None,
        agent_tools: bool = False,
        agent_max_steps: int = 6,
        agent_timeout_seconds: int = 30,
        # dir-mode options
        max_indexes: int = 0,
        auto_index_missing: bool = True,
    ) -> None:
        self._ask_service = ask_service
        self._settings = settings
        self._model_override = model_override

        self._k = int(k or settings.default_top_k)
        self._agent_tools = bool(agent_tools)
        self._agent_max_steps = int(agent_max_steps)
        self._agent_timeout_seconds = int(agent_timeout_seconds)

        self._dir_mode = bool(dir_mode)
        self._root_dir: Path = (
            Path(root_dir).expanduser().resolve()
            if root_dir is not None
            else Path.cwd().resolve()
        )

        # (question, answer) history for the current session
        self._history: list[tuple[str, str]] = []

        if self._dir_mode:
            # In dir-mode, the CLI computes/chooses index_dirs and supplies them via set_index_dirs.
            self._index_dirs: list[Path] = []
            self._max_indexes = int(max_indexes)
            self._auto_index_missing = bool(auto_index_missing)
        else:
            resolved = (
                Path(index_dir).expanduser().resolve()
                if index_dir is not None
                else default_index_dir(self._root_dir)
            )
            self._index_dirs = [resolved]
            self._max_indexes = 0
            self._auto_index_missing = False

    def set_index_dirs(self, index_dirs: list[Path]) -> None:
        """Used by the CLI to supply the computed dir-mode index list."""
        self._index_dirs = [Path(p).resolve() for p in index_dirs]

    @property
    def index_dirs(self) -> list[Path]:
        return list(self._index_dirs)

    def ask_conversation(self, question: str) -> Any:
        """Execute a model-selected conversational turn without a second router."""
        qna_chain = getattr(self._ask_service, "qna_chain", None)
        chat = getattr(qna_chain, "chat", None)
        if not callable(chat):
            raise RuntimeError(
                "Conversation route selected, but the conversational model is not configured."
            )
        return chat(str(question or "").strip())

    def ask(
        self,
        question: str,
        *,
        callbacks: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Ask a question using either:
          - dir-mode: multiple indexes via search_service / agent tools, or
          - single-index mode: one index via store / agent tools.

        `callbacks` is optional and is forwarded when supported by downstream services.
        If downstream does not support callbacks, we gracefully retry without them.
        """
        question = (question or "").strip()
        if not question:
            return None

        def _call_with_optional_callbacks(fn, /, **call_kwargs: Any) -> Any:
            """
            Call `fn` with `callbacks` if supported; otherwise retry without callbacks.
            """
            if callbacks is None:
                return fn(**call_kwargs)
            try:
                return fn(**call_kwargs, callbacks=callbacks)
            except TypeError:
                # Downstream method does not accept callbacks
                return fn(**call_kwargs)

        if self._dir_mode:
            if not self._index_dirs:
                raise RuntimeError(
                    "No indexes configured for dir-mode chat. "
                    "Compute selected indexes in CLI and call chat_service.set_index_dirs(...)."
                )

            call_k = kwargs.get("k", self._k) if kwargs else self._k
            extra = {kk: vv for kk, vv in (kwargs or {}).items() if kk != "k"}
            if self._agent_tools:
                # Tool/agent dir-mode path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_with_tools_dir_mode,
                    index_dirs=self._index_dirs,
                    question=question,
                    k=call_k,
                    max_steps=self._agent_max_steps,
                    timeout_seconds=self._agent_timeout_seconds,
                    root_dir=self._root_dir,
                    **extra,
                )
            else:
                # Classic dir-mode path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_dir_mode,
                    index_dirs=self._index_dirs,
                    question=question,
                    k=call_k,
                    root_dir=self._root_dir,
                    **extra,
                )
        else:
            if not self._index_dirs:
                raise RuntimeError(
                    "No index configured for chat. "
                    "Compute selected index in CLI and call chat_service.set_index_dirs(...)."
                )

            index_dir = self._index_dirs[0]
            call_k = kwargs.get("k", self._k) if kwargs else self._k
            extra = {kk: vv for kk, vv in (kwargs or {}).items() if kk != "k"}
            if self._agent_tools:
                # Tool/agent single-index path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_with_tools,
                    index_dir=index_dir,
                    question=question,
                    k=call_k,
                    max_steps=self._agent_max_steps,
                    timeout_seconds=self._agent_timeout_seconds,
                    **extra,
                )
            else:
                # Classic single-index path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask,
                    index_dir=index_dir,
                    question=question,
                    k=call_k,
                    **extra,
                )

        # Record history safely
        try:
            answer_text = getattr(response, "answer", None)
            if isinstance(answer_text, str) and answer_text:
                self._history.append((question, answer_text))
        except Exception:
            logger.debug("Failed to record chat history.", exc_info=True)

        return response

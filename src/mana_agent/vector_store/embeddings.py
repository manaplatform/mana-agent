"""Embedding-client construction.

Chat and embeddings share a single base URL in this project, so the embedding
client has to adapt to whichever provider that URL points at. OpenAI and NVIDIA
NIM endpoints are both OpenAI-compatible for *chat*, but their ``/embeddings``
endpoints differ in two ways that break the stock :class:`OpenAIEmbeddings`
client against NVIDIA:

* NVIDIA expects raw-string input, while ``OpenAIEmbeddings`` pre-tokenizes text
  into integer token-id arrays by default (``check_embedding_ctx_length=True``).
  Sending token arrays makes NVIDIA return ``500 Internal Server Error``.
* NVIDIA retrieval models (e.g. ``nv-embedqa-*``) require an ``input_type`` body
  field that is ``"query"`` for search queries and ``"passage"`` for documents.

:func:`build_embeddings` returns a plain ``OpenAIEmbeddings`` for OpenAI and an
NVIDIA-aware subclass for NVIDIA base URLs.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langchain_openai import OpenAIEmbeddings

from mana_agent.config.settings import resolve_embed_model


def is_nvidia_base_url(base_url: str | None) -> bool:
    return bool(base_url and "nvidia" in base_url.lower())


class NvidiaOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI-compatible embeddings tuned for NVIDIA NIM ``/embeddings``.

    Disables client-side tokenization (NVIDIA wants raw strings) and tags each
    request with the correct ``input_type`` so passages and queries are embedded
    in their respective spaces.
    """

    @contextmanager
    def _input_type(self, value: str) -> Iterator[None]:
        previous = self.model_kwargs.get("extra_body")
        extra_body = dict(previous or {})
        extra_body["input_type"] = value
        extra_body.setdefault("truncate", "END")
        self.model_kwargs["extra_body"] = extra_body
        try:
            yield
        finally:
            if previous is None:
                self.model_kwargs.pop("extra_body", None)
            else:
                self.model_kwargs["extra_body"] = previous

    def embed_documents(self, texts: list[str], *args: Any, **kwargs: Any) -> list[list[float]]:
        with self._input_type("passage"):
            return super().embed_documents(texts, *args, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        with self._input_type("query"):
            return super().embed_query(text)


def build_embeddings(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str | None = None,
    provider: str | None = None,
) -> OpenAIEmbeddings:
    """Build an embedding client appropriate for ``base_url`` / provider.

    The embedding model is auto-selected from the base URL or provider when
    ``model`` is not provided (see :func:`resolve_embed_model`).
    """
    resolved_model = resolve_embed_model(base_url, model, provider=provider)
    use_nvidia = str(provider or "").strip().lower() == "nvidia" or is_nvidia_base_url(base_url)
    if use_nvidia:
        return NvidiaOpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=resolved_model,
            check_embedding_ctx_length=False,
        )
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=resolved_model,
    )

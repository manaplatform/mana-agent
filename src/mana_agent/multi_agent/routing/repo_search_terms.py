"""Model-driven repository search terms.

Repo search must never use the full natural-language user message as the
literal search needle. A validated model decision selects compact terms; if
that decision is missing or invalid, no search runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.utils.project_search import (
    ProjectSearchMatch,
    ProjectSearchResult,
    project_search,
)


MAX_TERMS = 6
MAX_TERM_CHARS = 80
MIN_TERM_CHARS = 1

REPO_SEARCH_TERMS_PROMPT = """You are Mana-Agent's repository search term decision layer.
Convert the user request into compact repository text-search terms.

Rules:
- Return terms that can appear literally in source code, identifiers, paths, comments, or docs.
- Never return the full user sentence, question, or chat message as a term.
- Prefer symbols, APIs, filenames, library names, and domain keywords.
- Expand related spellings when useful (e.g. socket -> socket, websocket, WebSocket).
- Keep each term short (typically 1-3 tokens, never a full natural-language sentence).
- Do not invent long prose queries.
- If the request is already a compact code/search token, return it as the single term.
- If no useful repository search term can be chosen, return an empty terms list with a clear reason.

Return JSON only:
{
  "terms": ["compact", "needles"],
  "confidence": 0.0,
  "reason": "short reason",
  "fixed_strings": true
}
"""


class RepoSearchTermsDecisionError(RuntimeError):
    """Raised when repository search terms cannot be obtained from a model decision."""


@dataclass(slots=True)
class RepoSearchTermsDecision:
    terms: list[str]
    confidence: float
    reason: str
    fixed_strings: bool = True
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RepoSearchTermsDecisionEngine:
    llm: Any | None = None

    def decide(self, *, user_request: str) -> RepoSearchTermsDecision:
        request = str(user_request or "").strip()
        if not request:
            raise RepoSearchTermsDecisionError(
                "Model decision failed: repo_search_terms. No action executed. "
                "Reason: empty user request."
            )
        if self.llm is None or not hasattr(self.llm, "invoke"):
            raise RepoSearchTermsDecisionError(
                "Model decision failed: repo_search_terms. No action executed. "
                "Reason: term decision model is unavailable."
            )
        payload = {"user_request": request}
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=REPO_SEARCH_TERMS_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - converted to explicit decision failure
            raise RepoSearchTermsDecisionError(
                f"Model decision failed: repo_search_terms. No action executed. Reason: {exc}"
            ) from exc
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content
            )
        try:
            data = _extract_json(str(content))
            return validate_repo_search_terms_decision(data, user_request=request, source="model")
        except Exception as exc:  # noqa: BLE001 - validation failure blocks search
            raise RepoSearchTermsDecisionError(
                f"Model decision failed: repo_search_terms. No action executed. Reason: {exc}"
            ) from exc


def extract_terms_from_tool_inputs(
    tool_inputs: dict[str, Any] | None,
    *,
    tool_name: str = "repo_search",
) -> list[str]:
    """Return candidate terms from an agent decision tool_inputs payload."""

    if not isinstance(tool_inputs, dict):
        return []
    raw = tool_inputs.get(tool_name)
    if not isinstance(raw, dict):
        return []
    terms: list[str] = []
    raw_terms = raw.get("terms")
    if isinstance(raw_terms, list):
        terms.extend(str(item).strip() for item in raw_terms if str(item).strip())
    query = str(raw.get("query") or raw.get("q") or raw.get("pattern") or "").strip()
    if query:
        terms.append(query)
    # Preserve order while de-duplicating exact strings.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def decision_from_tool_inputs(
    tool_inputs: dict[str, Any] | None,
    *,
    user_request: str,
    tool_name: str = "repo_search",
) -> RepoSearchTermsDecision | None:
    """Build a validated term decision from tool_inputs, or None if absent."""

    candidates = extract_terms_from_tool_inputs(tool_inputs, tool_name=tool_name)
    if not candidates:
        return None
    return validate_repo_search_terms_decision(
        {
            "terms": candidates,
            "confidence": 0.8,
            "reason": f"Model-selected {tool_name} tool_inputs.",
            "fixed_strings": True,
        },
        user_request=user_request,
        source="tool_inputs",
    )


def resolve_repo_search_terms(
    *,
    user_request: str,
    llm: Any | None = None,
    tool_inputs: dict[str, Any] | None = None,
    tool_name: str = "repo_search",
    engine: RepoSearchTermsDecisionEngine | None = None,
) -> RepoSearchTermsDecision:
    """Resolve validated search terms from tool_inputs or a model decision.

    Never falls back to the full user message as a search term.
    """

    try:
        from_inputs = decision_from_tool_inputs(
            tool_inputs,
            user_request=user_request,
            tool_name=tool_name,
        )
    except Exception as exc:  # noqa: BLE001
        raise RepoSearchTermsDecisionError(
            f"Model decision failed: repo_search_terms. No action executed. Reason: {exc}"
        ) from exc
    if from_inputs is not None:
        return from_inputs
    decision_engine = engine or RepoSearchTermsDecisionEngine(llm=llm)
    return decision_engine.decide(user_request=user_request)


def validate_repo_search_terms_decision(
    data: dict[str, Any],
    *,
    user_request: str,
    source: str = "model",
) -> RepoSearchTermsDecision:
    if not isinstance(data, dict):
        raise ValueError("repo_search_terms output must be a JSON object")
    raw_terms = data.get("terms")
    if raw_terms is None and data.get("query") is not None:
        raw_terms = [data.get("query")]
    if not isinstance(raw_terms, list):
        raise ValueError("terms must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_terms:
        term = str(item or "").strip()
        if not term:
            continue
        if len(term) < MIN_TERM_CHARS:
            continue
        if len(term) > MAX_TERM_CHARS:
            raise ValueError(f"term exceeds {MAX_TERM_CHARS} characters: {term[:40]}...")
        # Preserve case variants (e.g. websocket vs WebSocket) for case-sensitive
        # fixed-string search; only drop exact duplicates.
        if term in seen:
            continue
        seen.add(term)
        cleaned.append(term)
        if len(cleaned) >= MAX_TERMS:
            break
    if not cleaned:
        raise ValueError("at least one non-empty search term is required")
    for term in cleaned:
        if _term_is_whole_message(term, user_request):
            raise ValueError(
                "search term must not equal the full user message; choose compact code/search needles"
            )
    try:
        confidence = float(data.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    reason = str(data.get("reason") or "").strip() or "Model-selected repository search terms."
    fixed_strings = bool(data.get("fixed_strings", True))
    return RepoSearchTermsDecision(
        terms=cleaned,
        confidence=confidence,
        reason=reason[:600],
        fixed_strings=fixed_strings,
        source=source,
    )


def project_search_with_terms(
    terms: Sequence[str],
    root: str | Path,
    *,
    max_results: int = 50,
    max_output_chars: int = 12_000,
    fixed_strings: bool = True,
) -> ProjectSearchResult:
    """Search each model-selected term and merge unique matches."""

    cleaned = [str(term).strip() for term in terms if str(term).strip()]
    if not cleaned:
        return ProjectSearchResult(matches=[], backend="none", truncated=False)

    per_term = max(1, max_results // max(1, len(cleaned)))
    merged: list[ProjectSearchMatch] = []
    seen: set[tuple[str, int, str]] = set()
    backend = "none"
    truncated = False
    for term in cleaned:
        result = project_search(
            term,
            root,
            max_results=per_term,
            max_output_chars=max_output_chars,
            fixed_strings=fixed_strings,
        )
        if result.backend != "none":
            backend = result.backend
        truncated = truncated or result.truncated
        for match in result.matches:
            key = (match.file_path, match.line_number, match.line_text)
            if key in seen:
                continue
            seen.add(key)
            merged.append(match)
            if len(merged) >= max_results:
                truncated = True
                return ProjectSearchResult(matches=merged, backend=backend, truncated=truncated)
    return ProjectSearchResult(matches=merged, backend=backend, truncated=truncated)


def _term_is_whole_message(term: str, user_request: str) -> bool:
    normalized_term = " ".join(str(term or "").split()).casefold()
    normalized_request = " ".join(str(user_request or "").split()).casefold()
    if not normalized_term or not normalized_request:
        return False
    if normalized_term != normalized_request:
        return False
    tokens = normalized_request.split()
    # Compact one- or two-token requests may legitimately equal the term
    # (e.g. "socket", "ToolManager", "search router"). Full questions may not.
    if len(tokens) >= 3:
        return True
    if normalized_request.endswith("?"):
        return True
    if any(ch in normalized_request for ch in ".!"):
        return True
    return False


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("repo_search_terms output must decode to an object")
    return data

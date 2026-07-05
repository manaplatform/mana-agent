from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from mana_agent import __version__ as MANA_AGENT_VERSION
from mana_agent.agent.flow import build_agent_flow
from mana_agent.agent.task_context import render_task_context
from mana_agent.multi_agent.runtime.prompts import (
    CODING_AGENT_LANGUAGE_TOOLING_PROMPT,
    CODING_AGENT_RECOGNITION_PROMPT,
    CODING_FLOW_MEMORY_PROMPT,
    FULL_AUTO_EXECUTION_PROMPT,
)
from mana_agent.prompting.layers import (
    EphemeralPromptContext,
    PROMPT_TEMPLATE_VERSION,
    PromptLayer,
    StablePromptState,
    compose_layers,
)
from mana_agent.prompting.memory_snapshot import render_memory_snapshot
from mana_agent.prompting.mode_rules import render_mode_rules
from mana_agent.prompting.output_contract import render_output_contract
from mana_agent.prompting.repo_rules import render_repo_rules
from mana_agent.prompting.skills_index import render_matched_skill_context, render_stable_skills_index


logger = logging.getLogger(__name__)

STABLE_PROMPT_BUDGET_CHARS = 24_000
EPHEMERAL_PROMPT_BUDGET_CHARS = 12_000
MAX_EPHEMERAL_ITEMS = 12


def _join_sections(*sections: str | None) -> str:
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _compact_text(text: str, *, max_chars: int) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, (len(str(text or "")) + 3) // 4)


def _normalize_tools(tools: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(sorted(str(tool).strip() for tool in tools or () if str(tool).strip()))


def _profile_text(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "default"
    stable_items = {
        key: str(value)
        for key, value in sorted(config.items())
        if key
        in {
            "model",
            "provider",
            "base_url",
            "full_auto_mode",
            "planner_model",
            "allowed_prefixes",
        }
    }
    return repr(stable_items)


def _behavior_rules(*, full_auto_mode: bool = False) -> str:
    sections = [
        "Agent Behavior Rules\n"
        "- Keep continuity through compact summaries and durable memory, not by appending old turns to the system prompt.\n"
        "- Inspect before editing, use repository-scoped tools, and preserve unrelated user changes.\n"
        "- Prefer batch tools: repo_batch_read for multiple files, repo_batch_search for multiple queries, run_script_once for grouped commands, and apply_patch_batch for related multi-file patches.\n"
        "- Keep per-turn task details, retrieved files, tool outputs, and temporary plans out of the stable prompt.\n"
        "- Summarize oversized context before continuing.",
        CODING_AGENT_RECOGNITION_PROMPT,
    ]
    if full_auto_mode:
        sections.append(FULL_AUTO_EXECUTION_PROMPT)
    return _join_sections(*sections)


def _verification_rules() -> str:
    return (
        "Safety and Verification Rules\n"
        "- Run the most relevant checks after code changes when available.\n"
        "- Report verification commands and outcomes honestly.\n"
        "- Do not claim edits, fixes, or verification without tool or command evidence.\n"
        "- Do not log full prompts, private file contents, API keys, or secret-like values by default."
    )


def _stable_layers(state: StablePromptState) -> list[PromptLayer]:
    return [
        PromptLayer("core_identity", state.identity),
        PromptLayer("tool_rules", state.tool_rules),
        PromptLayer("agent_behavior_rules", state.behavior_rules),
        PromptLayer("skills_index", state.skill_index),
        PromptLayer("repo_rules", state.repo_rules),
        PromptLayer("verification_rules", state.verification_rules),
    ]


class PromptCache:
    """Session-local stable prompt cache keyed only by stable inputs."""

    def __init__(self) -> None:
        self._state: StablePromptState | None = None
        self.last_hit: bool = False
        self.last_invalidation_reason: str = "empty"

    def get_or_build(
        self,
        repo: str | Path | None,
        config: Mapping[str, Any] | None,
        skills: str | None,
        tools: Sequence[str] | None,
        *,
        identity: str,
        tool_rules: str,
        behavior_rules: str,
        repo_rules: str,
        verification_rules: str,
    ) -> StablePromptState:
        _ = repo
        normalized_tools = _normalize_tools(tools)
        skill_index = skills or "Compact Skills Index\n- none available"
        key_parts = {
            "mana_agent_version": MANA_AGENT_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "enabled_tools": normalized_tools,
            "skill_index_hash": _hash_text(skill_index),
            "repository_rules_hash": _hash_text(repo_rules),
            "identity_rules_hash": _hash_text(_join_sections(identity, tool_rules, behavior_rules, verification_rules)),
            "model_provider_profile": _hash_text(_profile_text(config)),
        }
        cache_key = _hash_text(repr(sorted(key_parts.items())))
        if self._state is not None and self._state.cache_key == cache_key:
            self.last_hit = True
            self.last_invalidation_reason = "cache_key_unchanged"
            self._log("hit", self._state)
            return self._state

        reason = self._reason_for_miss(cache_key, key_parts)
        state = StablePromptState(
            identity=identity,
            tool_rules=tool_rules,
            behavior_rules=behavior_rules,
            skill_index=skill_index,
            repo_rules=repo_rules,
            verification_rules=verification_rules,
            cache_key=cache_key,
            version=PROMPT_TEMPLATE_VERSION,
        )
        prompt = compose_stable_prompt(state)
        if len(prompt) > STABLE_PROMPT_BUDGET_CHARS:
            state = StablePromptState(
                identity=_compact_text(identity, max_chars=6000),
                tool_rules=_compact_text(tool_rules, max_chars=5000),
                behavior_rules=_compact_text(behavior_rules, max_chars=5000),
                skill_index=_compact_text(skill_index, max_chars=4000),
                repo_rules=_compact_text(repo_rules, max_chars=2500),
                verification_rules=verification_rules,
                cache_key=cache_key,
                version=PROMPT_TEMPLATE_VERSION,
            )
        self._state = state
        self.last_hit = False
        self.last_invalidation_reason = reason
        self._log("miss", state)
        return state

    def _reason_for_miss(self, cache_key: str, key_parts: Mapping[str, Any]) -> str:
        if self._state is None:
            return "initial_build"
        previous = self._state.cache_key
        if previous != cache_key:
            logger.debug("stable prompt invalidation candidates=%s", sorted(key_parts.keys()))
            return "stable_inputs_changed"
        return "cache_key_unchanged"

    def _log(self, status: str, state: StablePromptState) -> None:
        stable_prompt = compose_stable_prompt(state)
        logger.debug(
            "stable prompt cache %s key=%s invalidation_reason=%s stable_tokens=%s",
            status,
            state.cache_key[:16],
            self.last_invalidation_reason,
            _estimate_tokens(stable_prompt),
        )


prompt_cache = PromptCache()


def compose_stable_prompt(stable_prompt: StablePromptState) -> str:
    return compose_layers(_stable_layers(stable_prompt))


def build_ephemeral_context(
    task: str,
    *,
    mode: str = "answer_only",
    retrieved_files: Sequence[str] | None = None,
    tool_results: Sequence[str] | None = None,
    recent_summary: str | None = None,
    temporary_constraints: Sequence[str] | None = None,
) -> EphemeralPromptContext:
    return EphemeralPromptContext(
        current_task=_compact_text(task, max_chars=3000),
        mode=str(mode or "answer_only"),
        retrieved_files=[_compact_text(item, max_chars=900) for item in list(retrieved_files or ())[:MAX_EPHEMERAL_ITEMS]],
        tool_results=[_compact_text(item, max_chars=900) for item in list(tool_results or ())[:MAX_EPHEMERAL_ITEMS]],
        recent_summary=_compact_text(recent_summary, max_chars=2000) if recent_summary else None,
        temporary_constraints=[
            _compact_text(item, max_chars=500)
            for item in list(temporary_constraints or ())[:MAX_EPHEMERAL_ITEMS]
            if str(item).strip()
        ],
    )


def render_ephemeral_context(context: EphemeralPromptContext) -> str:
    lines = [
        "Ephemeral Prompt Context",
        "- This message is per-call only. Do not cache it as stable prompt state.",
        f"- current_task: {context.current_task}",
        f"- current_mode: {context.mode}",
    ]
    if context.recent_summary:
        lines.extend(["- recent_local_reasoning_summary:", context.recent_summary])
    if context.temporary_constraints:
        lines.append("- temporary_constraints:")
        lines.extend(f"  - {item}" for item in context.temporary_constraints)
    if context.retrieved_files:
        lines.append("- retrieved_files_relevant_snippets:")
        lines.extend(f"  - {item}" for item in context.retrieved_files)
    if context.tool_results:
        lines.append("- summarized_tool_results:")
        lines.extend(f"  - {item}" for item in context.tool_results)
    rendered = "\n".join(lines)
    return _compact_text(rendered, max_chars=EPHEMERAL_PROMPT_BUDGET_CHARS)


def compose(
    stable_prompt: StablePromptState,
    ephemeral_prompt: EphemeralPromptContext,
    user_message: str,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": compose_stable_prompt(stable_prompt)},
        {"role": "developer", "content": render_ephemeral_context(ephemeral_prompt)},
        {"role": "user", "content": str(user_message or "")},
    ]
    messages.extend({"role": "tool", "content": item} for item in ephemeral_prompt.tool_results if str(item).strip())
    return messages


def get_or_build_stable_prompt(
    *,
    base_prompt: str,
    repo_root: str | Path | None = None,
    full_auto_mode: bool = False,
    enabled_tools: Sequence[str] | None = None,
    model_profile: Mapping[str, Any] | None = None,
    cache: PromptCache | None = None,
) -> StablePromptState:
    repo_rules = render_repo_rules(repo_root=repo_root)
    skills = render_stable_skills_index(repo_root=repo_root)
    active_cache = cache or prompt_cache
    return active_cache.get_or_build(
        repo_root,
        {
            **dict(model_profile or {}),
            "full_auto_mode": str(bool(full_auto_mode)),
        },
        skills,
        enabled_tools,
        identity=base_prompt,
        tool_rules=CODING_AGENT_LANGUAGE_TOOLING_PROMPT,
        behavior_rules=_behavior_rules(full_auto_mode=full_auto_mode),
        repo_rules=repo_rules,
        verification_rules=_verification_rules(),
    )


def build_coding_system_prompt(
    *,
    base_prompt: str,
    request: str,
    repo_root: str | Path | None = None,
    flow_context: str | None = None,
    full_auto_mode: bool = False,
    include_edit_rules: bool = False,
    explicit_mode: str | None = None,
    prompt_cache: PromptCache | None = None,
    enabled_tools: Sequence[str] | None = None,
    model_profile: Mapping[str, Any] | None = None,
) -> str:
    flow = build_agent_flow(
        request,
        repo_root=repo_root,
        explicit_mode=explicit_mode,
        flow_context=flow_context,
    )
    stable = get_or_build_stable_prompt(
        base_prompt=base_prompt,
        repo_root=repo_root,
        full_auto_mode=full_auto_mode,
        enabled_tools=enabled_tools,
        model_profile=model_profile,
        cache=prompt_cache,
    )
    temporary_constraints = [
        render_mode_rules(flow.context.mode),
        render_output_contract(flow.context.mode),
    ]
    if include_edit_rules:
        temporary_constraints.append("Current task has edit intent; inspect, patch, and verify.")
    if flow_context:
        temporary_constraints.append(_join_sections(CODING_FLOW_MEMORY_PROMPT, f"Active Flow Context\n{flow_context.strip()}"))
    ephemeral = build_ephemeral_context(
        render_task_context(flow.context),
        mode=flow.context.mode,
        retrieved_files=[
            render_matched_skill_context(request, repo_root=repo_root),
            render_memory_snapshot(repo_root=repo_root),
        ],
        recent_summary=flow_context,
        temporary_constraints=temporary_constraints,
    )
    stable_text = compose_stable_prompt(stable)
    ephemeral_text = render_ephemeral_context(ephemeral)
    logger.debug(
        "prompt assembly stable_key=%s stable_tokens=%s ephemeral_tokens=%s total_tokens=%s",
        stable.cache_key[:16],
        _estimate_tokens(stable_text),
        _estimate_tokens(ephemeral_text),
        _estimate_tokens(stable_text) + _estimate_tokens(ephemeral_text),
    )
    return _join_sections(stable_text, ephemeral_text)

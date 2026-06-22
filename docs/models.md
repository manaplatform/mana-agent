# Models

This project uses LLM *model names* (for chat) and embedding *model names* for vector search.

> Note: the actual model/provider used at runtime can be overridden via environment variables (see `src/mana_analyzer/config/settings.py`).

## Chat models (LLM)

### Default chat model
- **`gpt-4.1-mini`** (default) — configured as `Settings.openai_chat_model`.
  - Defined in: `src/mana_analyzer/config/settings.py:36-40`

### Optional overrides
The code supports per-purpose chat model overrides via settings/env:
- **`OPENAI_TOOL_WORKER_MODEL`** → `Settings.openai_tool_worker_model`.
  - Defined in: `src/mana_analyzer/config/settings.py:36-40`
- **`OPENAI_CODING_PLANNER_MODEL`** → `Settings.openai_coding_planner_model`.
  - Defined in: `src/mana_analyzer/config/settings.py:36-40`
- **`OPENAI_CHAT_MODEL`** → `Settings.openai_chat_model`.
  - Defined in: `src/mana_analyzer/config/settings.py:36-40`

### Planner model selection (coding agent)
The coding agent planner model is selected in `_setup_planner()` using, in priority order:
- `OPENAI_CODING_PLANNER_MODEL` / `CODING_AGENT_PLANNER_MODEL` env var
- `self.planner_model`
- `self.ask_agent.model`
- `OPENAI_CHAT_MODEL`
- fallback: **`gpt-4.1-mini`**

Source: `src/mana_analyzer/llm/coding_agent.py:432-463`.

## Tool worker model
A separate model can be configured for the tool-worker subprocess via `OPENAI_TOOL_WORKER_MODEL`.
- Setting: `src/mana_analyzer/config/settings.py:36-45`

## Embedding models

The base URL (`OPENAI_BASE_URL`) determines which embedding model to use by default.

### OpenAI embedding default
- **`text-embedding-3-small`**
  - Defined in: `src/mana_analyzer/config/settings.py:13-18`

### NVIDIA embedding default
- **`nvidia/nv-embedqa-e5-v5`**
  - Defined in: `src/mana_analyzer/config/settings.py:13-18`

### Resolution logic
`resolve_embed_model(base_url, explicit_model)`:
- If `OPENAI_EMBED_MODEL` is explicitly set, it always wins.
- Otherwise, if the `base_url` contains `"nvidia"` → use `nvidia/nv-embedqa-e5-v5`.
- Otherwise → use `text-embedding-3-small`.

Source: `src/mana_analyzer/config/settings.py:20-35`.

### Settings
- **`OPENAI_EMBED_MODEL`** → `Settings.openai_embed_model` (optional; auto-selected when unset)
  - Defined in: `src/mana_analyzer/config/settings.py:41-45`

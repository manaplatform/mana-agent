# Entry routing and chat sessions

Every gateway turn begins with one structured model decision before a conversational model, coding agent, search service, or connector can run. The entry decision selects one registered route: `conversation`, `coding`, `gmail`, `calendar`, `search`, `repository`, `automation`, or `unsupported`.

The route registry is runtime data, not a keyword table. Each route publishes its description, tools, configuration state, authorization state, and any setup action. The model chooses the route from the request, prior route, conversation context, and this live registry snapshot. Invalid model output stops with an entry-decision error; it does not execute a default route or generate an integration refusal.

Gmail readiness is checked from enabled email-account metadata, granted `email.read` permission, and the referenced keyring credential. A configured Gmail request runs through an email-only AskAgent policy. Missing configuration or credentials returns the registry's actionable setup/reconnect error. Provider authorization errors from Gmail retain their provider code, HTTP status, and `reconnect_required` detail.

## Session lifecycle

A frontend opens exactly one workspace session for a chat. All turns, route decisions, connector calls, model calls, coding work, memory, and persisted messages reuse its `session_id`; its `conversation_id` and each `turn_id` are passed through the gateway result and connector execution context.

Session records use `active`, `closed`, or `abandoned` lifecycle states (`archived` remains readable for older records), with opening and closing timestamps. CLI exit, TUI unmount/quit, dashboard shutdown, and `/new` use the same idempotent gateway close operation. `/new` closes the current session and opens a new one. Closing never deletes `messages.jsonl`, so historical conversations remain inspectable.

On a newly opened chat, Mana-Agent creates a new session rather than reopening a closed chat. Active sessions owned by a process that no longer exists are finalized as `abandoned`; opening a new chat also abandons any previous active chat for the same repository before creating the new identity.

Frontends should construct or reuse `AgentChatGateway`, call `create_session()` once when the chat opens, use `process_turn()` for every message, and call `close_session()` on every shutdown path. They must not instantiate a gateway or workspace session per message.

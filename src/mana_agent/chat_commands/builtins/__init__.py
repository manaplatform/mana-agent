from __future__ import annotations

import json
from typing import Any

from mana_agent.chat_commands.models import CommandContext, CommandDefinition, CommandResult


def _sessions(context: CommandContext, args: list[str]) -> CommandResult:
    if context.sessions is None:
        raise RuntimeError("Session service is unavailable. No fallback action was executed.")
    action = args[0].lower() if args else "list"
    if action == "list":
        rows = context.sessions.list(workspace_id=context.workspace_id or None, current_id=context.session_id)
        data = [row.model_dump(mode="json") for row in rows]
        lines = [f"{'*' if row.current else ' '} {row.short_id}  {row.title}  {row.status}  {row.message_count} messages" for row in rows]
        events = [{"type": "session.picker", "sessions": data}] if context.frontend in {"tui", "dashboard"} else []
        return CommandResult(status="success", message="\n".join(lines) or "No sessions found.", data={"sessions": data}, events=events)
    if action == "current":
        row = context.sessions.workspaces.store.get_session(context.session_id)
        summary = context.sessions.summary(row, current_id=context.session_id)
        return CommandResult(status="success", message=json.dumps(summary.model_dump(mode="json"), indent=2), data={"session": summary.model_dump(mode="json")})
    if action == "show":
        sid = args[1] if len(args) > 1 else context.session_id
        row = context.sessions.workspaces.store.get_session(sid)
        summary = context.sessions.summary(row, current_id=context.session_id)
        messages = [item.to_dict() for item in context.sessions.history.list(sid, limit=5000)]
        return CommandResult(status="success", message=json.dumps(summary.model_dump(mode="json"), indent=2), data={"session": summary.model_dump(mode="json"), "messages": messages})
    if action == "switch" and len(args) == 2:
        messages = context.gateway.switch_session(args[1], frontend=context.frontend) if context.gateway else context.sessions.bind(args[1], frontend=context.frontend, workspace_id=context.workspace_id or None).messages
        return CommandResult(status="success", message=f"Switched to session {args[1]}.", data={"session_id": args[1], "messages": messages}, events=[{"type": "timeline.replace", "messages": messages}])
    if action == "rename" and len(args) >= 3:
        row = context.sessions.rename(args[1], " ".join(args[2:]))
        return CommandResult(status="success", message=f"Renamed session {row.session_id} to {row.title}.", data={"session": row.model_dump(mode="json")})
    if action == "delete" and len(args) == 2:
        deleted_id = args[1]
        if context.gateway is not None and hasattr(context.gateway, "delete_session"):
            context.gateway.delete_session(deleted_id)
        else:
            context.sessions.delete(deleted_id, gateway=context.gateway)
        events = [{"type": "session.deleted", "session_id": deleted_id}]
        data: dict[str, Any] = {}
        if deleted_id == context.session_id:
            if context.gateway is None or not hasattr(context.gateway, "create_new_session"):
                raise RuntimeError("The active session was deleted but no gateway can bind its replacement.")
            replacement = context.gateway.create_new_session(frontend=context.frontend)
            data["session_id"] = replacement
            events.extend([
                {"type": "timeline.replace", "messages": []},
                {"type": "session.activated", "session_id": replacement},
            ])
        return CommandResult(status="success", message=f"Deleted session {deleted_id}.", data=data, events=events)
    raise ValueError("Usage: /sessions [list|current|show [id]|switch <id>|rename <id> <title>|delete <id>]")


def _new(context: CommandContext, _args: list[str]) -> CommandResult:
    if not context.gateway or not context.session_id:
        raise RuntimeError("An active gateway session is required. No fallback action was executed.")
    sid = context.gateway.start_new_conversation(context.session_id, frontend=context.frontend)
    return CommandResult(status="success", message="", data={"session_id": sid}, events=[{"type": "timeline.replace", "messages": []}, {"type": "session.activated", "session_id": sid}])


def _processes(context: CommandContext, args: list[str]) -> CommandResult:
    if context.processes is None:
        raise RuntimeError("Background-process service is unavailable. No fallback action was executed.")
    action = args[0].lower() if args else "list"
    if action == "list":
        rows = [row.model_dump(mode="json") for row in context.processes.list()]
        return CommandResult(status="success", message=json.dumps(rows, indent=2), data={"processes": rows})
    if action == "cleanup":
        return CommandResult(status="success", message=f"Cleaned {context.processes.cleanup()} stale process records.")
    if len(args) < 2:
        raise ValueError("Usage: /processes show|logs|stop|restart <id> | cleanup")
    pid = args[1]
    if action == "show":
        row = context.processes.inspect(pid)
        return CommandResult(status="success", message=json.dumps(row.model_dump(mode="json"), indent=2), data={"process": row.model_dump(mode="json")})
    if action == "logs":
        value = context.processes.logs(pid)
        return CommandResult(status="success", message=value, data={"process_id": pid, "logs": value})
    if action == "stop":
        row = context.processes.stop(pid)
    elif action == "restart":
        row = context.processes.restart(pid)
    else:
        raise ValueError("Usage: /processes [list|show <id>|logs <id>|stop <id>|restart <id>|cleanup]")
    return CommandResult(status="success", message=f"Process {row.process_id}: {row.state}.", data={"process": row.model_dump(mode="json")})


def _connect(context: CommandContext, args: list[str]) -> CommandResult:
    if context.connectors is None:
        raise RuntimeError("Connector service is unavailable. No fallback action was executed.")
    return context.connectors.command(context, args)


def _gateway(name: str):
    def handler(context: CommandContext, args: list[str]) -> CommandResult:
        if context.gateway is None:
            raise RuntimeError("Gateway is unavailable. No fallback action was executed.")
        value = context.gateway.handle_control_command("/" + " ".join([name, *args]), session_id=context.session_id)
        if value is None:
            raise RuntimeError(f"/{name} has no registered gateway adapter.")
        return CommandResult(status="success", message=value)
    return handler


def _status(context: CommandContext, _args: list[str]) -> CommandResult:
    if context.gateway is None:
        raise RuntimeError("Gateway is unavailable. No fallback action was executed.")
    return CommandResult(status="success", message=f"Mana-Agent session status: {context.gateway.status(context.session_id)}.")


def _cancel(context: CommandContext, _args: list[str]) -> CommandResult:
    if context.gateway is None:
        raise RuntimeError("Gateway is unavailable. No fallback action was executed.")
    cancelled = context.gateway.cancel(context.session_id)
    return CommandResult(status="success", message="Cancellation requested." if cancelled else "No cancellable turn is active.")


def _identity(context: CommandContext, _args: list[str]) -> CommandResult:
    if context.frontend != "telegram":
        return CommandResult(status="error", message="/id is intentionally available only in Telegram.")
    values = context.frontend_data
    return CommandResult(status="success", message=f"User ID: {values.get('user_id')}\nChat ID: {values.get('chat_id')}\nTopic ID: {values.get('topic_id', 'none')}.")


def _agent_command(name: str):
    def handler(context: CommandContext, args: list[str]) -> CommandResult:
        if context.gateway is None or not hasattr(context.gateway, "process_turn"):
            raise RuntimeError(f"/{name} requires the shared agent gateway. No fallback action was executed.")
        request = f"Run the registered {name} workflow"
        if args:
            request += ": " + " ".join(args)
        result = context.gateway.process_turn(context.session_id, request)
        if getattr(result, "error", None) and not getattr(result, "answer", None):
            raise RuntimeError(str(result.error))
        return CommandResult(status="success", message=str(getattr(result, "answer", result) or ""))
    return handler


def definitions() -> list[CommandDefinition]:
    rows = [
        CommandDefinition(canonical_name="new", description="Permanently replace the current chat.", required_capability="sessions", handler=_new),
        CommandDefinition(canonical_name="sessions", aliases=("session",), description="List and manage canonical chats.", argument_schema="[list|current|show|switch|rename|delete]", required_capability="sessions", confirmation_actions=frozenset({"delete"}), handler=_sessions),
        CommandDefinition(canonical_name="processes", description="Inspect persistent operating-system services.", argument_schema="[list|show|logs|stop|restart|cleanup]", required_capability="processes", confirmation_actions=frozenset({"stop"}), handler=_processes),
        CommandDefinition(canonical_name="connect", description="Configure or manage a connector.", argument_schema="[list|telegram ...]", required_capability="connectors", accepts_secrets=True, handler=_connect),
        CommandDefinition(canonical_name="disconnect", description="Remove connector configuration.", argument_schema="<name>", required_capability="connectors", confirmation_required=True, handler=lambda context, args: _connect(context, ["disconnect", *args])),
        CommandDefinition(canonical_name="status", description="Show the active chat status.", required_capability="gateway", handler=_status),
        CommandDefinition(canonical_name="cancel", description="Cancel the active agent turn.", required_capability="gateway", handler=_cancel),
        CommandDefinition(canonical_name="id", description="Show connector identity details.", required_capability="gateway", frontends=frozenset({"telegram"}), handler=_identity),
    ]
    for name, description in {
        "route": "Inspect the model routing decision.", "tasks": "List active agent tasks.",
        "task": "Inspect or control an agent task.", "budget": "Show task budget usage.",
        "candidates": "Show candidate task executions.", "models": "Inspect model health and selection.",
    }.items():
        rows.append(CommandDefinition(canonical_name=name, description=description, required_capability="gateway", handler=_gateway(name)))
    rows.extend([
        CommandDefinition(canonical_name="plan", description="Create a repository-aware implementation plan.", argument_schema="[request]", required_capability="gateway", execution_mode="task", handler=_agent_command("plan")),
        CommandDefinition(canonical_name="analyze", description="Analyze the selected repository context.", argument_schema="[request]", required_capability="gateway", execution_mode="task", handler=_agent_command("analyze")),
        CommandDefinition(canonical_name="doctor", description="Diagnose Mana-Agent and optionally apply safe repairs.", argument_schema="[--safe-fixes]", required_capability="gateway", confirmation_actions=frozenset({"--safe-fixes", "safe-fixes"}), execution_mode="task", handler=_agent_command("doctor")),
    ])
    rows.append(CommandDefinition(canonical_name="help", description="List commands available in this frontend.", handler=lambda _context, _args: CommandResult(status="success", message="")))
    return rows

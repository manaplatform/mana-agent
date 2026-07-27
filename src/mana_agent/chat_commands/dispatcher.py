from __future__ import annotations

import shlex
from typing import Protocol

from mana_agent.chat_commands.models import CommandContext, CommandInvocation, CommandResult
from mana_agent.chat_commands.registry import CommandRegistry


class NaturalLanguageCommandResolver(Protocol):
    def resolve_command(self, text: str, *, commands: tuple[str, ...], context: CommandContext) -> CommandInvocation | None: ...


class CommandDispatcher:
    def __init__(self, registry: CommandRegistry, *, natural_language_resolver: NaturalLanguageCommandResolver | None = None) -> None:
        self.registry = registry
        self.natural_language_resolver = natural_language_resolver

    def dispatch(self, text: str, context: CommandContext, *, confirmed: bool = False) -> CommandResult | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("/"):
            try:
                parts = shlex.split(raw[1:], posix=True)
            except ValueError as exc:
                return CommandResult(status="error", message=f"Invalid command arguments: {exc}")
            invocation = CommandInvocation(name=parts[0] if parts else "", arguments=parts[1:], confirmed=confirmed)
        else:
            if self.natural_language_resolver is None:
                return None
            invocation = self.natural_language_resolver.resolve_command(
                raw,
                commands=tuple(item.canonical_name for item in self.registry.definitions()),
                context=context,
            )
            if invocation is None:
                return None
        definition = self.registry.resolve(invocation.name)
        if definition is None:
            return CommandResult(status="error", message=f"Unsupported command: /{invocation.name}. Use /help to list supported commands.")
        if context.frontend not in definition.frontends:
            return CommandResult(status="error", message=f"/{definition.canonical_name} is intentionally unavailable in {context.frontend}.")
        if definition.required_capability not in context.capabilities and "*" not in context.capabilities:
            return CommandResult(status="error", message=f"Missing capability: {definition.required_capability}.")
        try:
            validated_arguments = definition.argument_model.model_validate({"values": invocation.arguments})
            invocation.arguments = list(getattr(validated_arguments, "values", invocation.arguments))
        except ValueError as exc:
            return CommandResult(status="error", message=f"Invalid /{definition.canonical_name} arguments: {exc}")
        action_requires_confirmation = bool(
            invocation.arguments
            and invocation.arguments[0].lower() in definition.confirmation_actions
        )
        if (definition.confirmation_required or action_requires_confirmation) and not invocation.confirmed:
            return CommandResult(
                status="confirmation_required",
                message=f"Confirm /{definition.canonical_name} before this irreversible action is executed.",
                data={"command": definition.canonical_name, "arguments": invocation.arguments},
            )
        try:
            return definition.handler(context, invocation.arguments)
        except (FileNotFoundError, ValueError, RuntimeError, PermissionError) as exc:
            return CommandResult(status="error", message=str(exc))

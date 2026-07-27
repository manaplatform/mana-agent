from __future__ import annotations

from collections.abc import Iterable

from mana_agent.chat_commands.models import CommandContext, CommandDefinition, CommandResult


class CommandRegistry:
    def __init__(self, definitions: Iterable[CommandDefinition] = ()) -> None:
        self._commands: dict[str, CommandDefinition] = {}
        self._aliases: dict[str, str] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: CommandDefinition) -> None:
        name = definition.canonical_name.strip().lower().lstrip("/")
        if not name or name in self._commands or name in self._aliases:
            raise ValueError(f"duplicate or invalid command: {name}")
        self._commands[name] = definition
        for raw in definition.aliases:
            alias = raw.strip().lower().lstrip("/")
            if not alias or alias in self._commands or alias in self._aliases:
                raise ValueError(f"duplicate or invalid command alias: {alias}")
            self._aliases[alias] = name

    def resolve(self, name: str) -> CommandDefinition | None:
        clean = str(name or "").strip().lower().lstrip("/")
        return self._commands.get(self._aliases.get(clean, clean))

    def definitions(self) -> tuple[CommandDefinition, ...]:
        return tuple(sorted(self._commands.values(), key=lambda item: item.canonical_name))

    def help_result(self, context: CommandContext) -> CommandResult:
        available = [
            item for item in self.definitions()
            if context.frontend in item.frontends
        ]
        lines = [f"/{item.canonical_name} {item.argument_schema}".rstrip() + f" — {item.description}" for item in available]
        commands = []
        for item in available:
            payload = item.model_dump(exclude={"handler", "argument_model"}, mode="json")
            payload["typed_argument_schema"] = item.argument_model.model_json_schema()
            commands.append(payload)
        return CommandResult(status="success", message="\n".join(lines), data={"commands": commands})


def build_default_registry() -> CommandRegistry:
    from mana_agent.chat_commands.builtins import definitions

    registry = CommandRegistry(definitions())
    help_definition = registry.resolve("help")
    if help_definition is not None:
        help_definition.handler = lambda context, _args: registry.help_result(context)
    return registry

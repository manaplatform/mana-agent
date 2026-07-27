from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegisteredBackgroundCommand:
    identifier: str
    runner: Callable[[dict[str, str]], Any]
    secret_environment_names: Callable[[dict[str, str]], tuple[str, ...]] = lambda _args: ()


def _telegram(args: dict[str, str]) -> None:
    from mana_agent.connectors.telegram.config import load_telegram_config
    from mana_agent.connectors.telegram.connector import TelegramConnector

    config = load_telegram_config()
    config.validate_runtime()

    async def run() -> None:
        if config.effective_transport == "webhook":
            import uvicorn
            from mana_agent.api.app import create_app

            server = uvicorn.Server(uvicorn.Config(create_app(telegram_config=config), host=config.webhook.listen_host, port=config.webhook.listen_port))
            await server.serve()
        else:
            connector = TelegramConnector(config)
            await connector.start()
            try:
                assert connector._transport_task is not None
                await connector._transport_task
            finally:
                await connector.stop(remove_webhook=False)

    asyncio.run(run())


def _telegram_secrets(_args: dict[str, str]) -> tuple[str, ...]:
    from mana_agent.connectors.telegram.config import load_telegram_config

    config = load_telegram_config()
    return tuple(name for name in (config.bot_token_env, config.webhook.secret_env) if name)


_COMMANDS = {
    "connector.telegram": RegisteredBackgroundCommand("connector.telegram", _telegram, _telegram_secrets),
}


def get_registered_command(identifier: str) -> RegisteredBackgroundCommand:
    try:
        return _COMMANDS[identifier]
    except KeyError as exc:
        raise ValueError(f"unregistered background command: {identifier}") from exc


def registered_identifiers() -> tuple[str, ...]:
    return tuple(sorted(_COMMANDS))

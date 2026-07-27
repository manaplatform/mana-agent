"""CLI surfaces for ACP and A2A protocol adapters."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from mana_agent.config.user_config import load_effective_settings


acp_app = typer.Typer(help="Serve Mana-Agent to ACP-compatible editors and IDEs.")
a2a_app = typer.Typer(help="Serve and invoke Agent2Agent 1.0 agents.")
a2a_agents_app = typer.Typer(help="Manage registered remote A2A agents.")
a2a_app.add_typer(a2a_agents_app, name="agents")


@acp_app.command("serve")
def acp_serve(repo: str = typer.Option(".", "--repo", help="Approved repository root.")) -> None:
    """Serve ACP v1 over stdio; stdout is reserved for JSON-RPC."""
    from mana_agent.protocols.acp.server import run_acp_stdio

    try:
        run_acp_stdio(Path(repo).expanduser().resolve())
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@acp_app.command("info")
def acp_info() -> None:
    from mana_agent.protocols.acp.server import acp_sdk_info

    typer.echo(json.dumps(acp_sdk_info(), sort_keys=True))


@acp_app.command("doctor")
def acp_doctor() -> None:
    from mana_agent.protocols.acp.server import acp_sdk_info

    info = acp_sdk_info()
    info["ok"] = bool(info["installed"])
    if not info["installed"]:
        info["fix"] = "Install mana-agent with the `acp` extra."
    typer.echo(json.dumps(info, sort_keys=True))
    if not info["ok"]:
        raise typer.Exit(code=1)


@a2a_app.command("serve")
def a2a_serve(
    repo: str = typer.Option(".", "--repo", help="Approved repository root."),
    host: str | None = typer.Option(None, "--host", help="Safe bind address; defaults to managed configuration."),
    port: int | None = typer.Option(None, "--port"),
    public_base_url: str | None = typer.Option(None, "--public-base-url"),
    token: str = typer.Option("", "--token", envvar="MANA_A2A_SERVER_TOKEN", hide_input=True),
) -> None:
    """Serve authenticated A2A 1.0 JSON-RPC and HTTP+JSON endpoints."""
    import uvicorn
    from mana_agent.protocols.a2a.server import create_a2a_app

    settings = load_effective_settings(include_env=True)
    configured = str(settings.get("MANA_A2A_SERVER_TOKEN") or "")
    resolved_host = str(host or settings.get("MANA_A2A_HOST") or "127.0.0.1")
    resolved_port = int(port or settings.get("MANA_A2A_PORT") or 8766)
    resolved_public_url = str(public_base_url or settings.get("MANA_A2A_PUBLIC_BASE_URL") or f"http://127.0.0.1:{resolved_port}")
    enabled_skills = {item.strip() for item in str(settings.get("MANA_A2A_ENABLED_SKILLS") or "").split(",") if item.strip()} or None
    try:
        app = create_a2a_app(
            root=repo,
            public_base_url=resolved_public_url,
            token=token or configured,
            enabled_skills=enabled_skills,
            max_concurrent_tasks=int(settings.get("MANA_A2A_MAX_CONCURRENT_TASKS") or 4),
            max_request_bytes=int(settings.get("MANA_A2A_MAX_REQUEST_BYTES") or 1_048_576),
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if resolved_host not in {"127.0.0.1", "::1", "localhost"} and not resolved_public_url.startswith("https://"):
        raise typer.BadParameter("Non-local A2A binding requires an HTTPS public base URL and TLS termination.")
    uvicorn.run(app, host=resolved_host, port=resolved_port)


@a2a_app.command("card")
def a2a_card(public_base_url: str = typer.Option("http://127.0.0.1:8766", "--public-base-url")) -> None:
    try:
        from google.protobuf.json_format import MessageToDict
        from mana_agent.protocols.a2a.agent_card import build_agent_card

        payload = MessageToDict(build_agent_card(public_base_url=public_base_url), preserving_proto_field_name=False)
    except ImportError as exc:
        typer.echo("A2A support is not installed. Install mana-agent with the `a2a` extra.", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@a2a_app.command("doctor")
def a2a_doctor(public_base_url: str = typer.Option("http://127.0.0.1:8766", "--public-base-url")) -> None:
    from mana_agent.protocols.a2a.server import a2a_sdk_info

    info = a2a_sdk_info()
    info["ok"] = bool(info["installed"])
    info["tls_warning"] = not public_base_url.startswith("https://")
    if not info["installed"]:
        info["fix"] = "Install mana-agent with the `a2a` extra."
    typer.echo(json.dumps(info, sort_keys=True))
    if not info["ok"]:
        raise typer.Exit(code=1)


@a2a_agents_app.command("add")
def a2a_agents_add(
    agent_card_url: str = typer.Argument(...),
    name: str = typer.Option(..., "--name"),
    skill: list[str] = typer.Option([], "--skill"),
    trust: bool = typer.Option(False, "--trust"),
) -> None:
    from mana_agent.protocols.a2a.registry import RemoteAgentRegistry

    record = RemoteAgentRegistry().add(name=name, card_url=agent_card_url, allowed_skills=skill, trusted=trust)
    typer.echo(json.dumps(record.to_dict(), indent=2, sort_keys=True))


@a2a_agents_app.command("list")
def a2a_agents_list() -> None:
    from mana_agent.protocols.a2a.registry import RemoteAgentRegistry

    typer.echo(json.dumps([item.to_dict() for item in RemoteAgentRegistry().list()], indent=2, sort_keys=True))


@a2a_agents_app.command("inspect")
def a2a_agents_inspect(name: str = typer.Argument(...), refresh: bool = typer.Option(False, "--refresh")) -> None:
    from google.protobuf.json_format import MessageToDict
    from mana_agent.protocols.a2a.client import ManaA2AClient

    card = asyncio.run(ManaA2AClient().discover(name, refresh=refresh))
    typer.echo(json.dumps(MessageToDict(card), indent=2, sort_keys=True))


@a2a_agents_app.command("remove")
def a2a_agents_remove(name: str = typer.Argument(...)) -> None:
    from mana_agent.protocols.a2a.registry import RemoteAgentRegistry

    RemoteAgentRegistry().remove(name)
    typer.echo(json.dumps({"removed": name}))


@a2a_app.command("delegate")
def a2a_delegate(
    name: str = typer.Argument(...),
    task: str = typer.Argument(...),
    skill: str = typer.Option(..., "--skill", help="Explicit remote skill selected for this handoff."),
    repo: str = typer.Option(".", "--repo", help="Workspace used for the local task-board record."),
    token: str = typer.Option("", "--token", hide_input=True),
    allow_local: bool = typer.Option(False, "--allow-local", help="Permit loopback endpoints for development."),
) -> None:
    from google.protobuf.json_format import MessageToDict
    from mana_agent.protocols.a2a.client import ManaA2AClient
    from mana_agent.protocols.a2a.delegation import DelegationPolicy, RemoteDelegationService

    client = ManaA2AClient(allow_local_urls=allow_local)
    remote = client.registry.get(name)
    policy = DelegationPolicy(enabled=True, allowed_skills=frozenset({skill}))
    events = asyncio.run(
        RemoteDelegationService(root=repo, client=client, policy=policy).delegate(
            remote,
            task=task,
            skill=skill,
            bearer_token=token,
        )
    )
    typer.echo(json.dumps([MessageToDict(event) for event in events], indent=2, sort_keys=True))

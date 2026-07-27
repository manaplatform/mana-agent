from __future__ import annotations

import json

import typer

from mana_agent.config.settings import Settings
from mana_agent.github_autopilot import GitHubAutopilotService, GitHubAutopilotSettings

github_app = typer.Typer(help="Operate the webhook-driven Mana GitHub App.")


def _service() -> GitHubAutopilotService:
    return GitHubAutopilotService(GitHubAutopilotSettings.from_mana_settings(Settings()))


@github_app.command("serve")
def serve(host: str = typer.Option("127.0.0.1", "--host"), port: int = typer.Option(8000, "--port")) -> None:
    """Serve the API with GitHub webhook workers enabled."""
    import uvicorn

    settings = GitHubAutopilotSettings.from_mana_settings(Settings())
    errors = settings.startup_errors()
    if errors:
        for error in errors:
            typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2)
    uvicorn.run("mana_agent.api.app:app", host=host, port=port)


@github_app.command("doctor")
def doctor(no_auth: bool = typer.Option(False, "--no-auth")) -> None:
    report = _service().doctor(authenticate=not no_auth)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    raise typer.Exit(0 if report.get("ok") else 1)


@github_app.command("deliveries")
def deliveries() -> None:
    service = _service()
    rows = []
    for path in sorted(service.store.deliveries.glob("*.json"), reverse=True):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@github_app.command("jobs")
def jobs() -> None:
    typer.echo(json.dumps([job.model_dump(mode="json") for job in _service().store.list_jobs()], indent=2, sort_keys=True))


@github_app.command("retry")
def retry(job_id: str) -> None:
    import asyncio

    job = asyncio.run(_service().retry(job_id))
    typer.echo(json.dumps(job.model_dump(mode="json"), indent=2, sort_keys=True))


@github_app.command("cancel")
def cancel(job_id: str) -> None:
    job = _service().cancel(job_id)
    typer.echo(json.dumps(job.model_dump(mode="json"), indent=2, sort_keys=True))

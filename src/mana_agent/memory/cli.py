"""Capsule operations through the authenticated API service."""

from __future__ import annotations

import json
import os
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer

from mana_agent.memory.capsules.models import CapsuleScope, MergeStrategy


memory_app = typer.Typer(help="Inspect and review authorized memory state.")
capsules_app = typer.Typer(help="Operate on scoped memory capsules through the Mana API.")
memory_app.add_typer(capsules_app, name="capsules")


def _api(path: str, *, base_url: str, body: dict | None = None):
    base = str(base_url or os.getenv("MANA_API_BASE") or "").rstrip("/")
    if not base:
        raise typer.BadParameter("Set --api-base or MANA_API_BASE; direct provider access is not allowed.")
    token = str(os.getenv("MANA_API_TOKEN") or "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"{base}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310 - explicit Mana API
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Capsule API request failed: {exc}") from exc


def _show(value) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@capsules_app.command("list")
def list_capsules(
    scope: list[CapsuleScope] = typer.Option(..., "--scope", help="Explicit allowed scope; repeat as needed."),
    query: str = typer.Option("", "--query"),
    max_capsules: int = typer.Option(12, "--limit", min=1, max=100),
    api_base: str = typer.Option("", "--api-base"),
) -> None:
    _show(_api("/api/v1/memory/capsules/query", base_url=api_base, body={
        "query": query,
        "allowed_scopes": [item.value for item in scope],
        "max_capsules": max_capsules,
        "max_tokens": 4000,
    }))


@capsules_app.command("inspect")
def inspect_capsule(capsule_id: str, api_base: str = typer.Option("", "--api-base")) -> None:
    _show(_api(f"/api/v1/memory/capsules/{capsule_id}", base_url=api_base))


@capsules_app.command("lineage")
def inspect_lineage(capsule_id: str, api_base: str = typer.Option("", "--api-base")) -> None:
    _show(_api(f"/api/v1/memory/capsules/{capsule_id}/lineage", base_url=api_base))


@capsules_app.command("staged")
def list_staged(api_base: str = typer.Option("", "--api-base")) -> None:
    _show(_api("/api/v1/memory/capsules/staged", base_url=api_base))


@capsules_app.command("review")
def review_capsule(
    capsule_id: str,
    strategy: MergeStrategy = typer.Option(..., "--strategy", help="append, replace, patch, supersede, or reject"),
    reason: str = typer.Option(..., "--reason"),
    target_capsule_id: str = typer.Option("", "--target"),
    expected_revision: int = typer.Option(0, "--expected-revision", min=0),
    expected_hash: str = typer.Option("", "--expected-hash"),
    api_base: str = typer.Option("", "--api-base"),
) -> None:
    _show(_api(f"/api/v1/memory/capsules/{capsule_id}/merge", base_url=api_base, body={
        "request_id": f"cli-{uuid.uuid4().hex}",
        "strategy": strategy.value,
        "decision_reason": reason,
        "target_capsule_id": target_capsule_id or None,
        "expected_target_revision": expected_revision or None,
        "expected_target_hash": expected_hash or None,
    }))

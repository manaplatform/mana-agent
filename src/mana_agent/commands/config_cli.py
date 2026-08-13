from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from mana_agent.config import user_config
from mana_agent.config.settings import Settings
from mana_agent.config.migrations.registry import registry

app = typer.Typer(help="Manage Mana-Agent configuration")

@app.command("schema")
def config_schema(
    json_output: bool = typer.Option(False, "--json", help="Output JSON schema instead of human readable"),
    output: Path = typer.Option(None, "--output", help="Write schema to file")
) -> None:
    """Export the configuration schema."""
    schema = Settings.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    
    if output:
        output.write_text(schema_str + "\n")
        typer.echo(f"Schema written to {output}")
    elif json_output:
        typer.echo(schema_str)
    else:
        # A human readable printout of the fields and types
        typer.echo("Mana-Agent Configuration Schema")
        typer.echo(f"Version: {schema.get('properties', {}).get('mana_config_schema_version', {}).get('default', 2)}")
        typer.echo("---")
        for prop, details in schema.get("properties", {}).items():
            prop_type = details.get("type", "any")
            if "anyOf" in details:
                prop_type = " | ".join(d.get("type", "any") for d in details["anyOf"])
            desc = details.get("description", "")
            default = details.get("default", "none")
            
            typer.echo(f"{prop}: {prop_type}")
            if desc:
                typer.echo(f"  Description: {desc}")
            typer.echo(f"  Default: {default}")
            typer.echo("")


@app.command("validate")
def config_validate() -> None:
    """Pre-flight static configuration validation."""
    try:
        values = user_config.load_effective_settings()
        user_config.validate_config_values(values)
        typer.echo("Configuration validates successfully.")
        raise typer.Exit(code=0)
    except Exception as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command("explain")
def config_explain() -> None:
    """Explain configuration precedence and sources (secrets redacted)."""
    typer.echo("Configuration precedence:")
    typer.echo("1. ~/.mana/secrets.toml (highest)")
    typer.echo("2. ~/.mana/config.toml")
    typer.echo("3. Defaults (lowest)\n")
    
    settings = Settings()
    # Mask secrets
    for secret in user_config.SECRET_KEYS:
        env_key = user_config.FIELD_NAME_BY_ENV.get(secret)
        if env_key and getattr(settings, env_key, None):
            setattr(settings, env_key, "<set>")

    typer.echo("Effective configuration (secrets redacted):")
    for k, v in settings.model_dump(by_alias=True).items():
        if v:
            typer.echo(f"{k} = {v}")
    raise typer.Exit(code=0)


@app.command("migrate")
def config_migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show proposed changes without modifying"),
    check: bool = typer.Option(False, "--check", help="Exit with non-zero status if migrations are needed")
) -> None:
    """Migrate configuration to the latest version."""
    config_file = user_config.config_file()
    raw_config = user_config.load_user_config()
    target_version = Settings.model_fields['mana_config_schema_version'].default
    
    new_config, explanations = registry.migrate(raw_config, target_version, dry_run=True)
    
    if not explanations:
        typer.echo("Configuration is already up to date.")
        raise typer.Exit(code=0)

    if check:
        typer.echo("Migrations are required:")
        for explanation in explanations:
            typer.echo(f"- {explanation}")
        raise typer.Exit(code=1)

    typer.echo("Proposed migrations:")
    for explanation in explanations:
        typer.echo(f"- {explanation}")

    if not dry_run:
        # Re-run without dry_run to actually apply
        new_config, _ = registry.migrate(raw_config, target_version, dry_run=False)
        # Create a backup
        import shutil
        from datetime import datetime, timezone
        backup_file = config_file.with_name(f"{config_file.name}.migrate-{datetime.now(timezone.utc):%Y%m%d%H%M%S}.bak")
        shutil.copy2(config_file, backup_file)
        typer.echo(f"Created backup at {backup_file}")
        
        user_config._write_toml(config_file, new_config)
        typer.echo("Migrations applied successfully.")
    else:
        typer.echo("Dry run completed. No changes made.")

    raise typer.Exit(code=0)


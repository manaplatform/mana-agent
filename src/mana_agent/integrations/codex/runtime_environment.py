"""Lifecycle and environment isolation for a single managed Codex process."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mana_agent.compat import tomllib
from mana_agent.config.settings import mana_home
from mana_agent.integrations.codex.exceptions import CodexConfigurationError
from mana_agent.integrations.codex.runtime_config import CodexRuntimeConfig, RUNTIME_API_KEY_ENV

_REMOVED_ENVIRONMENT_KEYS = {
    "CODEX_HOME",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
}


@dataclass(slots=True)
class CodexRuntimeContext:
    config: CodexRuntimeConfig
    home: Path
    environment: dict[str, str] = field(repr=False)
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            shutil.rmtree(self.home, ignore_errors=True)
            self._closed = True

    def __enter__(self) -> "CodexRuntimeContext":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class CodexRuntimeEnvironment:
    @staticmethod
    def create(config: CodexRuntimeConfig) -> CodexRuntimeContext:
        root = mana_home() / "runtime" / "codex"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        home = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        home.chmod(0o700)
        # Keep the secret only in the child environment. The on-disk config is
        # rendered from non-secret fields so credentials are never written in clear text.
        api_key = config.api_key
        try:
            from dataclasses import replace

            non_secret = replace(config, api_key="")
            rendered = non_secret.to_toml()
            tomllib.loads(rendered)
            if api_key and api_key in rendered:
                raise CodexConfigurationError(
                    "Refusing to write Codex configuration that contains credential material."
                )
            config_path = home / "config.toml"
            config_path.write_text(rendered, encoding="utf-8")
            config_path.chmod(0o600)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            shutil.rmtree(home, ignore_errors=True)
            raise CodexConfigurationError(
                "Unable to create valid isolated Codex configuration."
            ) from exc
        except CodexConfigurationError:
            shutil.rmtree(home, ignore_errors=True)
            raise
        environment = {
            key: value
            for key, value in os.environ.copy().items()
            if key not in _REMOVED_ENVIRONMENT_KEYS
        }
        environment["CODEX_HOME"] = str(home)
        environment[RUNTIME_API_KEY_ENV] = api_key
        return CodexRuntimeContext(config=config, home=home, environment=environment)


@contextmanager
def isolated_codex_probe_environment() -> Iterator[dict[str, str]]:
    """Isolate read-only Codex executable probes from the user's global config."""

    root = mana_home() / "runtime" / "codex"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="probe-", dir=root))
    home.chmod(0o700)
    environment = {
        key: value for key, value in os.environ.copy().items() if key not in _REMOVED_ENVIRONMENT_KEYS
    }
    environment["CODEX_HOME"] = str(home)
    try:
        yield environment
    finally:
        shutil.rmtree(home, ignore_errors=True)


__all__ = [
    "CodexRuntimeContext",
    "CodexRuntimeEnvironment",
    "isolated_codex_probe_environment",
]

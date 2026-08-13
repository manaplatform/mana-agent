from __future__ import annotations

import copy
from typing import Any


class Migration:
    """Base class for a configuration migration."""
    
    @property
    def source_version(self) -> int:
        raise NotImplementedError

    @property
    def target_version(self) -> int:
        raise NotImplementedError

    def check(self, config: dict[str, Any]) -> bool:
        """Return True if this migration should be applied."""
        return config.get("MANA_CONFIG_SCHEMA_VERSION", 1) == self.source_version

    def apply(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply the migration to the configuration, returning the mutated config."""
        raise NotImplementedError

    def explain(self) -> str:
        """Return a string explaining what this migration does."""
        raise NotImplementedError


class ConfigurationMigrationRegistry:
    def __init__(self) -> None:
        self._migrations: list[Migration] = []

    def register(self, migration: Migration) -> None:
        self._migrations.append(migration)

    def get_migrations(self, current_version: int, target_version: int) -> list[Migration]:
        # Simple linear search for now, assuming sequential migrations (1->2, 2->3)
        applicable = []
        cv = current_version
        
        # Sort migrations by source version to ensure we apply them in order
        sorted_migrations = sorted(self._migrations, key=lambda m: m.source_version)
        
        for m in sorted_migrations:
            if m.source_version == cv and m.target_version <= target_version:
                applicable.append(m)
                cv = m.target_version
        
        # If we couldn't reach the target or didn't find any applicable, return what we found (or empty)
        # In a real graph this would be BFS/DFS.
        return applicable

    def migrate(self, config: dict[str, Any], target_version: int, dry_run: bool = False) -> tuple[dict[str, Any], list[str]]:
        current_version = config.get("MANA_CONFIG_SCHEMA_VERSION", 1)
        if current_version >= target_version:
            return config, []

        migrations = self.get_migrations(current_version, target_version)
        if not migrations:
            return config, []

        new_config = copy.deepcopy(config)
        explanations = []

        for m in migrations:
            if m.check(new_config):
                explanations.append(m.explain())
                if not dry_run:
                    new_config = m.apply(new_config)
                    new_config["MANA_CONFIG_SCHEMA_VERSION"] = m.target_version

        return new_config, explanations


registry = ConfigurationMigrationRegistry()

# Example migration (V1 to V2) - can be extracted to a separate file later if needed
class MigrateV1ToV2(Migration):
    @property
    def source_version(self) -> int:
        return 1
    
    @property
    def target_version(self) -> int:
        return 2

    def apply(self, config: dict[str, Any]) -> dict[str, Any]:
        # Perform actual v1 to v2 migration logic here.
        # e.g. renaming deprecated keys
        if "search.provider_name" in config:
            config["MANA_WEB_SEARCH_PROVIDER"] = config.pop("search.provider_name")
        return config

    def explain(self) -> str:
        return "Migrate configuration from version 1 to 2. Renames deprecated keys (e.g. search.provider_name -> MANA_WEB_SEARCH_PROVIDER)."

registry.register(MigrateV1ToV2())

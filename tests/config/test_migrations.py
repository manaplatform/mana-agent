from mana_agent.config.migrations.registry import registry, MigrateV1ToV2


def test_migration_registry():
    config = {"MANA_CONFIG_SCHEMA_VERSION": 1, "search.provider_name": "google"}
    new_config, explanations = registry.migrate(config, target_version=2)
    
    assert new_config["MANA_CONFIG_SCHEMA_VERSION"] == 2
    assert "MANA_WEB_SEARCH_PROVIDER" in new_config
    assert new_config["MANA_WEB_SEARCH_PROVIDER"] == "google"
    assert "search.provider_name" not in new_config
    assert len(explanations) == 1

def test_migration_registry_already_migrated():
    config = {"MANA_CONFIG_SCHEMA_VERSION": 2, "search.provider_name": "google"}
    new_config, explanations = registry.migrate(config, target_version=2)
    
    assert new_config["MANA_CONFIG_SCHEMA_VERSION"] == 2
    assert "search.provider_name" in new_config
    assert "MANA_WEB_SEARCH_PROVIDER" not in new_config
    assert len(explanations) == 0

def test_migration_registry_dry_run():
    config = {"MANA_CONFIG_SCHEMA_VERSION": 1, "search.provider_name": "google"}
    new_config, explanations = registry.migrate(config, target_version=2, dry_run=True)
    
    assert new_config["MANA_CONFIG_SCHEMA_VERSION"] == 1
    assert "MANA_WEB_SEARCH_PROVIDER" not in new_config
    assert "search.provider_name" in new_config
    assert len(explanations) == 1

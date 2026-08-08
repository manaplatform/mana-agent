"""SWE-bench Verified prediction generation for mana-agent.

Predictions include ``agent_name`` (default ``mana-agent``), the LLM as
``agent_model``, optional ``agent_provider``, and a harness system id
``model_name_or_path`` defaulting to ``{agent_name}__{agent_model}`` (provider
qualified when non-OpenAI).

When ``--model`` / ``--provider`` are omitted, values come from
``~/.mana/config.toml`` (``MANA_PRIMARY_MODEL`` / ``MANA_AI_PROVIDER``).
If only ``--model`` is set, the configured provider is still used.

Instance selection: omit ``--instance-ids`` to run every id from the SWE-bench
dataset split; pass ``--instance-ids`` / ``--instance-ids-file`` to run only
those specific ids.
"""

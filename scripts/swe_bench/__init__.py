"""SWE-bench Verified prediction generation for mana-agent.

Predictions include ``agent_name`` (default ``mana-agent``), the forced LLM as
``agent_model``, and a harness system id ``model_name_or_path`` defaulting to
``{agent_name}__{agent_model}``.

Instance selection: omit ``--instance-ids`` to run every id from the SWE-bench
dataset split; pass ``--instance-ids`` / ``--instance-ids-file`` to run only
those specific ids.
"""

"""Machine-readable Teach Mode capability contracts."""

from mana_agent.tools.contracts import ToolContract


def teach_tool_contracts() -> list[ToolContract]:
    names = {
        "teach_start": "Start a visible local semantic demonstration recording.",
        "teach_stop": "Stop recording and compile a draft workflow for review.",
        "teach_explain": "Add a typed explanation to parameterize the demonstration.",
        "teach_list_flows": "List reusable local Mana Flows.",
        "teach_replay": "Replay a selected flow with verification-driven status.",
    }
    schema = {
        "type": "object",
        "properties": {"source_decision_id": {"type": "string", "minLength": 1}},
        "required": ["source_decision_id"],
        "additionalProperties": True,
    }
    return [
        ToolContract(
            name=name,
            description=description,
            input_schema=schema,
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            error_format={"type": "object"},
            safety_rules=[
                "Execute only after a validated model decision selects this exact tool.",
                "Never report replay success unless final observable verification passed.",
                "Never expose recorded secrets or bypass confirmation requirements.",
            ],
        )
        for name, description in names.items()
    ]

from __future__ import annotations

from typing import Any
from mana_agent.tools.contracts import ToolContract


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def browser_tool_contracts() -> list[ToolContract]:
    common = {"ok": {"type": "boolean"}, "error_code": {"type": "string"}, "message": {"type": "string"}}
    output = _schema({**common, "url": {"type": "string"}, "page_version": {"type": "integer"}, "tab_id": {"type": "string"}}, [])
    error = _schema(common, ["ok", "error_code", "message"])
    session = {"session_id": {"type": "string", "minLength": 1}}
    inspected = {**session, "tab_id": {"type": ["string", "null"]}}
    action = {
        **inspected,
        "target": {"type": "string"}, "value": {}, "observed_page_version": {"type": ["integer", "null"]},
        "expected_origin": {"type": ["string", "null"]}, "risk": {"enum": ["read_only", "reversible", "sensitive", "irreversible"]},
        "confirmation_required": {"type": "boolean"}, "approval_token": {"type": ["string", "null"]}, "timeout_ms": {"type": ["integer", "null"]},
    }
    safety = ["Execute only after a validated model decision selects this exact tool and arguments.", "Never bypass CAPTCHA, MFA, access restrictions, or security controls.", "Sensitive and irreversible terminal actions require an exact-action approval token.", "Do not expose credentials, cookies, or sensitive form values in results or traces."]
    specs = [
        ("browser_open", "Open an HTTP(S) page in an isolated browser session.", {**session, "url": {"type": "string"}, "profile_name": {"type": ["string", "null"]}}, ["session_id", "url"]),
        ("browser_inspect", "Inspect current page text, accessibility semantics, forms, controls, links, and refs.", inspected, ["session_id"]),
        *[(f"browser_{name}", f"Perform the model-selected {name} action using current page evidence.", action, ["session_id"]) for name in ("click", "type", "select", "scroll", "wait", "download", "back")],
        ("browser_screenshot", "Capture the current page to the private session artifact directory.", {**inspected, "full_page": {"type": "boolean"}}, ["session_id"]),
        ("browser_upload", "Upload an allowed local file using a current file input ref.", {**session, "ref": {"type": "string"}, "path": {"type": "string"}, "observed_page_version": {"type": "integer"}, "tab_id": {"type": ["string", "null"]}}, ["session_id", "ref", "path", "observed_page_version"]),
        ("browser_check_links", "Validate rendered HTTP(S) links without navigating the active page.", {**inspected, "max_links": {"type": "integer", "minimum": 1, "maximum": 100}}, ["session_id"]),
        ("browser_tabs", "List tabs and popups in the selected session.", session, ["session_id"]),
        ("browser_switch_tab", "Switch to a selected tab id.", {**session, "tab_id": {"type": "string"}}, ["session_id", "tab_id"]),
        ("browser_close", "Close and clean the selected browser session.", session, ["session_id"]),
    ]
    return [ToolContract(name=name, description=description, input_schema=_schema(props, required), output_schema=output, error_format=error, safety_rules=safety) for name, description, props, required in specs]

"""Canned native responses for gateway tests that need repository evidence."""

from types import SimpleNamespace

from mana_agent.llm.messages import ToolCall
from mana_agent.llm.types import ModelResponse


def investigation(scope, **fields):
    return {"action": "investigate", "reason": "Repository evidence is needed",
            "objective": "Explain the current behavior", "scope": scope, **fields}


def evidence_call(identity, tool="read_file", **arguments):
    return ToolCall(id=identity, name=tool, args={"level": 1, "reason": "Observe the named target", **arguments})


def inspection(path=None, *calls):
    if path is not None:
        calls = (evidence_call("read", paths=[path]), *calls)
    return {"action": "inspect", "reason": "Read evidence needed for the next decision",
            "requests": [{"tool": call.name, **call.args} for call in calls]}


def report_call(*references):
    return ToolCall(id="report-1", name="report_investigation", args={
        "findings": "Observed the current repository behavior", "evidence_ids": list(references), "gaps": "",
    })


class FileInvestigator:
    investigation_path = "app.py"

    def profile(self):
        return SimpleNamespace(require=lambda *_: None, supports=lambda _: False)

    def with_tools(self, tools):
        assert {"read_file", "report_investigation"}.issubset({tool["name"] for tool in tools})

        def generate(messages, *, continuation):
            if len(messages) == 2:
                return ModelResponse(tool_calls=[evidence_call("read-1", paths=[self.investigation_path])])
            return ModelResponse(tool_calls=[report_call("evidence-1")])

        return SimpleNamespace(generate_sync=generate)

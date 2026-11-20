"""Bound, progressively loaded evidence tools for the native coding engine."""

from dataclasses import dataclass

from mana_agent.repository_evidence.models import EvidenceDetailQuery, EvidenceQuery


@dataclass(frozen=True)
class EvidenceContextTool:
    name: str
    description: str
    input_model: type
    invoke: object

    def call(self, arguments):
        return self.invoke(self.input_model.model_validate(arguments))


def context_tools(board):
    def detail(query):
        try:
            return board.detail(query)
        except KeyError:
            return {"ok": False, "error_code": "UNKNOWN_EVIDENCE_REF",
                    "error": (f"Evidence reference {query.evidence_id}:{query.revision} is unavailable in this repository. "
                              "Use an evidence_id and revision returned together by repository_evidence_query, "
                              "or use the original tool observation if it is still current.")}

    return [
        EvidenceContextTool("repository_evidence_query", "Query scoped repository evidence; results and context are bounded.", EvidenceQuery, board.query),
        EvidenceContextTool("repository_evidence_get", "Read a bounded page of a cited evidence revision and provenance.",
                            EvidenceDetailQuery, detail),
    ]

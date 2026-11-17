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
    return [
        EvidenceContextTool("repository_evidence_query", "Query scoped repository evidence; results and context are bounded.", EvidenceQuery, board.query),
        EvidenceContextTool("repository_evidence_get", "Read a bounded page of a cited evidence revision and provenance.",
                            EvidenceDetailQuery, board.detail),
    ]

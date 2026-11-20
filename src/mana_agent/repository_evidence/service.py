"""Durable repository facts with source freshness and supervisor-fenced publication."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import uuid

from mana_agent.repository_evidence.models import (
    EvidenceCandidate, EvidenceDetailQuery, EvidenceProvenance, EvidenceQuery, EvidenceRecord, EvidenceRef,
    EvidenceRelation, EvidenceRequirement, EvidenceSource,
)
from mana_agent.utils.redaction import redact_secrets


class EvidenceConflict(RuntimeError):
    """A concurrent writer advanced an evidence or task revision."""


class StaleEvidenceSource(ValueError):
    """A source changed after its read completed."""


class RepositoryEvidenceBoard:
    def __init__(self, *, db, repository_id: str, root: Path, publication=None):
        self.db = db
        self.repository_id = repository_id
        self.root = Path(root).resolve()
        self.publication = publication

    def _fence(self):
        return self.publication() if self.publication else nullcontext()

    def _source_current(self, source: EvidenceSource) -> bool:
        requested = self.root / source.path
        target = requested.resolve()
        if not target.is_relative_to(self.root) or target != requested:
            return False
        try:
            if source.source_kind == "directory":
                from mana_agent.repository_evidence.observations import read_directory
                return read_directory(self.root, source.path).get("source_evidence", {}).get("sha256") == source.sha256
            if not target.is_file():
                return False
            if target.stat().st_size > 2_000_000:
                return False
            from mana_agent.repository_evidence.models import relative_source
            relative_source(target.relative_to(self.root).as_posix())
            digest = sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest() == source.sha256
        except (OSError, ValueError):
            return False

    def get(self, ref: EvidenceRef, *, refresh: bool = True, _visited=None, _budget=None) -> EvidenceRecord:
        with self.db.connect() as conn:
            row = conn.execute("SELECT payload FROM repository_evidence WHERE evidence_id=? AND revision=? AND repository_id=?",
                               (ref.evidence_id, ref.revision, self.repository_id)).fetchone()
        if row is None:
            raise KeyError(f"Unknown repository evidence {ref.evidence_id}:{ref.revision}")
        record = EvidenceRecord.model_validate_json(row["payload"])
        if not refresh or record.freshness != "current":
            return record
        with self.db.connect() as conn:
            latest = conn.execute("SELECT freshness FROM repository_evidence WHERE evidence_id=? ORDER BY revision DESC LIMIT 1", (ref.evidence_id,)).fetchone()
            contradiction = conn.execute("SELECT 1 FROM evidence_relations r JOIN repository_evidence e ON e.evidence_id=r.evidence_id AND e.revision=r.revision WHERE r.target_id=? AND r.target_revision=? AND r.relation='contradicts' AND e.validation='supported' AND e.freshness='current' AND e.revision=(SELECT MAX(n.revision) FROM repository_evidence n WHERE n.evidence_id=e.evidence_id) LIMIT 1", (ref.evidence_id, ref.revision)).fetchone()
        if latest["freshness"] == "invalidated":
            return record.model_copy(update={"freshness": "invalidated", "freshness_reason": "A later revision invalidated this finding"})
        if contradiction or any(r.relation == "contradicts" for r in record.relationships):
            record = record.model_copy(update={"validation": "contradicted"})
        visited = set(_visited or ())
        if (ref.evidence_id, ref.revision) in visited:
            return record.model_copy(update={"freshness": "unknown", "freshness_reason": "Cyclic support"})
        visited.add((ref.evidence_id, ref.revision))
        budget = _budget if _budget is not None else [64]
        budget[0] -= 1
        if budget[0] < 0:
            return record.model_copy(update={"freshness": "unknown", "freshness_reason": "Supporting evidence exceeds bounded retrieval"})
        current = bool(record.sources or record.supporting_evidence) and all(self._source_current(s) for s in record.sources)
        for support in record.supporting_evidence:
            supporting = self.get(support, _visited=visited, _budget=budget)
            current = current and supporting.freshness == "current"
            if supporting.validation != "supported":
                record = record.model_copy(update={"validation": "contradicted" if supporting.validation == "contradicted" else "pending"})
        if not current:
            # Query-time freshness is a projection. Publication of a new revision
            # remains fenced; a read does not acquire write authority.
            return record.model_copy(update={"freshness": "stale", "freshness_reason": "Source or supporting evidence changed"})
        return record

    def _persist(self, record: EvidenceRecord, *, task_id: str, role: str, deduplicate=False) -> EvidenceRecord:
        clean = record.model_dump(mode="json")
        from mana_agent.services.activity_context import current_activity_context
        from mana_agent.chat.activity import safe_text
        activity = current_activity_context()
        encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True)
        if safe_text(encoded, None, protected=getattr(activity, "protected_text", ())) != encoded or redact_secrets(clean) != clean:
            raise ValueError("Protected material cannot be persisted as repository evidence")
        with self._fence(), self.db.transaction() as conn:
            prior = conn.execute("SELECT payload FROM repository_evidence WHERE evidence_id=? ORDER BY revision DESC LIMIT 1", (record.evidence_id,)).fetchone()
            latest = EvidenceRecord.model_validate_json(prior["payload"]) if prior else None
            if (deduplicate and latest is not None and (latest.freshness == "current"
                    or latest.freshness == "unknown" and record.classification == "observed")):
                conn.execute("INSERT OR IGNORE INTO task_evidence_links VALUES (?, ?, ?, ?)",
                             (task_id, latest.evidence_id, latest.revision, role))
                conn.execute("INSERT INTO task_events(task_id,event_type,payload,created_at) VALUES (?,?,?,?)",
                             (task_id, "repository.evidence.reused", json.dumps({"evidence_id": latest.evidence_id,
                              "revision": latest.revision, "provenance": clean["provenance"]}), record.discovered_at.isoformat()))
                return latest
            if deduplicate and latest is not None:
                record = record.model_copy(update={"revision": latest.revision + 1})
                clean = record.model_dump(mode="json")
                encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True)
            if (latest.revision if latest is not None else 0) != record.revision - 1:
                raise EvidenceConflict("Evidence revision changed; reload before publishing")
            conn.execute("INSERT INTO repository_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (record.evidence_id, record.revision, record.repository_id, record.source_view,
                          record.category, record.classification, record.validation, record.freshness, encoded))
            conn.executemany("INSERT INTO evidence_sources VALUES (?, ?, ?, ?, ?, ?)", [
                (record.evidence_id, record.revision, record.repository_id, s.path, s.symbol, s.sha256) for s in record.sources])
            conn.executemany("INSERT INTO evidence_dependencies VALUES (?, ?, ?)",
                             [(record.evidence_id, record.revision, dep) for dep in record.dependencies])
            relations = list(record.relationships) + [EvidenceRelation(relation="derived_from", target=s) for s in record.supporting_evidence]
            conn.executemany("INSERT OR IGNORE INTO evidence_relations VALUES (?, ?, ?, ?, ?)", [
                (record.evidence_id, record.revision, r.relation, r.target.evidence_id, r.target.revision) for r in relations])
            conn.execute("INSERT OR IGNORE INTO task_evidence_links VALUES (?, ?, ?, ?)",
                         (task_id, record.evidence_id, record.revision, role))
            conn.execute("INSERT INTO task_events (task_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                         (task_id, "repository.evidence.updated", json.dumps({"evidence_id": record.evidence_id, "revision": record.revision}), record.discovered_at.isoformat()))
        from mana_agent.services.activity_context import emit_activity
        if activity is not None:
            emit_activity("coding.progress", title="Repository evidence saved", task_id=task_id,
                summary=f"{record.classification} · {record.category} · {record.freshness}",
                metadata={"evidence_id": record.evidence_id, "evidence_revision": record.revision,
                          "validation": record.validation, "freshness": record.freshness})
        return record

    def record_observation(self, *, provenance: EvidenceProvenance, category: str,
                           statement: str, observation: dict[str, Any], sources: tuple[EvidenceSource, ...] = ()) -> EvidenceRecord:
        """Host ingestion only: callers must derive provenance from an actual receipt."""
        if not provenance.result_digest:
            raise ValueError("Repository observation requires an executed result digest")
        if not all(self._source_current(s) for s in sources):
            raise StaleEvidenceSource("Observation sources no longer match the executed read")
        identity = json.dumps([self.repository_id, str(self.root), category, statement,
            [source.model_dump(mode="json") for source in sources] if sources else
            [provenance.task_id, observation.get("content_digest") or provenance.result_digest]], sort_keys=True)
        record = EvidenceRecord(
            evidence_id=f"repo_ev_{sha256(identity.encode()).hexdigest()}", revision=1, workspace_id=self.db.workspace_id,
            repository_id=self.repository_id, source_view=str(self.root), category=category, statement=statement,
            classification="observed", validation="supported", freshness="current" if sources else "unknown",
            confidence=1, sources=sources, observation=observation, provenance=provenance,
        )
        return self._persist(record, task_id=provenance.task_id, role="produced", deduplicate=True)

    def submit_findings(self, candidate: EvidenceCandidate, *, provenance: EvidenceProvenance) -> EvidenceRecord:
        supports = [self.get(ref) for ref in candidate.supporting_evidence]
        for relation in candidate.relationships:
            self.get(relation.target)
        if candidate.classification == "inferred" and not supports:
            raise ValueError("Inferences require actual supporting observations")
        if not all(self._source_current(source) for source in candidate.sources):
            raise ValueError("Finding cites stale source content")
        supported = bool(supports) and all(s.freshness == "current" and s.validation == "supported" for s in supports)
        identity = json.dumps({"repository": self.repository_id, "source_view": str(self.root),
                               "candidate": candidate.model_dump(mode="json")}, sort_keys=True)
        record = EvidenceRecord(
            evidence_id=f"repo_finding_{sha256(identity.encode()).hexdigest()}", revision=1, workspace_id=self.db.workspace_id,
            repository_id=self.repository_id, source_view=str(self.root), category=candidate.category,
            statement=candidate.statement, classification=candidate.classification,
            validation="supported" if supported and candidate.classification == "inferred" else "pending",
            freshness="current" if supported else "unknown", confidence=candidate.confidence,
            sources=candidate.sources, supporting_evidence=candidate.supporting_evidence,
            relationships=candidate.relationships, dependencies=candidate.dependencies,
            architecture_relationships=candidate.architecture_relationships,
            limitations=candidate.limitations, errors=candidate.errors, provenance=provenance,
        )
        return self._persist(record, task_id=provenance.task_id, role="produced", deduplicate=True)

    def detail(self, query: EvidenceDetailQuery) -> dict:
        """Page detail without loading unbounded source text into model context."""
        record = self.get(EvidenceRef(evidence_id=query.evidence_id, revision=query.revision))
        encoded = json.dumps(record.model_dump(mode="json"), ensure_ascii=False).encode()
        if query.offset > len(encoded) or (query.offset < len(encoded) and encoded[query.offset] & 0xC0 == 0x80):
            raise ValueError("Detail offset must be a returned UTF-8 page boundary")
        end = min(len(encoded), query.offset + query.max_bytes)
        while end > query.offset:
            try:
                content = encoded[query.offset:end].decode()
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            content = ""
        return {"evidence_id": record.evidence_id, "revision": record.revision,
                "content": content, "offset": query.offset, "next_offset": end, "has_more": end < len(encoded)}

    def query(self, query: EvidenceQuery) -> dict[str, Any]:
        clauses = ["e.repository_id=?", "e.revision=(SELECT MAX(n.revision) FROM repository_evidence n WHERE n.evidence_id=e.evidence_id)"]
        args: list[Any] = [self.repository_id]
        for field, column in (("categories", "e.category"), ("evidence_ids", "e.evidence_id")):
            values = getattr(query, field)
            if values:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                args.extend(values)
        if query.task_id:
            clauses.append("EXISTS(SELECT 1 FROM task_evidence_links t WHERE t.evidence_id=e.evidence_id AND t.task_id=?)")
            args.append(query.task_id)
        if query.related_to:
            clauses.append("EXISTS(SELECT 1 FROM evidence_relations r WHERE r.evidence_id=e.evidence_id AND r.revision=e.revision AND (" + " OR ".join("(r.target_id=? AND r.target_revision=?)" for _ in query.related_to) + "))")
            for ref in query.related_to:
                args.extend((ref.evidence_id, ref.revision))
        if query.dependencies:
            clauses.append("EXISTS(SELECT 1 FROM evidence_dependencies d WHERE d.evidence_id=e.evidence_id AND d.revision=e.revision AND d.dependency IN (" + ",".join("?" for _ in query.dependencies) + "))")
            args.extend(query.dependencies)
        for field, column in (("paths", "s.path"), ("symbols", "s.symbol")):
            values = getattr(query, field)
            if values:
                if field == "paths":
                    clauses.append("EXISTS(SELECT 1 FROM evidence_sources s WHERE s.evidence_id=e.evidence_id AND s.revision=e.revision AND (" + " OR ".join("(s.path=? OR substr(s.path,1,?)=?)" for _ in values) + "))")
                    for value in values:
                        args.extend((value, len(value) + 1, value + "/"))
                else:
                    clauses.append(f"EXISTS(SELECT 1 FROM evidence_sources s WHERE s.evidence_id=e.evidence_id AND s.revision=e.revision AND {column} IN ({','.join('?' for _ in values)}))")
                    args.extend(values)
        if query.query:
            clauses.append("instr(lower(e.payload), lower(?))>0")
            args.append(query.query)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT e.evidence_id, e.revision FROM repository_evidence e WHERE " + " AND ".join(clauses)
                + " ORDER BY (e.source_view=?) DESC, (e.validation='supported') DESC, e.rowid DESC LIMIT ? OFFSET ?",
                (*args, str(self.root), query.limit, query.offset),
            ).fetchall()
        records, used, consumed = [], 0, 0
        # Four UTF-8 bytes per token is deliberately not assumed: bound by bytes,
        # a conservative upper bound for these context envelopes.
        budget = query.max_context_tokens
        for row in rows:
            record = self.get(EvidenceRef(evidence_id=row["evidence_id"], revision=row["revision"]))
            if not query.include_stale and record.freshness != "current":
                consumed += 1
                continue
            summary = {key: value for key, value in record.model_dump(mode="json").items()
                       if key not in {"observation", "provenance"}}
            summary["sources"] = [{"path": s.path, "source_kind": s.source_kind, "symbol": s.symbol, "sha256": s.sha256,
                                   "start_line": s.start_line, "end_line": s.end_line, "excerpt": s.excerpt[:500]} for s in record.sources]
            size = len(json.dumps(summary, ensure_ascii=False).encode())
            if used + size > budget:
                if not records:
                    summary = {"evidence_id": record.evidence_id, "revision": record.revision, "detail_required": True}
                    size = len(json.dumps(summary).encode())
                    if size <= budget:
                        records.append(summary)
                        used += size
                        consumed += 1
                break
            records.append(summary)
            used += size
            consumed += 1
        return {"records": records, "next_offset": query.offset + consumed,
                "has_more": consumed < len(rows) or len(rows) == query.limit,
                "context_bytes": used, "repository_id": self.repository_id}

    def link(self, task_id: str, refs: tuple[EvidenceRef, ...], *, role="supports") -> None:
        for ref in refs:
            self.get(ref)
        with self._fence(), self.db.transaction() as conn:
            conn.executemany("INSERT OR IGNORE INTO task_evidence_links VALUES (?, ?, ?, ?)",
                             [(task_id, ref.evidence_id, ref.revision, role) for ref in refs])

    def satisfies(self, requirement: EvidenceRequirement) -> bool:
        if not requirement.evidence:
            return False
        try:
            records = [self.get(ref) for ref in requirement.evidence]
        except KeyError:
            return False
        if not all(r.freshness == "current" and r.validation == "supported"
                   and r.classification in requirement.acceptable_classifications
                   and (not requirement.categories or r.category in requirement.categories)
                   for r in records):
            return False
        support, visited = list(records), set()
        tools, sources = set(), []
        while support:
            record = support.pop()
            key = (record.evidence_id, record.revision)
            if key in visited:
                continue
            visited.add(key)
            if len(visited) > 64 or record.freshness != "current" or record.validation != "supported":
                return False
            sources.extend(record.sources)
            if record.classification == "observed":
                tools.add(record.observation.get("tool"))
            try:
                support.extend(self.get(ref) for ref in record.supporting_evidence)
            except KeyError:
                return False
        return (set(requirement.required_tools).issubset(tools)
                and all(any(s.path == path or path == "." or s.path.startswith(path + "/")
                            for s in sources) for path in requirement.paths))

    def invalidate_sources(self, paths: list[str], *, task_id: str, reason="Observed source mutation") -> list[EvidenceRef]:
        if not paths:
            return []
        with self.db.connect() as conn:
            affected = list(dict.fromkeys([*paths, *(str(Path(path).parent) for path in paths)]))
            rows = conn.execute("SELECT DISTINCT evidence_id, revision FROM evidence_sources WHERE repository_id=? AND path IN (" + ",".join("?" for _ in affected) + ") ORDER BY revision ASC",
                                (self.repository_id, *affected)).fetchall()
        pending = [EvidenceRef(evidence_id=r["evidence_id"], revision=r["revision"]) for r in rows]
        changed, seen = [], set()
        while pending:
            ref = pending.pop()
            if ref.evidence_id in seen:
                continue
            old = self.get(ref, refresh=False)
            # A mutation in another worktree cannot invalidate this view.
            if old.source_view != str(self.root):
                continue
            with self.db.connect() as conn:
                latest = conn.execute("SELECT MAX(revision) FROM repository_evidence WHERE evidence_id=?", (ref.evidence_id,)).fetchone()[0]
                dependents = conn.execute("SELECT evidence_id, revision FROM evidence_relations WHERE target_id=? AND target_revision=?", (ref.evidence_id, ref.revision)).fetchall()
            if latest != old.revision or old.freshness == "invalidated":
                continue
            seen.add(ref.evidence_id)
            updated = old.model_copy(update={"revision": old.revision + 1, "freshness": "invalidated", "freshness_reason": reason})
            self._persist(updated, task_id=task_id, role="invalidated")
            changed.append(updated.ref)
            pending.extend(EvidenceRef(evidence_id=r["evidence_id"], revision=r["revision"]) for r in dependents)
        return changed

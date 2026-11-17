"""Additive workspace migration; existing approval payloads remain byte-compatible."""

STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS repository_evidence (
        evidence_id TEXT NOT NULL, revision INTEGER NOT NULL,
        repository_id TEXT NOT NULL, source_view TEXT NOT NULL,
        category TEXT NOT NULL, classification TEXT NOT NULL,
        validation TEXT NOT NULL, freshness TEXT NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY(evidence_id, revision))""",
    """CREATE TABLE IF NOT EXISTS evidence_sources (
        evidence_id TEXT NOT NULL, revision INTEGER NOT NULL, repository_id TEXT NOT NULL,
        path TEXT NOT NULL, symbol TEXT NOT NULL, sha256 TEXT NOT NULL,
        FOREIGN KEY(evidence_id, revision) REFERENCES repository_evidence(evidence_id, revision))""",
    """CREATE TABLE IF NOT EXISTS evidence_relations (
        evidence_id TEXT NOT NULL, revision INTEGER NOT NULL, relation TEXT NOT NULL,
        target_id TEXT NOT NULL, target_revision INTEGER NOT NULL,
        UNIQUE(evidence_id, revision, relation, target_id, target_revision))""",
    """CREATE TABLE IF NOT EXISTS task_evidence_links (
        task_id TEXT NOT NULL, evidence_id TEXT NOT NULL, revision INTEGER NOT NULL,
        role TEXT NOT NULL, UNIQUE(task_id, evidence_id, revision, role))""",
    """CREATE TABLE IF NOT EXISTS evidence_dependencies (
        evidence_id TEXT NOT NULL, revision INTEGER NOT NULL, dependency TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS evidence_dependency ON evidence_dependencies(dependency)",
    "CREATE INDEX IF NOT EXISTS evidence_repository ON repository_evidence(repository_id, category, freshness)",
    "CREATE INDEX IF NOT EXISTS evidence_path ON evidence_sources(repository_id, path, symbol)",
    "CREATE INDEX IF NOT EXISTS evidence_target ON evidence_relations(target_id, target_revision)",
    "CREATE INDEX IF NOT EXISTS evidence_task ON task_evidence_links(task_id)",
    """CREATE TABLE IF NOT EXISTS tool_queue_jobs (
        job_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
        status TEXT NOT NULL, worker_id TEXT NOT NULL, payload TEXT NOT NULL,
        row_version INTEGER NOT NULL DEFAULT 1, kind TEXT NOT NULL DEFAULT 'queue_job')""",
    "CREATE INDEX IF NOT EXISTS tool_queue_task ON tool_queue_jobs(task_id, status)",
)

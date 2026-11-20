"""Persistence and atomic claims for existing tool queues, without a scheduler."""

import json

from mana_agent.multi_agent.core.types import QueueJobStatus
from mana_agent.multi_agent.taskboard.deserializers import queue_job_from_dict, serialize
from mana_agent.repository_evidence.service import EvidenceConflict


class ToolQueueStore:
    def __init__(self, db):
        self.db = db
        self._instances: dict[str, Any] = {}

    def jobs(self):
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload FROM tool_queue_jobs WHERE kind='queue_job'").fetchall()
        result = {}
        for row in rows:
            payload = json.loads(row["payload"])
            if payload.get("kind") == "queue_job":
                item = queue_job_from_dict(payload["job"])
                if item.job_id in self._instances:
                    cached = self._instances[item.job_id]
                    for k, v in item.__dict__.items():
                        setattr(cached, k, v)
                    result[item.job_id] = cached
                else:
                    self._instances[item.job_id] = item
                    result[item.job_id] = item
        return result

    def save(self, job):
        self._instances[job.job_id] = job
        with self.db.transaction() as conn:
            prior = conn.execute("SELECT row_version FROM tool_queue_jobs WHERE job_id=?", (job.job_id,)).fetchone()
            current = int(prior["row_version"]) if prior else 0
            if current != job.row_version:
                raise EvidenceConflict("Tool queue job changed before publication")
            serialized = serialize(job)
            serialized["row_version"] = current + 1
            from mana_agent.utils.redaction import redact_secrets
            from mana_agent.chat.activity import safe_text
            from mana_agent.services.activity_context import current_activity_context
            activity = current_activity_context()
            original = json.dumps({"kind": "queue_job", "job": serialized}, sort_keys=True)
            encoded = safe_text(json.dumps(redact_secrets({"kind": "queue_job", "job": serialized}), sort_keys=True),
                                None, protected=getattr(activity, "protected_text", ()))
            if encoded != original:
                raise PermissionError("Durable tool jobs require credential references instead of inline secrets")
            conn.execute("""INSERT INTO tool_queue_jobs(job_id,task_id,fingerprint,status,worker_id,payload,row_version)
                VALUES (?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
                worker_id=excluded.worker_id,payload=excluded.payload,row_version=excluded.row_version""",
                (job.job_id, job.task_id, job.fingerprint, job.status.value, job.assigned_worker_agent_id or "", encoded, current + 1))
        job.row_version = current + 1

    def admit(self, job):
        self._instances[job.job_id] = job
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT job_id FROM tool_queue_jobs WHERE kind='queue_job' AND task_id=? AND fingerprint=? AND status IN ('pending','queued','claimed','running')",
                (job.task_id, job.fingerprint),
            ).fetchone()
            if row:
                return self.jobs()[row["job_id"]]
            self.save(job)
        return job

    def claim(self, job_id, worker_id):
        with self.db.transaction() as conn:
            job = self.jobs()[job_id]
            if job.assigned_worker_agent_id != worker_id:
                raise PermissionError("Tool queue claim belongs to another worker")
            jobs = self.jobs()
            if any(dep not in jobs or jobs[dep].status != QueueJobStatus.DONE for dep in job.depends_on):
                return None
            if job.status not in {QueueJobStatus.QUEUED, QueueJobStatus.PENDING}:
                return None
            job.status = QueueJobStatus.CLAIMED
            job.agent_id = worker_id
            if worker_id.startswith("subagent_"):
                job.subagent_id = worker_id
            job.payload = {**job.payload, "claimed_by_agent_id": worker_id}
            self.save(job)
            return job

    def begin_operation(self, request):
        key = f"{request.task_id}:{request.attempt_generation}:{request.operation_id}"
        from hashlib import sha256
        from mana_agent.chat.activity import safe_text
        from mana_agent.services.activity_context import current_activity_context
        from mana_agent.utils.redaction import redact_secrets
        activity = current_activity_context()
        encoded = json.dumps(request.model_dump(mode="json"), sort_keys=True)
        digest = sha256(encoded.encode()).hexdigest()
        sanitized = json.loads(safe_text(json.dumps(redact_secrets(request.model_dump(mode="json", exclude={"arguments"}))), None,
                                         protected=getattr(activity, "protected_text", ())))
        payload = {"kind": "operation", "request": sanitized, "request_digest": digest}
        with self.db.transaction() as conn:
            prior = conn.execute("SELECT status,payload FROM tool_queue_jobs WHERE job_id=?", (key,)).fetchone()
            if prior:
                saved = json.loads(prior["payload"])
                if saved.get("request_digest") != payload["request_digest"]:
                    raise EvidenceConflict("Operation identity already has different arguments")
                raise EvidenceConflict("Operation already dispatched; reconcile its durable result before replay")
            conn.execute("INSERT INTO tool_queue_jobs(job_id,task_id,fingerprint,status,worker_id,payload,kind) VALUES (?,?,?,?,?,?,?)",
                         (key, request.task_id, request.tool_call_id, "running", request.agent_id, json.dumps(payload), "operation"))
        return key

    def finish_operation(self, key, result):
        with self.db.transaction() as conn:
            row = conn.execute("SELECT payload FROM tool_queue_jobs WHERE job_id=?", (key,)).fetchone()
            payload = json.loads(row["payload"])
            payload["result"] = result.model_dump(mode="json", exclude={"result", "mutations"})
            payload["result"]["mutations"] = [item.model_dump(exclude={"diff"}) for item in result.mutations]
            conn.execute("UPDATE tool_queue_jobs SET status=?,payload=?,row_version=row_version+1 WHERE job_id=? AND status='running'",
                         (result.status, json.dumps(payload), key))

    def operation_receipts(self, task_id):
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload FROM tool_queue_jobs WHERE task_id=? AND kind='operation' ORDER BY rowid",
                                (task_id,)).fetchall()
        return [receipt for row in rows if (receipt := json.loads(row["payload"]).get("result"))]

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from mana_agent.search.models import SearchMemoryRecord, SearchResult, SearchTarget, utc_iso


def normalize_search_query(value: str) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s./:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def search_fingerprint(query: str, source_type: SearchTarget) -> str:
    return stable_hash({"query": normalize_search_query(query), "source_type": source_type})


def stable_hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class SearchMemoryStore:
    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / ".mana" / "search_memory.jsonl"
        self._records: list[SearchMemoryRecord] | None = None

    def _load(self) -> list[SearchMemoryRecord]:
        if self._records is not None:
            return self._records
        records: list[SearchMemoryRecord] = []
        if self.path.exists():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                    records.append(SearchMemoryRecord(**payload))
                except Exception:
                    continue
        self._records = records
        return records

    def find(
        self,
        *,
        query: str,
        source_type: SearchTarget,
        min_confidence: float = 0.7,
        limit: int = 5,
    ) -> list[SearchMemoryRecord]:
        normalized = normalize_search_query(query)
        now = datetime.now(timezone.utc)
        exact = search_fingerprint(query, source_type)
        out: list[SearchMemoryRecord] = []
        for record in reversed(self._load()):
            if record.source_type != source_type or record.negative:
                continue
            if record.confidence < min_confidence:
                continue
            expires_at = _parse_dt(record.expires_at)
            if expires_at is not None and expires_at < now:
                continue
            same_query = record.query_fingerprint == exact
            fuzzy_query = normalized and (
                normalized in record.normalized_query or record.normalized_query in normalized
            )
            if same_query or fuzzy_query:
                out.append(record)
                if len(out) >= limit:
                    break
        return out

    def has_negative(self, *, query: str, source_type: SearchTarget) -> bool:
        fingerprint = search_fingerprint(query, source_type)
        now = datetime.now(timezone.utc)
        for record in reversed(self._load()):
            if record.source_type != source_type or not record.negative:
                continue
            if record.query_fingerprint != fingerprint:
                continue
            expires_at = _parse_dt(record.expires_at)
            if expires_at is None or expires_at >= now:
                return True
        return False

    def store_results(
        self,
        *,
        original_query: str,
        source_type: SearchTarget,
        results: Iterable[SearchResult],
        task_id: str | None = None,
        ttl_days: int = 14,
    ) -> list[SearchMemoryRecord]:
        normalized = normalize_search_query(original_query)
        fingerprint = search_fingerprint(original_query, source_type)
        expires_at = (datetime.now(timezone.utc) + timedelta(days=max(1, int(ttl_days)))).isoformat()
        rows: list[SearchMemoryRecord] = []
        for result in results:
            summary = " ".join((result.summary or result.snippet or "").split())
            if len(summary.split()) < 4 or result.confidence < 0.45:
                continue
            record = SearchMemoryRecord(
                id=stable_hash({"query": fingerprint, "url": result.canonical_url(), "at": result.fetched_at}),
                task_id=task_id,
                query_fingerprint=fingerprint,
                original_query=original_query,
                normalized_query=normalized,
                source_type=source_type,
                title=result.title,
                url=result.canonical_url(),
                repo=result.repo,
                source_domain=result.source_domain,
                summary=summary[:1000],
                key_findings=list(result.key_findings[:5]),
                confidence=float(result.confidence),
                fetched_at=result.fetched_at or utc_iso(),
                expires_at=expires_at,
                tags=[],
            )
            rows.append(record)
        if not rows:
            self.store_negative(original_query=original_query, source_type=source_type, ttl_days=min(ttl_days, 3))
            return []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for record in rows:
                fh.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")
        if self._records is not None:
            self._records.extend(rows)
        return rows

    def store_negative(self, *, original_query: str, source_type: SearchTarget, ttl_days: int = 3) -> SearchMemoryRecord:
        normalized = normalize_search_query(original_query)
        fingerprint = search_fingerprint(original_query, source_type)
        record = SearchMemoryRecord(
            id=stable_hash({"query": fingerprint, "negative": True, "at": utc_iso()}),
            task_id=None,
            query_fingerprint=fingerprint,
            original_query=original_query,
            normalized_query=normalized,
            source_type=source_type,
            title="negative search marker",
            url="",
            repo=None,
            source_domain=None,
            summary="No useful high-confidence search result was found.",
            key_findings=[],
            confidence=0.0,
            fetched_at=utc_iso(),
            expires_at=(datetime.now(timezone.utc) + timedelta(days=max(1, int(ttl_days)))).isoformat(),
            negative=True,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")
        if self._records is not None:
            self._records.append(record)
        return record

# Supermemory v5 migration

## Trigger and root cause

The external Supermemory backend pinned SDK 3.x and used legacy document and
search methods. Upgrading the dependency alone breaks those calls because v5
requires an explicit namespace in each content path and uses different request
fields and response models. The old recall fixture accepted either SDK method
and returned a flat result, so it could miss both changes. The old delete path
also passed a document ID to the memory-forgetting operation before deleting the
document itself.

## Adapter contract

- Derive the namespace from the full `MemoryScope` using the same string as
  existing hosted data. Preserve the caller ID calculation so an identical add
  resolves to the same document. Existing credentials and hosted records can be
  reused in place.
- Use hybrid recall at threshold `0.6`, with a typed filter for scope and request
  metadata. Search hits carry memory or chunk IDs. Document reads, updates, and
  deletes use document IDs or the caller ID supplied on ingest. The optional
  search source document provides its document ID.
- A PATCH acknowledgement supplies a document ID and processing status. A
  subsequent read can still return the previous version until processing
  finishes. A metadata-only PATCH omits content. The stable caller ID survives
  a PATCH, so re-adding the original payload with that ID can append old text to
  the updated document. Use PATCH to replace canonical content.
- Delete removes source documents and their derived knowledge. Clear lists
  matching document IDs in one scope, then deletes in batches of up to 100.
  Supermemory can return HTTP 200 with per-document failures, including documents
  still processing. Successful deletions persist when another ID fails. Retry
  from a fresh scoped listing after processing finishes. Writes concurrent with
  pagination can affect which documents the listing observes.
- The connection check requests one namespace. It checks authentication
  and availability even for an empty account. Write access and same-scope recall
  are verified separately during rollout.

**Do not reintroduce** legacy request fields, unscoped document operations, or
success based solely on the HTTP status of a deletion. Keep the namespace and
document identity compatible with data written before v5.

## User-owned verification

```bash
python -m pip install -e '.[dev,supermemory]'
python -m pytest -q tests/test_memory_architecture.py tests/gateway/test_followup_and_memory_integration.py -k supermemory
```

The SDK transport tests use a local HTTP mock and a dummy key. Before rollout,
use the configured account to read a known old document under its existing
namespace and recall it with a known query. In a disposable scope, verify add,
search, metadata-only and content updates, delete, and clear. Poll document
processing and memory generation before judging recall. Verify that an ID from
the disposable scope cannot be accessed from a different scope. Roll out the
adapter and SDK together after these checks. Live account verification is
pending.

[Supermemory API migration guide](https://supermemory.ai/docs/migration/api-v5) ·
[Python SDK migration guide](https://github.com/supermemoryai/python-sdk/blob/main/MIGRATION.md)

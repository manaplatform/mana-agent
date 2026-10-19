# OpenAI Batch API Integration

## Overview

Mana-Agent integrates the official **OpenAI Batch API** (`/v1/batches` and `/v1/files` with `purpose=batch`) into its model execution runtime. This capability offloads eligible bulk, asynchronous model workloads—such as repository-wide audits, multi-file code reviews, documentation generation, and evaluation suites—to OpenAI's batch infrastructure, achieving a **50% discount on token inference costs** while improving throughput.

Interactive conversational turns, real-time tool loops, and streaming requests remain on the low-latency Realtime API.

## Architecture

```text
                     Mana-Agent Model Request Group
                                   │
                                   ▼
                        BatchExecutionRouter
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
        Realtime API                           OpenAI Batch API
   (Interactive / Streaming)             (Bulk Asynchronous Workloads)
                │                                     │
                │                                     ▼
                │                             JSONL Serialization
                │                                     │
                │                                     ▼
                │                            Upload: POST /v1/files
                │                                     │
                │                                     ▼
                │                           Create: POST /v1/batches
                │                                     │
                │                                     ▼
                │                           Durable SQLite State Store
                │                                (WAL Persistence)
                │                                     │
                │                                     ▼
                │                              Status Polling &
                │                            Restart Recovery Loop
                │                                     │
                │                                     ▼
                │                           Download: GET /v1/files/...
                │                                     │
                │                                     ▼
                │                           Revision Fencing Check
                │                       (Git commit SHA & File SHA-256)
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                            Autonomous Loop
                        (TaskBoard & Supervisor)
```

## Routing Rules

The `BatchExecutionRouter` automatically selects between Realtime execution and OpenAI Batch API using deterministic criteria:

1. **Provider**: Target model provider must be `openai` or `azure_openai`.
2. **Interactivity**: User-facing interactive turns requiring immediate response route to Realtime.
3. **Streaming**: Tasks requiring real-time token streaming route to Realtime.
4. **Dependencies**: Requests with intra-batch dependencies that require immediate intermediate results route to Realtime.
5. **Threshold**: Workloads contain at least `openai_batch_min_requests` (default: 1) independent items targeting `/v1/responses` or `/v1/chat/completions`.
6. **Model Eligibility & Auto-Enabling**: When `openai_batch_always_enable_if_supported` is enabled (default: `true`), requests targeting supported model families (`gpt-4o`, `gpt-4o-mini`, `o1`, `o3`, `gpt-4`, `gpt-3.5`, `text-embedding`) automatically route to OpenAI Batch API for a 50% cost discount. Unsupported endpoints (realtime audio, Whisper, TTS, DALL-E) route to Realtime API.
7. **Cost Optimization**: Eligible batches receive a 50% discount factor applied to input and output token accounting.

## Durable Lifecycle State Machine

All batch operations are tracked in a durable SQLite WAL-mode store (`BatchJobStore`), surviving process crashes and restarts:

| State | Description |
|---|---|
| `CREATED` | Batch record initialized and stored with idempotency key. |
| `UPLOADING` | JSONL request payload uploaded to OpenAI Files API (`purpose=batch`). |
| `SUBMITTED` | Batch job registered with OpenAI Batch API (`POST /v1/batches`). |
| `VALIDATING` | OpenAI validating batch input structure and schema. |
| `IN_PROGRESS` | OpenAI actively executing model requests in batch pool. |
| `FINALIZING` | OpenAI generating output and error files. |
| `COMPLETED` | OpenAI execution finished successfully. |
| `RESULTS_IMPORTED` | Output file downloaded, parsed, correlated by `custom_id`, and stored. |
| `VERIFIED` | Results verified through revision fencing and applied. |
| `FAILED` | Batch validation or execution failed on OpenAI. |
| `EXPIRED` | 24-hour completion window expired. |
| `CANCELLING` / `CANCELLED` | Batch cancelled via `client.cancel_batch(batch_id)`. |

### Idempotency Deduplication

Every batch submission accepts an `idempotency_key`. If an identical key is submitted while a prior job is non-terminal, the existing `BatchJobRecord` is returned immediately without duplicate file uploads or redundant API calls.

### Startup Crash Recovery

When `openai_batch_auto_resume` is enabled (default: `true`), `AutonomousExecutionController` and `BatchLifecycleManager` inspect the store on startup, query OpenAI for pending batch jobs, update their status, and automatically import completed results.

## Revision Fencing

Because batch workloads execute asynchronously over minutes or hours, target repository files or Git branches may drift concurrently:

- Before submitting a batch request, `InternalCodingEngine` records the target file's SHA-256 hash (`expected_sha256`) and the current repository Git commit SHA (`target_revision`).
- Upon downloading results, `verify_revision_fencing` recalculates the file's current SHA-256 and inspects HEAD.
- If concurrent edits modified the target file while the batch was queued, the stale result is safely rejected with `stale_revision_rejected` rather than overwriting fresh work.

## Configuration

Configure the Batch API via environment variables or `~/.mana/config.toml`:

```toml
[openai.batch]
enabled = true
mode = "auto"                    # "auto", "batch", or "realtime"
endpoint = "/v1/responses"       # "/v1/responses" or "/v1/chat/completions"
completion_window = "24h"
min_requests = 1
always_enable_if_supported = true
max_requests_per_batch = 50000
max_file_size_mb = 200
auto_resume = true
retry_failed_requests = true
cost_optimization = true
```

Corresponding environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_BATCH_ENABLED` | `true` | Enable or disable OpenAI Batch API integration. |
| `OPENAI_BATCH_MODE` | `auto` | Routing mode (`auto`, `batch`, `realtime`). |
| `OPENAI_BATCH_ENDPOINT` | `/v1/responses` | Target model endpoint. |
| `OPENAI_BATCH_COMPLETION_WINDOW` | `24h` | Execution SLA window. |
| `OPENAI_BATCH_MIN_REQUESTS` | `1` | Minimum independent requests to trigger batch. |
| `OPENAI_BATCH_ALWAYS_ENABLE_IF_SUPPORTED` | `true` | Always route eligible model requests to Batch API for 50% discount. |
| `OPENAI_BATCH_MAX_REQUESTS_PER_BATCH` | `50000` | Maximum requests per JSONL payload. |
| `OPENAI_BATCH_MAX_FILE_SIZE_MB` | `200` | Maximum JSONL file size in megabytes. |
| `OPENAI_BATCH_AUTO_RESUME` | `true` | Resume pending batches upon startup. |
| `OPENAI_BATCH_RETRY_FAILED_REQUESTS` | `true` | Re-submit failed individual requests. |
| `OPENAI_BATCH_COST_OPTIMIZATION` | `true` | Apply 50% token cost accounting discount. |

## Internal Coding Engine Integration

The Internal Coding Engine exposes native batch workflows:

```python
# Submit multi-file code audit
batch_job = engine.submit_code_analysis_batch(
    task=task,
    workspace=workspace,
    files=["src/agent.py", "src/models.py", "src/scheduler.py"],
    prompt="Audit code for unhandled exceptions or resource leaks.",
)

# Retrieve results with revision fencing
applied_results = engine.retrieve_and_apply_batch_results(
    batch_id=batch_job.batch_id,
    workspace=workspace,
)
```

## TUI Activity Box Streaming

In the terminal user interface, batch jobs are streamed live inside `CodingActivityBox` and `MainActivityBox`:
- Displayed with dedicated `📦` batch icons.
- Submissions highlight active batch IDs and total request counts.
- Token and monetary cost savings reflect the official 50% discount badge (`[50% off]`).
- Progress meters display completed versus total requests (`batch: 12/20 reqs`).

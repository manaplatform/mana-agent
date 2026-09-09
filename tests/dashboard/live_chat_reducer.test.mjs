import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { createState, reduce, snapshot } = require(
  "../../src/mana_agent/dashboard/components/live_chat.js",
);

const event = (sequence, type, extra = {}) => ({
  sequence,
  event_id: extra.event_id || `event-${sequence}`,
  type,
  status: extra.status || "running",
  started_at: `2026-07-23T00:00:${String(sequence).padStart(2, "0")}Z`,
  execution_id: extra.execution_id || "run-1",
  ...extra,
});

test("optimistic message reconciles with its canonical server message", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "optimistic",
    message: { message_id: "client-1", content: "hello", created_at: "2026-07-23T00:00:00Z" },
  });
  assert.equal(snapshot(state).messages[0].status, "sending");
  reduce(state, {
    type: "event",
    event: event(1, "message.accepted", {
      metadata: { message_id: "client-1", client_message_id: "client-1", content: "hello" },
    }),
  });
  const messages = snapshot(state).messages;
  assert.equal(messages.length, 1);
  assert.equal(messages[0].message_id, "client-1");
  assert.equal(messages[0].optimistic, false);
});

test("tool lifecycle updates one card and concurrent tools remain independent", () => {
  const state = createState("session-1");
  reduce(state, { type: "event", event: event(1, "tool.started", { event_id: "tool-a", metadata: { tool_call_id: "tool-a", tool_name: "search", args_summary: "query" } }) });
  reduce(state, { type: "event", event: event(2, "tool.started", { event_id: "tool-b", metadata: { tool_call_id: "tool-b", tool_name: "read" } }) });
  reduce(state, { type: "event", event: event(3, "tool.progress", { event_id: "tool-a", summary: "half", metadata: { tool_call_id: "tool-a" } }) });
  reduce(state, { type: "event", event: event(4, "tool.stdout", { event_id: "tool-a", summary: "live log", metadata: { tool_call_id: "tool-a" } }) });
  reduce(state, { type: "event", event: event(5, "tool.finished", { event_id: "tool-a", status: "success", duration_ms: 120, summary: "done", metadata: { tool_call_id: "tool-a", result_summary: "2 matches" } }) });
  const tools = snapshot(state).tools;
  assert.equal(tools.length, 2);
  const search = tools.find((tool) => tool.id === "tool-a");
  const read = tools.find((tool) => tool.id === "tool-b");
  assert.equal(search.status, "success");
  assert.deepEqual(search.progress, ["half"]);
  assert.deepEqual(search.logs, ["live log"]);
  assert.equal(search.result, "2 matches");
  assert.equal(read.status, "running");
});

test("assistant deltas are ordered into one message and final content reconciles", () => {
  const state = createState("session-1");
  reduce(state, { type: "event", event: event(1, "assistant.started") });
  reduce(state, { type: "event", event: event(3, "assistant.delta", { summary: "world" }) });
  reduce(state, { type: "event", event: event(2, "assistant.delta", { summary: "hello " }) });
  assert.equal(snapshot(state).messages.length, 1);
  assert.equal(snapshot(state).messages[0].content, "hello world");
  reduce(state, { type: "event", event: event(4, "turn.finished", { status: "success", metadata: { message_id: "server-assistant", content: "hello world!" } }) });
  const messages = snapshot(state).messages;
  assert.equal(messages.length, 1);
  assert.equal(messages[0].message_id, "server-assistant");
  assert.equal(messages[0].content, "hello world!");
});

test("events apply immediately, duplicates are idempotent, and replay does not duplicate", () => {
  const state = createState("session-1");
  reduce(state, { type: "optimistic", message: { message_id: "client-1", content: "go" } });
  const log = event(1, "log.info", { summary: "started" });
  reduce(state, { type: "event", event: log });
  assert.equal(snapshot(state).activities[0].summary, "started");
  reduce(state, { type: "event", event: log });
  reduce(state, { type: "hydrate", events: [log] });
  assert.equal(snapshot(state).activities.length, 1);
  assert.equal(snapshot(state).lastSequence, 1);
});

test("context and cost events update one replay-safe live meter", () => {
  const state = createState("session-1");
  reduce(state, { type: "event", event: event(1, "context.budget", {
    metadata: { used_tokens: 4000, context_window: 10000, utilization_ratio: 0.4, breakdown: { schema_tokens: 600 }, cumulative_cost: 0.02, estimated: true },
  }) });
  reduce(state, { type: "event", event: event(2, "cost.updated", {
    metadata: { cumulative_cost: 0.03, remaining_cost: 0.97, estimated: false },
  }) });
  assert.deepEqual(snapshot(state).contextBudget, {
    used_tokens: 4000,
    context_window: 10000,
    utilization_ratio: 0.4,
    breakdown: { schema_tokens: 600 },
    cumulative_cost: 0.03,
    estimated: false,
    event_type: "cost.updated",
    status: "running",
    remaining_cost: 0.97,
  });
});

test("computer permission requests remain actionable until a local decision event", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "event",
    event: event(1, "computer.waiting_permission", {
      metadata: {
        permission_request_id: "permission-1",
        permission_scope: "computer.screenshot.capture",
        preview: "Capture the full screen.",
      },
    }),
  });
  assert.deepEqual(snapshot(state).permissionRequests, [{
    requestId: "permission-1",
    scope: "computer.screenshot.capture",
    preview: "Capture the full screen.",
    kind: "computer",
    status: "pending",
    decision: "",
  }]);
  reduce(state, {
    type: "event",
    event: event(2, "computer.permission_decided", {
      status: "success",
      metadata: {
        permission_request_id: "permission-1",
        decision: "allow_once",
      },
    }),
  });
  assert.equal(snapshot(state).permissionRequests[0].status, "decided");
  assert.equal(snapshot(state).permissionRequests[0].decision, "allow_once");
});

test("server approval requests use the shared actionable permission state", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "event",
    event: event(1, "server.waiting_approval", {
      metadata: {
        permission_request_id: "server-approval-1",
        permission_scope: "server.action.execute",
        preview: "sudo apt-get install -y -- nginx",
        server_approval: true,
      },
    }),
  });
  assert.deepEqual(snapshot(state).permissionRequests, [{
    requestId: "server-approval-1",
    scope: "server.action.execute",
    preview: "sudo apt-get install -y -- nginx",
    kind: "server",
    status: "pending",
    decision: "",
  }]);
  reduce(state, {
    type: "event",
    event: event(2, "server.approval_decided", {
      status: "success",
      metadata: {
        permission_request_id: "server-approval-1",
        decision: "approve",
      },
    }),
  });
  assert.equal(snapshot(state).permissionRequests[0].status, "decided");
  assert.equal(snapshot(state).permissionRequests[0].decision, "approve");
});

test("API approval requests use the exact mutation approval endpoint state", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "event",
    event: event(1, "api.waiting_approval", {
      metadata: {
        permission_request_id: "api-approval-1",
        permission_scope: "api.request.execute",
        preview: "PATCH https://api.example.test/contacts/123",
        api_approval: true,
      },
    }),
  });
  assert.deepEqual(snapshot(state).permissionRequests, [{
    requestId: "api-approval-1",
    scope: "api.request.execute",
    preview: "PATCH https://api.example.test/contacts/123",
    kind: "api",
    status: "pending",
    decision: "",
  }]);
  reduce(state, {
    type: "event",
    event: event(2, "api.approval_decided", {
      status: "success",
      metadata: {
        permission_request_id: "api-approval-1",
        decision: "approve",
      },
    }),
  });
  assert.equal(snapshot(state).permissionRequests[0].status, "decided");
  assert.equal(snapshot(state).permissionRequests[0].decision, "approve");
});

test("transactional approval uses the durable inbox item as authority", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "event",
    event: event(1, "action.approval.required", {
      metadata: {
        permission_request_id: "inbox-1",
        inbox_item_id: "inbox-1",
        action_id: "act-1",
        permission_scope: "transactional_action.once",
        transactional_action_approval: true,
        preview: "Record one display.",
      },
    }),
  });
  assert.deepEqual(snapshot(state).permissionRequests, [{
    requestId: "inbox-1",
    actionId: "act-1",
    scope: "transactional_action.once",
    preview: "Record one display.",
    kind: "transactional",
    status: "pending",
    decision: "",
  }]);
});

test("out-of-order terminal events and failed submissions remain visible", () => {
  const state = createState("session-1");
  reduce(state, { type: "event", event: event(5, "tool.finished", { event_id: "tool-a", status: "success", metadata: { tool_call_id: "tool-a", tool_name: "search" } }) });
  reduce(state, { type: "event", event: event(4, "tool.started", { event_id: "tool-a", metadata: { tool_call_id: "tool-a", tool_name: "search" } }) });
  assert.equal(snapshot(state).tools.length, 1);
  assert.equal(snapshot(state).tools[0].status, "success");
  reduce(state, { type: "optimistic", message: { message_id: "client-fail", content: "keep me" } });
  reduce(state, { type: "submit_failed", messageId: "client-fail", error: "offline" });
  const failed = snapshot(state).messages.find((message) => message.message_id === "client-fail");
  assert.equal(failed.content, "keep me");
  assert.equal(failed.status, "failed");
  assert.equal(failed.error, "offline");
});

test("run errors terminate the streaming assistant and preserve the detailed event error", () => {
  const state = createState("session-1");
  reduce(state, { type: "optimistic", message: { message_id: "client-1", content: "go" } });
  reduce(state, { type: "event", event: event(1, "message.accepted", { metadata: { message_id: "client-1", content: "go" } }) });
  reduce(state, { type: "event", event: event(2, "assistant.started") });
  reduce(state, { type: "event", event: event(3, "error", { status: "failed", summary: "precise failure" }) });
  reduce(state, { type: "submit_failed", messageId: "client-1", error: "HTTP 500" });
  const messages = snapshot(state).messages;
  assert.equal(messages.find((message) => message.role === "assistant").status, "failed");
  assert.equal(messages.find((message) => message.role === "user").error, "precise failure");
});

test("persisted history reconstructs tools, logs, cancellation, and a fresh session is isolated", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "hydrate",
    messages: [{ message_id: "m1", role: "user", content: "task", created_at: "2026-07-23T00:00:00Z" }],
    events: [
      event(1, "tool.started", { event_id: "tool-a", metadata: { tool_call_id: "tool-a", tool_name: "verify" } }),
      event(2, "tool.stdout", { event_id: "tool-a", summary: "pytest", metadata: { tool_call_id: "tool-a" } }),
      event(3, "tool.cancelled", { event_id: "tool-a", status: "cancelled", metadata: { tool_call_id: "tool-a" } }),
      event(4, "turn.cancelled", { status: "cancelled", summary: "cancelled" }),
    ],
  });
  assert.equal(snapshot(state).tools[0].logs[0], "pytest");
  assert.equal(snapshot(state).tools[0].status, "cancelled");
  assert.equal(snapshot(state).runStatus, "cancelled");
  assert.equal(snapshot(createState("session-2")).messages.length, 0);
});

test("execution trace progresses chronologically and updates in-place without duplicate steps", () => {
  const state = createState("session-1");
  reduce(state, {
    type: "optimistic",
    message: { message_id: "client-1", content: "implement feature" },
  });

  // 1. Routing phase
  reduce(state, {
    type: "event",
    event: event(1, "routing_started", { execution_id: "client-1", status: "running" }),
  });
  let traces = snapshot(state).executionTraces;
  assert.equal(traces.length, 1);
  assert.equal(traces[0].steps.length, 1);
  assert.equal(traces[0].steps[0].phase, "routing");
  assert.equal(traces[0].steps[0].status, "running");

  reduce(state, {
    type: "event",
    event: event(2, "routing_completed", { execution_id: "client-1", status: "success", duration_ms: 40 }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].steps.length, 1); // Updated in-place, no duplicate
  assert.equal(traces[0].steps[0].status, "completed");
  assert.equal(traces[0].steps[0].durationMs, 40);

  // 2. Context phase
  reduce(state, {
    type: "event",
    event: event(3, "context_preparation_started", { execution_id: "client-1", status: "running" }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].steps.length, 2);
  assert.equal(traces[0].steps[1].phase, "context");

  // 3. Coding phase with command sub-events
  reduce(state, {
    type: "event",
    event: event(4, "coding_started", {
      execution_id: "client-1",
      status: "running",
      metadata: { backend: "codex" },
    }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].steps.length, 3);
  assert.equal(traces[0].steps[2].title, "Codex");

  reduce(state, {
    type: "event",
    event: event(5, "command.started", {
      event_id: "cmd-1",
      execution_id: "client-1",
      status: "running",
      metadata: { command: "git diff" },
    }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].steps.length, 3); // Sub-event attached to coding step
  assert.equal(traces[0].steps[2].subEvents.length, 2);

  // 4. Tool phase
  reduce(state, {
    type: "event",
    event: event(6, "tool.started", {
      event_id: "tool-1",
      execution_id: "client-1",
      metadata: { tool_call_id: "tool-1", tool_name: "read_file" },
    }),
  });
  reduce(state, {
    type: "event",
    event: event(7, "tool.finished", {
      event_id: "tool-1",
      execution_id: "client-1",
      status: "success",
      duration_ms: 25,
      metadata: { tool_call_id: "tool-1", tool_name: "read_file", result_summary: "contents" },
    }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].steps.length, 4);
  assert.equal(traces[0].steps[3].phase, "tool");
  assert.equal(traces[0].steps[3].status, "completed");

  // 5. Model phase and 6. Completion
  reduce(state, {
    type: "event",
    event: event(8, "assistant.started", { execution_id: "client-1", status: "running" }),
  });
  reduce(state, {
    type: "event",
    event: event(9, "turn.finished", { execution_id: "client-1", status: "success" }),
  });
  traces = snapshot(state).executionTraces;
  assert.equal(traces[0].isCompleted, true);
  assert.equal(traces[0].activeStepId, null);
  assert.equal(traces[0].steps.length, 6);
  assert.equal(traces[0].steps[4].status, "completed"); // Model step finalized
  assert.equal(traces[0].steps[5].phase, "completion");
});

test("execution trace cleans up running steps on error or cancellation", () => {
  const state = createState("session-fail");
  reduce(state, {
    type: "optimistic",
    message: { message_id: "client-f", content: "fail test" },
  });
  reduce(state, {
    type: "event",
    event: event(1, "routing_started", { execution_id: "client-f", status: "running" }),
  });
  assert.equal(snapshot(state).executionTraces[0].steps[0].status, "running");

  reduce(state, {
    type: "event",
    event: event(2, "error", { execution_id: "client-f", status: "failed", summary: "gateway unreachable" }),
  });
  const trace = snapshot(state).executionTraces[0];
  assert.equal(trace.isFailed, true);
  assert.equal(trace.activeStepId, null);
  assert.equal(trace.steps[0].status, "failed");
  assert.equal(trace.steps[0].detail, "gateway unreachable");
});

test("execution trace derives generic fallback cleanly for future runtime phases", () => {
  const state = createState("session-fallback");
  reduce(state, {
    type: "event",
    event: event(1, "audit_verification.started", { status: "running" }),
  });
  const trace = snapshot(state).executionTraces[0];
  assert.equal(trace.steps.length, 1);
  assert.equal(trace.steps[0].phase, "audit");
  assert.equal(trace.steps[0].title, "Audit");
});


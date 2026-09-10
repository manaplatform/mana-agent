(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.ManaLiveChat = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const terminal = new Set(["success", "failed", "cancelled", "timed_out"]);
  const text = (value) => String(value == null ? "" : value);
  const metaOf = (event) => event.metadata || event.details || event.payload || {};
  const eventType = (event) => text(event.type || event.event_type);
  const eventStatus = (event) => text(event.status || "running");
  const eventSequence = (event) => Number(event.sequence || 0);
  const eventTime = (event) => text(event.started_at || event.timestamp || event.created_at);

  const formatDuration = (ms) => {
    if (ms == null || isNaN(ms)) return "";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  function classifyRuntimePhase(eventType, metadata) {
    const et = text(eventType || "").trim().toLowerCase();
    const meta = metadata || {};

    const SEARCH_TOOLS = new Set([
      "web_search",
      "github_search",
      "repo_search",
      "code_search",
      "find_by_name",
      "grep_search",
      "search_files",
      "search",
      "document_query",
    ]);

    const toolName = text(meta.tool_name || meta.name || "").trim();

    // Routing
    if (
      et === "routing_started" ||
      et === "routing_envelope_created" ||
      et === "agent.routing" ||
      et === "agent.decision" ||
      et === "entry_route_decided" ||
      et === "routing_completed" ||
      et === "routing_failed" ||
      et === "gateway.entry_route" ||
      et === "followup_classified" ||
      et === "route_selected"
    ) {
      return ["routing", "Routing"];
    }

    // Context Preparation
    if (
      et === "context_preparation_started" ||
      et === "context_retrieval_started" ||
      et === "context_retrieval_completed" ||
      et === "context_retrieval" ||
      et === "context.budget" ||
      et === "context.compacted" ||
      et === "context.capabilities_loaded" ||
      et === "context.capabilities_unloaded" ||
      et === "workspace.repository_initialized" ||
      et.startsWith("context.") ||
      et.startsWith("budget.") ||
      et.startsWith("cost.")
    ) {
      return ["context", "Context preparation"];
    }

    // Search
    if (
      et === "search_started" ||
      et === "search_completed" ||
      et === "search_failed" ||
      et.startsWith("search.") ||
      SEARCH_TOOLS.has(toolName) ||
      (toolName && toolName.toLowerCase().includes("search")) ||
      meta.route === "search" ||
      meta.route === "repository" ||
      meta.route === "github"
    ) {
      return ["search", "Searching"];
    }

    // Coding / Codex Backend
    if (
      et === "coding_started" ||
      et === "coding.terminal" ||
      et === "coding.progress" ||
      et === "coding" ||
      et.startsWith("command.") ||
      et.startsWith("patch.") ||
      et.startsWith("file.") ||
      (meta.backend === "codex" &&
        !et.startsWith("turn.") &&
        !et.startsWith("tool.") &&
        !et.startsWith("search.") &&
        !et.startsWith("routing."))
    ) {
      const backend = text(meta.backend || "coding");
      const title = backend === "codex" ? "Codex" : "Coding";
      return ["coding", title];
    }

    // Tool Execution (non-search)
    if (
      et.startsWith("tool.") ||
      et === "tool_started" ||
      et === "tool_finished" ||
      et === "tool_failed" ||
      et === "tool_cancelled"
    ) {
      const title = toolName ? `Tool: ${toolName}` : "Tool execution";
      return ["tool", title];
    }

    // Model Execution
    if (
      et === "model_execution_started" ||
      et === "model.started" ||
      et === "model.completed" ||
      et === "assistant.started" ||
      et === "assistant.delta" ||
      et === "thinking_started"
    ) {
      return ["model", "Model execution"];
    }

    // Completion
    if (
      et === "turn.completed" ||
      et === "turn.finished" ||
      et === "conversation_response_created" ||
      et === "assistant.completed"
    ) {
      return ["completion", "Completed"];
    }

    // Failure / Cancellation
    if (
      et === "error" ||
      et === "turn.cancelled" ||
      et === "turn.timeout" ||
      et === "tool.timeout" ||
      et === "cancelled"
    ) {
      return ["failure", et.includes("cancelled") ? "Cancelled" : "Failed"];
    }

    // Generic Fallback
    const parts = et.replaceAll("_", ".").split(".");
    const phase = parts[0] || "activity";
    const title = phase.charAt(0).toUpperCase() + phase.slice(1);
    return [phase, title];
  }

  function createTrace(turnId) {
    return {
      turnId: text(turnId),
      aliases: new Set([text(turnId)]),
      steps: [],
      stepById: new Map(),
      activeStepId: null,
      isCompleted: false,
      isFailed: false,
      isCancelled: false,
    };
  }

  function cleanTrace(trace) {
    return {
      turnId: trace.turnId,
      activeStepId: trace.activeStepId,
      isCompleted: trace.isCompleted,
      isFailed: trace.isFailed,
      isCancelled: trace.isCancelled,
      steps: trace.steps.map((step) => ({
        stepId: step.stepId,
        turnId: step.turnId,
        phase: step.phase,
        title: step.title,
        status: step.status,
        detail: step.detail,
        startedAt: step.startedAt,
        endedAt: step.endedAt,
        durationMs: step.durationMs,
        subEvents: [...step.subEvents],
        metadata: { ...step.metadata },
      })),
    };
  }

  function resolveTrace(state, turnId) {
    const id = text(turnId);
    if (!id) return null;
    if (state.executionTraces.has(id)) return state.executionTraces.get(id);
    for (const trace of state.executionTraces.values()) {
      if (trace.turnId === id || trace.aliases.has(id)) {
        return trace;
      }
    }
    for (const msg of state.messages.values()) {
      if (msg.role === "user") {
        const mId = text(msg.message_id || msg.id);
        const execId = text(msg.execution_id || msg.run_id);
        if ((mId === id || execId === id) && (state.executionTraces.has(mId) || state.executionTraces.has(execId))) {
          const existing = state.executionTraces.get(mId) || state.executionTraces.get(execId);
          existing.aliases.add(id);
          state.executionTraces.set(id, existing);
          return existing;
        }
      }
    }
    return null;
  }

  function getTurnId(state, event) {
    const meta = metaOf(event);
    const direct = text(event.execution_id || event.turn_id || meta.execution_id || meta.turn_id || meta.client_message_id);
    if (direct) return direct;
    const userMessages = [...state.messages.values()].filter((m) => m.role === "user");
    if (userMessages.length > 0) {
      const latest = userMessages[userMessages.length - 1];
      return text(latest.execution_id || latest.message_id || latest.id);
    }
    return "turn_default";
  }

  function applyTraceEvent(state, event) {
    if (!event || typeof event !== "object") return;
    const type = eventType(event);
    if (!type) return;
    const meta = metaOf(event);
    const status = eventStatus(event);
    const rawStatus = status.toLowerCase();
    const title = text(event.title);
    const detail = eventSummary(event) || text(event.error);
    const durationMs = event.duration_ms != null ? Number(event.duration_ms) : null;
    const eventId = text(event.event_id || event.id);

    const [phase, defaultTitle] = classifyRuntimePhase(type, meta);
    const stepTitle = title || defaultTitle;

    const turnId = getTurnId(state, event);
    let trace = resolveTrace(state, turnId);
    if (!trace) {
      trace = createTrace(turnId);
      state.executionTraces.set(turnId, trace);
    }

    if (event.execution_id) trace.aliases.add(text(event.execution_id));
    if (event.turn_id) trace.aliases.add(text(event.turn_id));
    if (meta.client_message_id) trace.aliases.add(text(meta.client_message_id));
    if (meta.message_id) trace.aliases.add(text(meta.message_id));

    let stepId;
    if (phase === "tool") {
      const toolCallId = text(event.tool_call_id || meta.tool_call_id || meta.call_id || eventId);
      stepId = toolCallId ? `${trace.turnId}:tool:${toolCallId}` : `${trace.turnId}:tool:${type}`;
    } else {
      stepId = `${trace.turnId}:${phase}`;
    }

    const isTerminalStep = terminal.has(rawStatus) || rawStatus === "completed" || rawStatus === "done";

    if (phase === "completion" || phase === "failure") {
      if (phase === "completion") {
        trace.isCompleted = true;
      } else if (rawStatus === "cancelled" || rawStatus === "interrupted" || type.includes("cancelled")) {
        trace.isCancelled = true;
      } else {
        trace.isFailed = true;
      }

      for (const step of trace.steps) {
        if (step.status === "running") {
          step.status = trace.isCompleted ? "completed" : (trace.isCancelled ? "cancelled" : "failed");
          if (detail && !step.detail) {
            step.detail = detail;
          }
          step.endedAt = eventTime(event) || new Date().toISOString();
          if (step.durationMs == null && step.startedAt) {
            step.durationMs = Math.max(0, Date.parse(step.endedAt) - Date.parse(step.startedAt));
          }
        }
      }
      trace.activeStepId = null;
    }

    let existing = trace.stepById.get(stepId);

    if (existing) {
      if (isTerminalStep) {
        existing.status = rawStatus === "success" || rawStatus === "done" ? "completed" : rawStatus;
        if (detail && !existing.detail) existing.detail = detail;
        existing.endedAt = eventTime(event) || new Date().toISOString();
        if (durationMs != null) {
          existing.durationMs = durationMs;
        } else if (existing.durationMs == null && existing.startedAt) {
          existing.durationMs = Math.max(0, Date.parse(existing.endedAt) - Date.parse(existing.startedAt));
        }
        if (trace.activeStepId === stepId) {
          trace.activeStepId = null;
        }
      } else {
        if (detail) existing.detail = detail;
        if (stepTitle && existing.title === defaultTitle) existing.title = stepTitle;
      }

      if (eventId) {
        const alreadyHas = existing.subEvents.some((se) => text(se.event_id || se.id) === eventId);
        if (!alreadyHas) existing.subEvents.push({ ...event });
      } else {
        existing.subEvents.push({ ...event });
      }
      return existing;
    }

    if (trace.activeStepId && trace.stepById.has(trace.activeStepId)) {
      const activeStep = trace.stepById.get(trace.activeStepId);
      if (activeStep.phase !== phase && activeStep.status === "running") {
        activeStep.status = "completed";
        activeStep.endedAt = eventTime(event) || new Date().toISOString();
        if (activeStep.durationMs == null && activeStep.startedAt) {
          activeStep.durationMs = Math.max(0, Date.parse(activeStep.endedAt) - Date.parse(activeStep.startedAt));
        }
      }
    }

    const newStep = {
      stepId,
      turnId: trace.turnId,
      phase,
      title: stepTitle,
      status: isTerminalStep ? (rawStatus === "success" || rawStatus === "done" ? "completed" : rawStatus) : "running",
      detail,
      startedAt: eventTime(event) || new Date().toISOString(),
      endedAt: isTerminalStep ? (eventTime(event) || new Date().toISOString()) : "",
      durationMs,
      subEvents: eventId ? [{ ...event }] : (event ? [{ ...event }] : []),
      metadata: meta,
    };

    if (!isTerminalStep) {
      trace.activeStepId = stepId;
    } else {
      if (newStep.durationMs == null && newStep.startedAt && newStep.endedAt) {
        newStep.durationMs = Math.max(0, Date.parse(newStep.endedAt) - Date.parse(newStep.startedAt));
      }
    }

    trace.steps.push(newStep);
    trace.stepById.set(stepId, newStep);
    return newStep;
  }

  function createState(sessionId) {
    return {
      sessionId: text(sessionId),
      messages: new Map(),
      tools: new Map(),
      activities: new Map(),
      permissionRequests: new Map(),
      seenSequences: new Set(),
      seenUnsequenced: new Set(),
      lastSequence: 0,
      socketReady: false,
      submitting: false,
      runStatus: "idle",
      error: "",
      contextBudget: {},
      executionTraces: new Map(),
    };
  }

  function messageId(message) {
    const meta = message.metadata || {};
    return text(message.message_id || message.id || meta.message_id || meta.client_message_id);
  }

  function applyMessage(state, message) {
    const id = messageId(message);
    if (!id) return state;
    const previous = state.messages.get(id) || {};
    state.messages.set(id, {
      ...previous,
      ...message,
      id,
      message_id: id,
      role: text(message.role || previous.role || "system"),
      content: text(message.content != null ? message.content : previous.content),
      status: text(message.status || previous.status || "success"),
      optimistic: false,
      error: text(message.error || previous.error),
      created_at: text(message.created_at || previous.created_at),
      updated_at: text(message.updated_at || message.created_at || previous.updated_at),
    });
    return state;
  }

  function applyOptimistic(state, message) {
    const id = messageId(message);
    if (!id || state.messages.has(id)) return state;
    state.messages.set(id, {
      ...message,
      id,
      message_id: id,
      role: "user",
      content: text(message.content),
      status: "sending",
      optimistic: true,
      error: "",
      created_at: text(message.created_at || new Date().toISOString()),
    });
    if (!state.executionTraces.has(id)) {
      state.executionTraces.set(id, createTrace(id));
    }
    state.submitting = true;
    state.runStatus = "starting";
    return state;
  }

  function toolId(event) {
    const meta = metaOf(event);
    return text(
      event.tool_call_id ||
      meta.tool_call_id ||
      meta.call_id ||
      event.parent_event_id ||
      event.event_id ||
      event.id
    );
  }

  function assistantId(event) {
    const meta = metaOf(event);
    return text(meta.message_id || `assistant_${event.execution_id || event.turn_id || "active"}`);
  }

  function eventSummary(event) {
    const meta = metaOf(event);
    return text(
      meta.delta ||
      event.delta ||
      event.output_preview ||
      event.summary ||
      event.message ||
      meta.result_summary ||
      meta.progress
    );
  }

  function applyTool(state, event) {
    const id = toolId(event);
    if (!id) return;
    const meta = metaOf(event);
    const previous = state.tools.get(id) || {
      id,
      logs: [],
      progress: [],
      started_at: eventTime(event),
      first_sequence: eventSequence(event),
    };
    const type = eventType(event);
    const summary = eventSummary(event);
    const logs = [...previous.logs];
    const progress = [...previous.progress];
    if ((type === "tool.stdout" || type === "tool.stderr" || type.startsWith("log.")) && summary) {
      logs.push(summary);
    } else if (type === "tool.progress" && summary) {
      progress.push(summary);
    }
    const status = eventStatus(event);
    const resolvedStatus = terminal.has(previous.status) && !terminal.has(status)
      ? previous.status
      : status;
    state.tools.set(id, {
      ...previous,
      id,
      name: text(meta.tool_name || event.tool_name || previous.name || event.title || "tool"),
      arguments: meta.arguments || meta.args || meta.args_summary || previous.arguments || "",
      status: resolvedStatus,
      progress,
      logs,
      result: text(meta.result_summary || (type === "tool.finished" ? summary : previous.result)),
      error: text(event.error || meta.error || (type === "tool.failed" ? summary : previous.error)),
      started_at: text(previous.started_at || eventTime(event)),
      completed_at: terminal.has(resolvedStatus)
        ? text(previous.completed_at || event.ended_at || eventTime(event))
        : "",
      duration_ms: event.duration_ms != null ? Number(event.duration_ms) : previous.duration_ms,
      run_id: text(event.execution_id || event.turn_id || previous.run_id),
      last_sequence: eventSequence(event),
      details: { ...event, metadata: meta },
    });
  }

  function applyAssistant(state, event) {
    const type = eventType(event);
    const id = assistantId(event);
    const runId = text(event.execution_id || event.turn_id || "active");
    const temporaryId = `assistant_${runId}`;
    if (type === "turn.finished" && id !== temporaryId && state.messages.has(temporaryId)) {
      const temporary = state.messages.get(temporaryId);
      const canonical = state.messages.get(id) || {};
      state.messages.set(id, { ...temporary, ...canonical, id, message_id: id });
      state.messages.delete(temporaryId);
    }
    const previous = state.messages.get(id) || {
      id,
      message_id: id,
      role: "assistant",
      content: "",
      created_at: eventTime(event),
      deltas: new Map(),
    };
    if (!(previous.deltas instanceof Map)) previous.deltas = new Map();
    if (type === "assistant.delta") {
      const chunk = eventSummary(event);
      const key = eventSequence(event) || previous.deltas.size + 1;
      if (chunk && !previous.deltas.has(key)) previous.deltas.set(key, chunk);
      previous.content = [...previous.deltas.entries()]
        .sort((a, b) => a[0] - b[0])
        .map((entry) => entry[1])
        .join("");
    }
    const meta = metaOf(event);
    if (type === "turn.finished" && meta.content != null) previous.content = text(meta.content);
    previous.status = type === "assistant.started" ? "streaming" : eventStatus(event);
    previous.optimistic = false;
    previous.updated_at = eventTime(event);
    previous.run_id = runId;
    state.messages.set(id, previous);
  }

  function applyAcceptedMessage(state, event) {
    const meta = metaOf(event);
    const id = text(meta.message_id || meta.client_message_id);
    if (!id) return;
    const previous = state.messages.get(id) || {};
    applyMessage(state, {
      ...previous,
      message_id: id,
      role: "user",
      content: text(meta.content || event.summary || event.message || previous.content),
      status: "success",
      created_at: text(previous.created_at || eventTime(event)),
      execution_id: text(event.execution_id || event.turn_id),
    });
  }

  function markRunFailed(state, event) {
    const runId = text(event.execution_id || event.turn_id);
    const error = eventSummary(event) || text(event.error) || "Execution failed.";
    for (const message of state.messages.values()) {
      if (text(message.execution_id || message.run_id) === runId) {
        message.status = eventStatus(event);
        message.error = error;
        message.optimistic = false;
      }
    }
    const trace = resolveTrace(state, runId);
    if (trace) {
      const isCancelled = eventType(event).includes("cancelled") || eventStatus(event) === "cancelled";
      trace.isCancelled = isCancelled;
      trace.isFailed = !isCancelled;
      for (const step of trace.steps) {
        if (step.status === "running") {
          step.status = isCancelled ? "cancelled" : "failed";
          step.detail = error;
          step.endedAt = eventTime(event) || new Date().toISOString();
          if (step.durationMs == null && step.startedAt) {
            step.durationMs = Math.max(0, Date.parse(step.endedAt) - Date.parse(step.startedAt));
          }
        }
      }
      trace.activeStepId = null;
    }
    state.error = error;
    state.submitting = false;
    state.runStatus = eventStatus(event);
  }

  function applyEvent(state, event) {
    if (!event || typeof event !== "object") return state;
    const sequence = eventSequence(event);
    if (sequence && state.seenSequences.has(sequence)) return state;
    if (sequence) {
      state.seenSequences.add(sequence);
      state.lastSequence = Math.max(state.lastSequence, sequence);
    } else {
      const key = [event.event_id || event.id, eventType(event), eventStatus(event), eventSummary(event)].join("|");
      if (state.seenUnsequenced.has(key)) return state;
      state.seenUnsequenced.add(key);
    }
    applyTraceEvent(state, event);
    const type = eventType(event);
    const metadata = metaOf(event);
    if (type.startsWith("context.") || type.startsWith("cost.") || type.startsWith("budget.")) {
      state.contextBudget = { ...state.contextBudget, ...metadata, event_type: type, status: eventStatus(event) };
    }
    const permissionRequestId = text(metadata.permission_request_id);
    const inboxItemId = text(metadata.inbox_item_id);
    if ((type === "computer.waiting_permission" || type === "server.waiting_approval" || type === "api.waiting_approval" || type === "action.approval.required") && permissionRequestId) {
      const authorityId = type === "action.approval.required" && inboxItemId ? inboxItemId : permissionRequestId;
      const request = {
        requestId: authorityId,
        scope: text(metadata.permission_scope),
        preview: typeof (metadata.approval_display || metadata.preview) === "object"
          ? JSON.stringify(metadata.approval_display || metadata.preview, null, 2)
          : text(metadata.approval_display || metadata.preview || event.title || eventSummary(event)),
        kind: type === "server.waiting_approval" ? "server" : type === "api.waiting_approval" ? "api" : type === "action.approval.required" ? "transactional" : "computer",
        status: "pending",
        decision: "",
      };
      if (type === "action.approval.required") {
        request.actionId = text(metadata.action_id);
      }
      state.permissionRequests.set(authorityId, request);
    } else if ((type === "computer.permission_decided" || type === "server.approval_decided" || type === "api.approval_decided" || type === "action.approval.granted" || type === "action.approval.denied") && permissionRequestId) {
      const previous = state.permissionRequests.get(permissionRequestId) || {
        requestId: permissionRequestId,
      };
      state.permissionRequests.set(permissionRequestId, {
        ...previous,
        status: "decided",
        decision: text(metadata.decision),
      });
    } else if (type === "turn.resume_requested") {
      const reqId = text(metadata.approval_request_id) || permissionRequestId;
      if (reqId) {
        const previous = state.permissionRequests.get(reqId) || { requestId: reqId };
        state.permissionRequests.set(reqId, {
          ...previous,
          status: "resuming",
        });
      }
    }
    if (type === "message.accepted" || type === "turn.started" && metaOf(event).role === "user") {
      applyAcceptedMessage(state, event);
    } else if (type.startsWith("tool.") || type.startsWith("command.")) {
      applyTool(state, event);
    } else if (type === "assistant.started" || type === "assistant.delta" || type === "turn.finished") {
      applyAssistant(state, event);
      if (type === "turn.finished") {
        const reqId = text(metadata.approval_request_id);
        if (reqId && state.permissionRequests.has(reqId)) {
          const previous = state.permissionRequests.get(reqId);
          state.permissionRequests.set(reqId, {
            ...previous,
            status: "completed",
          });
        }
      }
    }
    const activityKey = text(event.event_id || event.id || `sequence-${sequence}`);
    if (!type.startsWith("tool.") && type !== "assistant.delta" && type !== "assistant.started" && type !== "message.accepted") {
      state.activities.set(`${activityKey}:${type}:${sequence}`, { ...event, sequence });
    }
    if (type === "turn.finished") {
      state.submitting = false;
      state.runStatus = eventStatus(event);
    } else if (type === "error" || type.endsWith(".cancelled") || eventStatus(event) === "failed") {
      markRunFailed(state, event);
    } else if (eventStatus(event) === "running") {
      state.runStatus = "running";
    }
    return state;
  }

  function reduce(state, action) {
    if (!action || !action.type) return state;
    if (action.type === "hydrate") {
      for (const message of action.messages || []) applyMessage(state, message);
      for (const event of [...(action.events || [])].sort((a, b) => eventSequence(a) - eventSequence(b))) {
        applyEvent(state, event);
      }
    } else if (action.type === "optimistic") {
      applyOptimistic(state, action.message || {});
    } else if (action.type === "event") {
      applyEvent(state, action.event || {});
    } else if (action.type === "submit_failed") {
      const message = state.messages.get(text(action.messageId));
      if (message) {
        message.status = "failed";
        message.error = text(message.error || action.error || "Submission failed.");
        message.optimistic = false;
      }
      const trace = resolveTrace(state, action.messageId);
      if (trace) {
        trace.isFailed = true;
        for (const step of trace.steps) {
          if (step.status === "running") {
            step.status = "failed";
            step.detail = text(action.error || "Submission failed.");
            step.endedAt = new Date().toISOString();
          }
        }
        trace.activeStepId = null;
      }
      state.submitting = false;
      state.runStatus = "failed";
      state.error = text(action.error || "Submission failed.");
    } else if (action.type === "socket") {
      state.socketReady = Boolean(action.ready);
    }
    return state;
  }

  function snapshot(state) {
    const cleanMessage = (message) => {
      const copy = { ...message };
      delete copy.deltas;
      return copy;
    };
    return {
      sessionId: state.sessionId,
      messages: [...state.messages.values()].map(cleanMessage),
      tools: [...state.tools.values()],
      activities: [...state.activities.values()],
      permissionRequests: [...state.permissionRequests.values()],
      lastSequence: state.lastSequence,
      socketReady: state.socketReady,
      submitting: state.submitting,
      runStatus: state.runStatus,
      error: state.error,
      contextBudget: { ...state.contextBudget },
      executionTraces: [...state.executionTraces.values()].map(cleanTrace),
    };
  }

  function init(config) {
    const mount = document.getElementById(config.mountId);
    const state = createState(config.sessionId);
    reduce(state, { type: "hydrate", messages: config.messages, events: config.events });
    let socket = null;
    let reconnects = 0;
    let closed = false;

    mount.innerHTML = `
      <style>
        #mana-live-chat,#mana-live-chat *{box-sizing:border-box} #mana-live-chat{color:#e8eaed;background:transparent;font:14px ui-sans-serif,system-ui}
        .shell{display:flex;flex-direction:column;height:${Number(config.height || 680)}px;border:1px solid #ffffff22;border-radius:12px;overflow:hidden}
        .status{padding:8px 12px;border-bottom:1px solid #ffffff18;color:#aeb4bd;display:flex;gap:10px;align-items:center}
        .dot{width:8px;height:8px;border-radius:50%;background:#f59e0b}.dot.ok{background:#22c55e}
        .timeline{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
        .message{max-width:88%;padding:10px 12px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}
        .user{align-self:flex-end;background:#2563eb}.assistant{align-self:flex-start;background:#272b33}
        .meta{font-size:11px;opacity:.72;margin-top:5px}.failed{border:1px solid #ef4444}.sending{opacity:.72}
        .tool,.activity{background:#171a20;border:1px solid #ffffff1c;border-radius:9px;padding:8px 10px}
        .running{border-color:#f59e0b88}.success{border-color:#22c55e66}.failed-card{border-color:#ef444488}
        summary{cursor:pointer}.tool pre{white-space:pre-wrap;max-height:220px;overflow:auto;color:#cbd5e1}
        form{display:flex;gap:8px;padding:10px;border-top:1px solid #ffffff18} textarea{flex:1;min-height:44px;max-height:120px;resize:vertical;border-radius:9px;padding:10px;background:#11141a;color:#fff;border:1px solid #ffffff28}
        button{border:0;border-radius:9px;padding:0 18px;background:#2563eb;color:#fff;font-weight:600}button:disabled{opacity:.45}
        .permission{border-color:#f59e0b88}.permission-actions{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}
        .permission-actions button{min-height:34px;padding:6px 12px}.permission-actions .deny{background:#b91c1c}
        .permission-state{margin-top:7px;color:#aeb4bd}.permission-error{margin-top:7px;color:#fca5a5}
        .error{color:#fca5a5}.logs{font-size:12px;color:#aeb4bd;white-space:pre-wrap}
        .execution-trace{display:flex;flex-direction:column;gap:4px;margin:6px 0 10px 8px;padding:8px 12px;background:#13161c;border-left:2px solid #3b82f644;border-radius:6px;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px}
        .trace-step{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:2px 0;line-height:1.4}
        .trace-step.running{opacity:.68;color:#94a3b8}
        .trace-step.completed{opacity:.95;color:#cbd5e1}
        .trace-step.failed{opacity:1;color:#fca5a5}
        .trace-step.cancelled{opacity:.8;color:#fcd34d}
        .step-icon{font-size:13px;width:14px;text-align:center;flex-shrink:0}
        .trace-step.running .step-icon{color:#60a5fa;animation:pulse 1.6s ease-in-out infinite}
        .trace-step.completed .step-icon{color:#22c55e}
        .trace-step.failed .step-icon{color:#ef4444}
        .trace-step.cancelled .step-icon{color:#f59e0b}
        .step-title{font-weight:500}
        .step-meta{color:#64748b;font-size:11px}
        .coding-box,.step-details{width:100%;margin-top:4px;margin-bottom:4px;background:#1a1e26;border:1px solid #ffffff14;border-radius:6px;padding:6px 8px;font-size:11px}
        .coding-box summary,.step-details summary{cursor:pointer;color:#93c5fd;font-weight:500;user-select:none}
        .coding-box pre,.step-details pre{margin:6px 0 0 0;white-space:pre-wrap;max-height:160px;overflow:auto;color:#cbd5e1}
        .step-error{width:100%;color:#fca5a5;font-size:11px;margin-top:2px}
        @keyframes pulse{0%,100%{opacity:.45}50%{opacity:1}}
        @media(max-width:520px){.message{max-width:96%}.shell{border-radius:7px}.timeline{padding:8px}}
      </style>
      <div class="shell"><div class="status"><span class="dot"></span><span class="statusText">Connecting to live events…</span><span class="contextMeter"></span></div>
      <div class="timeline"></div><form><textarea aria-label="Chat message" placeholder="Message this conversation"></textarea><button type="submit">Send</button></form></div>`;
    const timeline = mount.querySelector(".timeline");
    const form = mount.querySelector("form");
    const input = mount.querySelector("textarea");
    const submitButton = form.querySelector("button");
    const dot = mount.querySelector(".dot");
    const statusText = mount.querySelector(".statusText");
    const contextMeter = mount.querySelector(".contextMeter");
    const permissionBusy = new Set();
    const permissionErrors = new Map();

    const addText = (parent, tag, value, className) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      node.textContent = text(value);
      parent.appendChild(node);
      return node;
    };
    const render = () => {
      const nearBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 80;
      timeline.replaceChildren();
      const rows = [];
      for (const message of state.messages.values()) rows.push({ kind: "message", time: message.created_at || "", sequence: 0, value: message });
      for (const tool of state.tools.values()) rows.push({ kind: "tool", time: tool.started_at || "", sequence: tool.first_sequence || 0, value: tool });
      for (const activity of state.activities.values()) rows.push({ kind: "activity", time: eventTime(activity), sequence: activity.sequence || 0, value: activity });
      rows.sort((a, b) => a.time.localeCompare(b.time) || a.sequence - b.sequence);
      for (const row of rows) {
        if (row.kind === "message") {
          const item = row.value;
          const node = document.createElement("div");
          node.className = `message ${item.role === "user" ? "user" : "assistant"} ${item.status === "failed" ? "failed" : ""} ${item.optimistic ? "sending" : ""}`;
          addText(node, "div", item.content || (item.status === "streaming" ? "…" : ""));
          const meta = item.error ? `${item.status} · ${item.error}` : item.optimistic ? "sending…" : item.status === "streaming" ? "streaming…" : "";
          if (meta) addText(node, "div", meta, `meta ${item.error ? "error" : ""}`);
          timeline.appendChild(node);

          if (item.role === "user") {
            const trace = resolveTrace(state, item.execution_id || item.message_id || item.id);
            if (trace && trace.steps.length > 0) {
              const traceNode = document.createElement("div");
              traceNode.className = "execution-trace";
              for (const step of trace.steps) {
                const stepRow = document.createElement("div");
                stepRow.className = `trace-step ${step.status}`;

                const icon = step.status === "running" ? "◌" : step.status === "completed" ? "✓" : step.status === "failed" ? "✗" : "⊘";
                addText(stepRow, "span", icon, "step-icon");
                addText(stepRow, "span", step.title, "step-title");

                if (step.status === "running") {
                  const elapsed = step.startedAt ? ` · ${formatDuration(Math.max(0, Date.now() - Date.parse(step.startedAt)))}` : "";
                  addText(stepRow, "span", `· running${elapsed}`, "step-meta");
                } else if (step.status === "completed" && step.durationMs != null) {
                  addText(stepRow, "span", `(${formatDuration(step.durationMs)})`, "step-meta");
                } else if (step.status === "failed") {
                  addText(stepRow, "span", "· failed", "step-meta");
                } else if (step.status === "cancelled") {
                  addText(stepRow, "span", "· cancelled", "step-meta");
                }

                if (step.phase === "coding" && step.subEvents.length > 0) {
                  const codingBox = document.createElement("details");
                  codingBox.className = "coding-box";
                  if (step.status === "running") codingBox.open = true;
                  const cmdCount = step.subEvents.filter((se) => {
                    const st = eventType(se);
                    return st.startsWith("command.") || st.startsWith("tool.") || st.includes("terminal");
                  }).length;
                  const summaryLabel = cmdCount > 0 ? `Coding activity (${cmdCount} commands)` : `Coding activity (${step.subEvents.length} actions)`;
                  addText(codingBox, "summary", summaryLabel);
                  const subLogs = [];
                  for (const se of step.subEvents) {
                    const sum = eventSummary(se);
                    const seMeta = metaOf(se);
                    const cmd = seMeta.command || seMeta.cmd || se.title || "";
                    if (cmd && !subLogs.includes(`$ ${cmd}`)) subLogs.push(`$ ${cmd}`);
                    if (sum && !subLogs.includes(sum)) subLogs.push(sum);
                  }
                  if (subLogs.length > 0) addText(codingBox, "pre", subLogs.join("\n"));
                  stepRow.appendChild(codingBox);
                } else if (step.phase === "tool" || step.phase === "search") {
                  const toolCallId = text(step.metadata.tool_call_id || step.metadata.call_id || step.stepId.split(":tool:")[1] || step.stepId.split(":search:")[1]);
                  const toolObj = state.tools.get(toolCallId);
                  const args = toolObj ? toolObj.arguments : (step.metadata.arguments || step.metadata.args_summary);
                  const logs = toolObj ? toolObj.logs : [];
                  const result = toolObj ? toolObj.result : step.detail;
                  if (args || logs.length > 0 || result) {
                    const toolBox = document.createElement("details");
                    toolBox.className = "step-details";
                    addText(toolBox, "summary", "Details");
                    if (args) addText(toolBox, "div", `Arguments: ${typeof args === "string" ? args : JSON.stringify(args)}`, "logs");
                    if (logs.length > 0) addText(toolBox, "div", logs.join("\n"), "logs");
                    if (result) addText(toolBox, "div", `Result: ${result}`, "logs");
                    stepRow.appendChild(toolBox);
                  }
                }

                if (step.status === "failed" && step.detail) {
                  addText(stepRow, "div", step.detail, "step-error");
                }

                traceNode.appendChild(stepRow);
              }
              timeline.appendChild(traceNode);
            }
          }
        } else if (row.kind === "tool") {
          const tool = row.value;
          const alreadyInTrace = [...state.executionTraces.values()].some((t) =>
            t.steps.some((s) => (s.phase === "tool" || s.phase === "search") && (s.stepId.endsWith(`:${tool.id}`) || text(s.metadata.tool_call_id) === tool.id))
          );
          if (alreadyInTrace) continue;
          const node = document.createElement("details");
          node.className = `tool ${tool.status === "running" ? "running" : tool.status === "success" ? "success" : "failed-card"}`;
          const elapsed = tool.status === "running" && tool.started_at ? ` · ${Math.max(0, (Date.now() - Date.parse(tool.started_at)) / 1000).toFixed(1)}s` : tool.duration_ms != null ? ` · ${(tool.duration_ms / 1000).toFixed(2)}s` : "";
          addText(node, "summary", `${tool.status === "running" ? "⏳" : tool.status === "success" ? "✅" : "❌"} ${tool.name} · ${tool.status}${elapsed}`);
          if (tool.arguments) addText(node, "div", `Arguments: ${typeof tool.arguments === "string" ? tool.arguments : JSON.stringify(tool.arguments)}`, "logs");
          if (tool.progress.length) addText(node, "div", tool.progress.join("\n"), "logs");
          if (tool.logs.length) addText(node, "div", tool.logs.join("\n"), "logs");
          if (tool.result) addText(node, "div", `Result: ${tool.result}`, "logs");
          if (tool.error) addText(node, "div", `Error: ${tool.error}`, "logs error");
          addText(node, "pre", JSON.stringify(tool.details || {}, null, 2));
          timeline.appendChild(node);
        } else {
          const event = row.value;
          const node = document.createElement("div");
          const type = eventType(event);
          const metadata = metaOf(event);
          const permissionRequestId = text(metadata.permission_request_id);
          const inboxItemId = text(metadata.inbox_item_id);
          if ((type === "computer.waiting_permission" || type === "server.waiting_approval" || type === "api.waiting_approval" || type === "action.approval.required") && permissionRequestId) {
            node.className = "activity permission";
            const authorityId = type === "action.approval.required" && inboxItemId ? inboxItemId : permissionRequestId;
            const request = state.permissionRequests.get(authorityId) || {};
            const serverApproval = type === "server.waiting_approval" || request.kind === "server";
            const apiApproval = type === "api.waiting_approval" || request.kind === "api";
            const transactionalApproval = type === "action.approval.required" || request.kind === "transactional";
            addText(node, "div", serverApproval ? "Server action approval required" : apiApproval ? "API request approval required" : transactionalApproval ? "Transactional action approval required" : "Computer permission required");
            addText(node, "div", request.preview || metadata.preview || event.title, "logs");
            addText(node, "div", `Scope: ${request.scope || metadata.permission_scope || ""}`, "meta");
            if (request.status === "completed" || request.status === "approved" || request.status === "denied" || request.status === "decided") {
              const badgeText = request.status === "completed"
                ? "Completed · Resumed"
                : request.status === "approved"
                ? "Approved · Executed"
                : `Decision: ${request.decision || request.status}`;
              addText(node, "div", badgeText, "permission-state");
              if (request.result) addText(node, "div", request.result, "logs");
            } else if (request.status === "resuming" || permissionBusy.has(authorityId)) {
              addText(node, "div", "Approving & Resuming...", "permission-state");
            } else {
              const actions = document.createElement("div");
              actions.className = "permission-actions";
              const choices = serverApproval || apiApproval || transactionalApproval
                ? [["Deny", "deny", "deny"], ["Approve once", "approve", ""]]
                : [
                    ["Deny", "deny", "deny"],
                    ["Allow once", "allow_once", ""],
                    ["This session", "allow_session", ""],
                    ["Always", "always", ""],
                  ];
              for (const [label, decision, className] of choices) {
                const decisionButton = document.createElement("button");
                decisionButton.type = "button";
                decisionButton.textContent = label;
                decisionButton.className = className;
                decisionButton.disabled = permissionBusy.has(authorityId);
                decisionButton.addEventListener("click", async () => {
                  permissionBusy.add(authorityId);
                  permissionErrors.delete(authorityId);
                  render();
                  try {
                    const permissionPath = serverApproval ? "server-approvals" : apiApproval ? "api-approvals" : transactionalApproval ? "transactional-actions" : "computer-permissions";
                    const requestTarget = transactionalApproval ? authorityId : permissionRequestId;
                    const response = await fetch(
                      `${config.apiBase}/api/v1/conversations/${encodeURIComponent(config.sessionId)}/${permissionPath}/${encodeURIComponent(requestTarget)}`,
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) },
                        body: JSON.stringify({ decision, root: config.root }),
                      },
                    );
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
                    const resultMessage = text(payload.answer || (payload.result && (payload.result.answer || payload.result.message)));
                    if (payload.assistant_message) applyMessage(state, payload.assistant_message);
                    state.permissionRequests.set(authorityId, {
                      ...request,
                      requestId: authorityId,
                      status: payload.status || (payload.approved ? "completed" : "denied"),
                      decision,
                      result: resultMessage,
                    });
                  } catch (error) {
                    permissionErrors.set(authorityId, error.message || String(error));
                  } finally {
                    permissionBusy.delete(authorityId);
                    render();
                  }
                });
                actions.appendChild(decisionButton);
              }
              node.appendChild(actions);
              if (permissionErrors.has(authorityId)) {
                addText(node, "div", permissionErrors.get(authorityId), "permission-error");
              }
            }
          } else {
            const inTrace = [...state.executionTraces.values()].some((t) =>
              t.steps.some((s) => s.subEvents.some((se) => text(se.event_id || se.id) === text(event.event_id || event.id)))
            );
            if (inTrace) continue;
            node.className = "activity";
            addText(node, "div", `${event.title || type} · ${eventStatus(event)}`);
            if (eventSummary(event)) addText(node, "div", eventSummary(event), "logs");
            if (type.startsWith("media_generation_")) {
              const mediaLabel = [
                metadata.media_type,
                metadata.provider && metadata.model ? `${metadata.provider}/${metadata.model}` : "",
                metadata.progress != null ? `${Math.round(Number(metadata.progress) * 100)}%` : "",
              ].filter(Boolean).join(" · ");
              if (mediaLabel) addText(node, "div", mediaLabel, "meta");
              if (metadata.artifact_id) addText(node, "div", `Artifact: ${metadata.artifact_id}`, "logs");
              if (metadata.error) addText(node, "div", metadata.error, "permission-error");
            }
            if (type === "canvas.createSurface") {
              const link = document.createElement("a");
              const canvas = metadata.canvas_event || {};
              link.textContent = "Open Live Canvas";
              link.href = `${config.apiBase}/api/v1/dashboard/live-canvas?conversation_id=${encodeURIComponent(config.sessionId)}&root=${encodeURIComponent(config.root)}&surface_id=${encodeURIComponent(canvas.surface_id || metadata.surface_id || "")}`;
              link.target = "_blank";
              link.rel = "noopener noreferrer";
              link.style.color = "#93c5fd";
              node.appendChild(link);
            }
          }
          timeline.appendChild(node);
        }
      }
      submitButton.disabled = !state.socketReady || state.submitting;
      dot.classList.toggle("ok", state.socketReady);
      statusText.textContent = state.socketReady
        ? state.submitting || state.runStatus === "running" || state.runStatus === "starting" ? "Agent is working · live" : "Live events connected"
        : "Reconnecting to live events…";
      const budget = state.contextBudget || {};
      const breakdown = budget.breakdown || {};
      const percent = Math.round(Number(budget.utilization_ratio || 0) * 100);
      const exact = budget.estimated === false ? "exact" : "est";
      contextMeter.textContent = budget.context_window
        ? `ctx ${budget.used_tokens || 0}/${budget.context_window} (${percent}%) · schema ${breakdown.schema_tokens || budget.schema_tokens || 0} · ${exact} $${Number(budget.cumulative_cost || 0).toFixed(4)}${budget.remaining_cost == null ? "" : ` · $${Number(budget.remaining_cost).toFixed(4)} left`}`
        : "";
      if (nearBottom) timeline.scrollTop = timeline.scrollHeight;
    };

    const dispatchEvent = (event) => { reduce(state, { type: "event", event }); render(); };
    const socketUrl = () => `${config.wsBase}/api/v1/ws/conversations/${encodeURIComponent(config.sessionId)}?root=${encodeURIComponent(config.root)}&replay_limit=1000&after_sequence=${state.lastSequence}${config.token ? `&token=${encodeURIComponent(config.token)}` : ""}`;
    const switchSession = (sessionId) => {
      const replacementId = text(sessionId);
      if (!replacementId || replacementId === config.sessionId) return;
      config.sessionId = replacementId;
      state.sessionId = replacementId;
      state.messages.clear();
      state.tools.clear();
      state.activities.clear();
      state.permissionRequests.clear();
      state.seenSequences.clear();
      state.seenUnsequenced.clear();
      state.lastSequence = 0;
      state.submitting = false;
      state.runStatus = "idle";
      state.error = "";
      state.contextBudget = {};
      state.executionTraces.clear();
      reduce(state, { type: "socket", ready: false });
      // Keep the Streamlit shell informed when it later gains a message bridge.
      // The embedded client itself immediately reconnects to the new session.
      // Restrict target origin: same origin, or the embedding referrer when it is loopback.
      const parentOrigin = (() => {
        try {
          if (document.referrer) {
            const origin = new URL(document.referrer).origin;
            const host = new URL(origin).hostname;
            if (origin === window.location.origin || host === "localhost" || host === "127.0.0.1") {
              return origin;
            }
          }
        } catch (_error) { /* fall through */ }
        return window.location.origin;
      })();
      window.parent.postMessage(
        { type: "mana-live-chat.session-replaced", conversationId: replacementId },
        parentOrigin
      );
      if (socket) socket.close();
    };
    const connect = () => {
      if (closed) return;
      socket = new WebSocket(socketUrl());
      socket.onmessage = (message) => {
        const packet = JSON.parse(message.data);
        if (packet.type === "socket.ready") {
          reduce(state, { type: "socket", ready: true });
          reconnects = 0;
          render();
        } else if (packet.type === "event" || packet.type === "event.replay") {
          dispatchEvent(packet.event);
        }
      };
      socket.onclose = () => {
        reduce(state, { type: "socket", ready: false });
        render();
        if (!closed) setTimeout(connect, Math.min(10000, 400 * Math.pow(2, reconnects++)));
      };
      socket.onerror = () => socket.close();
    };

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const content = input.value.trim();
      if (!content || state.submitting || !state.socketReady) return;
      const id = `client_${Date.now().toString(36)}_${crypto.randomUUID().replaceAll("-", "")}`;
      reduce(state, { type: "optimistic", message: { message_id: id, content, created_at: new Date().toISOString() } });
      input.value = "";
      render();
      try {
        const response = await fetch(`${config.apiBase}/api/v1/conversations/${encodeURIComponent(config.sessionId)}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) },
          body: JSON.stringify({ content, client_message_id: id, root: config.root }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
        // Commands such as /new replace the canonical session. They have no
        // turn.finished event, so explicitly bind the live client to the
        // replacement instead of leaving the old, deleted session "working".
        if (payload.conversation_id && payload.conversation_id !== config.sessionId) {
          switchSession(payload.conversation_id);
          render();
          return;
        }
        reduce(state, { type: "hydrate", messages: [payload.user_message, payload.assistant_message].filter(Boolean), events: payload.events || [] });
        render();
      } catch (error) {
        reduce(state, { type: "submit_failed", messageId: id, error: error.message || String(error) });
        render();
      }
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    const elapsedTimer = setInterval(() => {
      const hasRunningTools = [...state.tools.values()].some((tool) => tool.status === "running");
      const hasRunningTraces = [...state.executionTraces.values()].some((t) => t.steps.some((s) => s.status === "running"));
      if (hasRunningTools || hasRunningTraces) render();
    }, 250);
    window.addEventListener("beforeunload", () => {
      closed = true;
      clearInterval(elapsedTimer);
      if (socket) socket.close();
    }, { once: true });
    render();
    connect();
    return { state, reduce, snapshot: () => snapshot(state), close: () => { closed = true; if (socket) socket.close(); clearInterval(elapsedTimer); } };
  }

  return { createState, reduce, snapshot, init };
});

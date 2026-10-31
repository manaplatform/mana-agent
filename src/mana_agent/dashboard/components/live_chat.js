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

  const formatSize = (bytes) => {
    const b = Number(bytes || 0);
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / (1024 * 1024)).toFixed(1)} MB`;
  };

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

  const CODING_TOOLS = new Set([
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "write_file",
    "create_file",
    "delete_file",
    "patch",
    "read_file",
    "repo_batch_read",
    "list_files",
    "ls",
    "find_symbols",
    "call_graph",
    "chunk_file",
    "list_tools",
    "tool_contracts",
    "read_skill",
    "git",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
    "git_branch",
    "git_switch",
    "git_checkout",
    "git_create_branch",
    "git_add",
    "git_restore",
    "git_stash",
    "git_commit",
    "git_push",
    "git_pull",
    "git_fetch",
    "git_remote",
    "git_tag",
    "git_merge",
    "git_rebase",
    "git_revert",
    "git_reset",
    "git_clean",
    "git_config",
    "git_generic",
    "git_help",
    "shell",
    "run_command",
    "run_script_once",
    "verify_project",
    "exec_command",
    "bash",
    "terminal",
    "local_shell",
  ]);

  function isCodingTool(toolName) {
    const name = text(toolName || "").trim().toLowerCase();
    if (!name) return false;
    if (name.includes("search") || SEARCH_TOOLS.has(name)) return false;
    if (
      name.startsWith("email_") ||
      name.startsWith("server_") ||
      name.startsWith("api_") ||
      name.startsWith("canvas_") ||
      name.startsWith("browser_") ||
      name.startsWith("media_") ||
      name.startsWith("generate_")
    ) {
      return false;
    }
    if (name === "approval_request" || name === "request_user_approval" || name === "wait_for_approval") {
      return false;
    }
    if (CODING_TOOLS.has(name)) return true;
    if (name.startsWith("git.") || name.startsWith("git_") || name.startsWith("patch.") || name.startsWith("patch_")) {
      return true;
    }
    if (name.startsWith("edit_") || name.startsWith("write_")) {
      return true;
    }
    return false;
  }

  function classifyRuntimePhase(eventType, metadata) {
    const et = text(eventType || "").trim().toLowerCase();
    const meta = metadata || {};
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
      ((meta.backend === "codex" || meta.backend === "internal" || meta.backend === "coding") &&
        !et.startsWith("turn.") &&
        !et.startsWith("search.") &&
        !et.startsWith("routing.") &&
        (!et.startsWith("tool.") || isCodingTool(toolName)))
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

  function isCodingEvent(type, toolName, meta) {
    const et = text(type).toLowerCase();
    const tn = text(toolName || (meta && (meta.tool_name || meta.name))).toLowerCase();
    const backend = text(meta && meta.backend).toLowerCase();
    if (backend === "codex" || backend === "internal" || backend === "coding") {
      if (!et.startsWith("turn.") && !et.startsWith("routing.") && !et.startsWith("model.")) {
        return true;
      }
    }
    if (tn && isCodingTool(tn)) return true;
    if (
      et.startsWith("command.") ||
      et.startsWith("patch.") ||
      et.startsWith("file.") ||
      et === "coding_started" ||
      et === "coding.terminal" ||
      et === "coding.progress" ||
      et === "coding"
    ) {
      return true;
    }
    return false;
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
      mainBox: {
        events: [],
        eventIdSet: new Set(),
        status: "running",
        startedAt: new Date().toISOString(),
        endedAt: null,
        durationMs: null,
      },
      codingBox: {
        events: [],
        eventIdSet: new Set(),
        changedFiles: new Set(),
        commands: [],
        backend: "",
        model: "",
        status: "running",
        startedAt: new Date().toISOString(),
        endedAt: null,
        durationMs: null,
      },
    };
  }

  function cleanTrace(trace) {
    return {
      turnId: trace.turnId,
      activeStepId: trace.activeStepId,
      isCompleted: trace.isCompleted,
      isFailed: trace.isFailed,
      isCancelled: trace.isCancelled,
      mainBox: {
        status: trace.mainBox ? trace.mainBox.status : "running",
        events: trace.mainBox ? [...trace.mainBox.events] : [],
      },
      codingBox: {
        status: trace.codingBox ? trace.codingBox.status : "running",
        backend: trace.codingBox ? trace.codingBox.backend : "",
        model: trace.codingBox ? trace.codingBox.model : "",
        changedFiles: trace.codingBox ? [...trace.codingBox.changedFiles] : [],
        commands: trace.codingBox ? [...trace.codingBox.commands] : [],
        events: trace.codingBox ? [...trace.codingBox.events] : [],
      },
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
    const meta = { ...metaOf(event) };
    if (event.backend && !meta.backend) meta.backend = event.backend;
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
      updateTraceActivityBoxes(trace, event, type, meta, rawStatus, isTerminalStep, stepTitle, detail, durationMs, eventId);
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
    updateTraceActivityBoxes(trace, event, type, meta, rawStatus, isTerminalStep, stepTitle, detail, durationMs, eventId);
    return newStep;
  }

  function updateTraceActivityBoxes(trace, event, type, meta, rawStatus, isTerminalStep, stepTitle, detail, durationMs, eventId) {
    if (!trace) return;
    if (!trace.mainBox) trace.mainBox = { events: [], eventIdSet: new Set(), status: "running" };
    if (!trace.codingBox) trace.codingBox = { events: [], eventIdSet: new Set(), changedFiles: new Set(), commands: [], backend: "", model: "", status: "running" };

    const isCoding = isCodingEvent(type, (meta && (meta.tool_name || meta.name)) || (event && event.tool_name), meta);
    const box = isCoding ? trace.codingBox : trace.mainBox;
    const boxEventId = eventId || `${type}:${Date.now()}`;
    const opStatus = isTerminalStep ? (rawStatus === "success" || rawStatus === "done" ? "completed" : rawStatus) : rawStatus;

    if (!box.eventIdSet.has(boxEventId)) {
      box.eventIdSet.add(boxEventId);
      box.events.push({
        id: boxEventId,
        type,
        title: stepTitle,
        tool_name: text((meta && (meta.tool_name || meta.name)) || (event && event.tool_name) || stepTitle),
        status: opStatus,
        detail,
        time: eventTime(event) || new Date().toISOString(),
        durationMs,
        metadata: meta,
      });
    } else {
      for (const be of box.events) {
        if (be.id === boxEventId) {
          be.status = opStatus;
          if (detail && !be.detail) be.detail = detail;
          if (durationMs != null) be.durationMs = durationMs;
          break;
        }
      }
    }

    if (isCoding) {
      if (meta && meta.backend) trace.codingBox.backend = text(meta.backend);
      if (meta && (meta.model || (event && event.model))) trace.codingBox.model = text(meta.model || event.model);
      const filePath = text((event && event.path) || (meta && (meta.path || meta.filePath || (meta.args && meta.args.path) || (meta.arguments && meta.arguments.path))));
      if (filePath) trace.codingBox.changedFiles.add(filePath);
      if (meta && Array.isArray(meta.changed_files)) {
        for (const cf of meta.changed_files) {
          const cfs = text(cf);
          if (cfs) trace.codingBox.changedFiles.add(cfs);
        }
      }
      const cmd = text((meta && (meta.command || meta.cmd || (meta.args && meta.args.command))) || (event && event.command));
      if (cmd && !trace.codingBox.commands.includes(cmd)) {
        trace.codingBox.commands.push(cmd);
      }
    }
    const hasRunning = box.events.some((e) => e.status === "running");
    if (hasRunning) {
      box.status = "running";
    } else if (trace.isCancelled) {
      box.status = "cancelled";
    } else if (box.events.some((e) => e.status === "failed") || trace.isFailed) {
      box.status = "failed";
    } else if (isTerminalStep || box.events.length > 0) {
      box.status = opStatus === "failed" ? "failed" : "completed";
    }
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
      attachments: message.attachments || previous.attachments || [],
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
      attachments: message.attachments || [],
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

    const turnId = getTurnId(state, event);
    let trace = resolveTrace(state, turnId);
    if (!trace) {
      trace = createTrace(turnId);
      state.executionTraces.set(turnId, trace);
    }
    const isTerminal = terminal.has(resolvedStatus) || resolvedStatus === "success" || resolvedStatus === "failed";
    updateTraceActivityBoxes(
      trace,
      event,
      type || "tool",
      meta,
      resolvedStatus,
      isTerminal,
      text(meta.tool_name || event.tool_name || previous.name || event.title || "tool"),
      summary || previous.result || previous.error,
      event.duration_ms != null ? Number(event.duration_ms) : previous.duration_ms,
      id
    );
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
    const permissionRequestId = text((type.startsWith("action.approval.") && metadata.inbox_item_id) || metadata.permission_request_id);
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
      const previous = state.permissionRequests.get(authorityId);
      // Replayed or duplicate request events must not reopen a recorded decision.
      if (!previous || previous.status === "pending") {
        state.permissionRequests.set(authorityId, { ...previous, ...request });
      }
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
      if (reqId && (!state.permissionRequests.has(reqId) || state.permissionRequests.get(reqId).status === "pending")) {
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
        .main-activity-box,.coding-activity-box{background:#151821;border:1px solid #ffffff1c;border-radius:8px;margin:6px 0;padding:8px 12px;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px}
        .main-activity-box{border-left:3px solid #3b82f6}
        .coding-activity-box{border-left:3px solid #10b981}
        .main-activity-box>summary,.coding-activity-box>summary{cursor:pointer;display:flex;align-items:center;gap:8px;font-weight:600;color:#e2e8f0;user-select:none}
        .activity-box-title{font-size:13px;font-weight:600}
        .activity-box-badge{background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:11px;color:#94a3b8}
        .activity-stream{margin-top:8px;display:flex;flex-direction:column;gap:4px;max-height:280px;overflow-y:auto;scroll-behavior:auto}
        .activity-event-row{display:flex;flex-direction:column;gap:2px;padding:4px 6px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px;border-radius:4px;transition:background 0.2s}
        .activity-event-row:last-child{border-bottom:none}
        .activity-event-row.activity-latest{background:rgba(59,130,246,0.12);border-left:2px solid #3b82f6}
        .activity-event-header{display:flex;align-items:center;justify-content:space-between;gap:6px}
        .activity-event-title{display:flex;align-items:center;gap:6px;font-weight:500;color:#cbd5e1}
        .activity-event-details{margin-top:4px;padding:4px 6px;background:rgba(0,0,0,0.2);border-radius:4px}
        .activity-event-details summary{cursor:pointer;color:#93c5fd;font-size:11px;user-select:none}
        .activity-row{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px}
        .activity-row:last-child{border-bottom:none}
        .activity-row-title{font-weight:500;color:#93c5fd}
        .activity-row-cmd{font-weight:600;color:#38bdf8}
        .activity-row-summary{color:#cbd5e1;flex:1}
        .activity-overflow-note{color:#94a3b8;font-style:italic;font-size:11px;padding:2px 0}
        .changed-files-bar{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
        .changed-file-chip{background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);color:#6ee7b7;padding:2px 6px;border-radius:4px;font-size:11px}
        .cmd-output,.error-log{width:100%;margin:4px 0 0 0;white-space:pre-wrap;max-height:120px;overflow:auto;font-size:11px;background:#0d1117;padding:6px;border-radius:4px}
        .error-log{color:#fca5a5}
        .cmd-output{color:#cbd5e1}
        form{display:flex;gap:8px;padding:10px;border-top:1px solid #ffffff18} textarea{flex:1;min-height:44px;max-height:120px;resize:vertical;border-radius:9px;padding:10px;background:#11141a;color:#fff;border:1px solid #ffffff28}
        button{border:0;border-radius:9px;padding:0 18px;background:#2563eb;color:#fff;font-weight:600}button:disabled{opacity:.45}
        .attach-btn{background:transparent !important;border:1px solid #ffffff28 !important;font-size:16px;padding:0 12px !important;cursor:pointer;border-radius:9px;display:flex;align-items:center;justify-content:center}
        .attach-btn:hover{background:#ffffff12 !important}
        .attachment-preview-bar{display:none;flex-wrap:wrap;gap:8px;padding:8px 12px;background:#161b22;border-top:1px solid #ffffff18}
        .attachment-chip{display:inline-flex;align-items:center;gap:6px;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:4px 8px;font-size:12px;color:#c9d1d9}
        .attachment-chip.uploading{opacity:.65;font-style:italic}
        .attachment-chip.error{border-color:#ef4444;color:#fca5a5}
        .remove-att{cursor:pointer;color:#8b949e;font-weight:bold;margin-left:4px;padding:0 2px}
        .remove-att:hover{color:#f85149}
        .drop-active{border-color:#3b82f6 !important;background:#1e3a8a22 !important}
        .message-attachments{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}
        .message-attachment-card{display:flex;align-items:center;gap:6px;background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.15);border-radius:6px;padding:4px 8px;font-size:11px}
        .permission{border-color:#f59e0b88}.permission-actions{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}
        .permission-actions button{min-height:34px;padding:6px 12px}.permission-actions .deny{background:#b91c1c}
        .permission-state{margin-top:7px;color:#aeb4bd}.permission-error{margin-top:7px;color:#fca5a5}
        .approval-drawer{flex:none;background:#1b202b;border-bottom:1px solid #f59e0b88;max-height:45%;overflow:auto;padding:8px 12px}
        .approval-drawer>summary{cursor:pointer;font-weight:600;color:#fcd34d}
        .approval-drawer-item{padding:8px 0;border-top:1px solid #ffffff18;overflow-wrap:anywhere}
        .approval-drawer-item .logs{max-height:160px;overflow:auto}
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
      <details class="approval-drawer" aria-label="Pending approvals" hidden><summary></summary><div class="approval-drawer-items"></div></details>
      <div class="timeline"></div><div class="attachment-preview-bar"></div><form><input type="file" id="mana-file-input" multiple style="display:none" /><button type="button" class="attach-btn" title="Attach file" aria-label="Attach file">📎</button><textarea aria-label="Chat message" placeholder="Message this conversation"></textarea><button type="submit">Send</button></form></div>`;
    const timeline = mount.querySelector(".timeline");
    const approvalDrawer = mount.querySelector(".approval-drawer");
    const approvalDrawerSummary = mount.querySelector(".approval-drawer > summary");
    const approvalDrawerItems = mount.querySelector(".approval-drawer-items");
    const form = mount.querySelector("form");
    const input = mount.querySelector("textarea");
    const submitButton = form.querySelector("button[type=submit]");
    const attachBtn = form.querySelector(".attach-btn");
    const fileInput = form.querySelector("#mana-file-input");
    const attachmentBar = mount.querySelector(".attachment-preview-bar");
    const dot = mount.querySelector(".dot");
    const statusText = mount.querySelector(".statusText");
    const contextMeter = mount.querySelector(".contextMeter");
    const permissionBusy = new Set();
    const permissionErrors = new Map();
    const pendingAttachments = [];

    const renderPreviewBar = () => {
      if (pendingAttachments.length === 0) {
        attachmentBar.style.display = "none";
        attachmentBar.replaceChildren();
        return;
      }
      attachmentBar.style.display = "flex";
      attachmentBar.replaceChildren();
      pendingAttachments.forEach((att, index) => {
        const chip = document.createElement("div");
        chip.className = `attachment-chip ${att.uploading ? "uploading" : ""} ${att.error ? "error" : ""}`;
        const name = text(att.filename || "file");
        const size = att.size_bytes ? ` (${formatSize(att.size_bytes)})` : "";
        chip.textContent = att.uploading ? `⏳ Uploading ${name}…` : att.error ? `⚠️ ${name}: ${att.error}` : `📎 ${name}${size}`;
        if (!att.uploading) {
          const remove = document.createElement("span");
          remove.className = "remove-att";
          remove.textContent = "✕";
          remove.title = "Remove";
          remove.addEventListener("click", () => {
            pendingAttachments.splice(index, 1);
            renderPreviewBar();
          });
          chip.appendChild(remove);
        }
        attachmentBar.appendChild(chip);
      });
    };

    const uploadFile = async (file) => {
      const pendingObj = { filename: file.name, size_bytes: file.size, uploading: true };
      pendingAttachments.push(pendingObj);
      renderPreviewBar();
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch(
          `${config.apiBase}/api/v1/conversations/${encodeURIComponent(config.sessionId)}/attachments?root=${encodeURIComponent(config.root)}`,
          {
            method: "POST",
            headers: config.token ? { Authorization: `Bearer ${config.token}` } : {},
            body: formData,
          }
        );
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(payload.detail || payload.error || `HTTP ${resp.status}`);
        Object.assign(pendingObj, payload.attachment, { uploading: false });
      } catch (err) {
        pendingObj.uploading = false;
        pendingObj.error = err.message || "Failed to upload";
      }
      renderPreviewBar();
    };

    const handleFiles = (files) => {
      for (const file of files) {
        uploadFile(file);
      }
    };

    attachBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFiles(fileInput.files);
        fileInput.value = "";
      }
    });

    form.addEventListener("dragover", (e) => {
      e.preventDefault();
      form.classList.add("drop-active");
    });
    form.addEventListener("dragleave", (e) => {
      e.preventDefault();
      form.classList.remove("drop-active");
    });
    form.addEventListener("drop", (e) => {
      e.preventDefault();
      form.classList.remove("drop-active");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    });

    const addText = (parent, tag, value, className) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      node.textContent = text(value);
      parent.appendChild(node);
      return node;
    };
    let forceScrollAfterApproval = false;
    const activityBoxState = new Map();
    const activityStreamScrollState = new Map();
    const surfacedApprovalIds = new Set();
    const isActionApproval = (ev) => ["action.approval.required", "server.waiting_approval", "api.waiting_approval"].includes(ev.type);
    const isPermissionRequest = (ev) => isActionApproval(ev) || ev.type === "computer.waiting_permission";
    const approvalId = (ev) => {
      const meta = ev.metadata || {};
      return text((ev.type === "action.approval.required" && meta.inbox_item_id) || meta.permission_request_id);
    };

    const newPendingApproval = (box) => {
      let found = false;
      for (const ev of box.events) {
        if (!isActionApproval(ev)) continue;
        const id = approvalId(ev);
        const request = state.permissionRequests.get(id);
        if (!id || !request || request.status !== "pending" || surfacedApprovalIds.has(id)) continue;
        surfacedApprovalIds.add(id);
        found = true;
      }
      return found;
    };

    const rememberActivityStreamPosition = (stream, key) => {
      if (stream.dataset.scrollPositionReady !== "true") return;
      const distanceFromBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
      activityStreamScrollState.set(key, {
        scrollTop: stream.scrollTop,
        followLatest: distanceFromBottom < 40,
      });
    };

    const restoreActivityStreamPosition = (stream, key) => {
      const previous = activityStreamScrollState.get(key);
      const restore = () => {
        const maxScrollTop = Math.max(0, stream.scrollHeight - stream.clientHeight);
        const followLatest = !previous || previous.followLatest;
        stream.scrollTop = followLatest
          ? maxScrollTop
          : Math.min(previous.scrollTop, maxScrollTop);
        stream.dataset.scrollPositionReady = "true";
        rememberActivityStreamPosition(stream, key);
      };
      if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(restore);
      } else {
        Promise.resolve().then(restore);
      }
    };

    const drawerSeenIds = new Set();
    let drawerSignature = "";
    const renderApprovalDrawer = () => {
      const pending = [...state.permissionRequests.values()].filter((request) => request.status === "pending");
      const signature = JSON.stringify([config.sessionId, pending, pending.map((request) => [
        permissionBusy.has(`${config.sessionId}\u0000${request.requestId}`), permissionErrors.get(request.requestId) || "",
      ])]);
      // Preserve focus and manual drawer state during unrelated streaming updates.
      if (signature === drawerSignature) return;
      drawerSignature = signature;
      approvalDrawer.hidden = pending.length === 0;
      approvalDrawerItems.replaceChildren();
      if (!pending.length) return;
      if (pending.some((request) => !drawerSeenIds.has(request.requestId))) approvalDrawer.open = true;
      for (const request of pending) drawerSeenIds.add(request.requestId);
      approvalDrawerSummary.textContent = `Pending approvals (${pending.length})`;
      for (const request of pending) {
        const id = request.requestId;
        const busyKey = `${config.sessionId}\u0000${id}`;
        const row = document.createElement("section");
        row.className = "approval-drawer-item";
        const titles = { transactional: "Transactional action approval required", server: "Server action approval required", api: "API request approval required", computer: "Computer permission required" };
        addText(row, "div", titles[request.kind] || "Approval required");
        addText(row, "div", request.preview, "logs");
        addText(row, "div", `Scope: ${request.scope || ""}`, "meta");
        const actions = document.createElement("div");
        actions.className = "permission-actions";
        const choices = request.kind === "computer"
          ? [["Deny", "deny"], ["Allow once", "allow_once"], ["This session", "allow_session"], ["Always", "always"]]
          : [["Deny", "deny"], ["Approve once", "approve"]];
        for (const [label, decision] of choices) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = label;
          button.disabled = permissionBusy.has(busyKey);
          if (decision === "deny") button.className = "deny";
          button.addEventListener("click", async () => {
            if (permissionBusy.has(busyKey) || (state.permissionRequests.get(id) || {}).status !== "pending") return;
            const sessionId = config.sessionId;
            permissionBusy.add(busyKey);
            permissionErrors.delete(id);
            render();
            try {
              const paths = { server: "server-approvals", api: "api-approvals", transactional: "transactional-actions", computer: "computer-permissions" };
              const response = await fetch(
                `${config.apiBase}/api/v1/conversations/${encodeURIComponent(sessionId)}/${paths[request.kind]}/${encodeURIComponent(id)}`,
                { method: "POST", headers: { "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) }, body: JSON.stringify({ decision, root: config.root }) },
              );
              const payload = await response.json();
              if (!response.ok || payload.ok === false) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
              if (config.sessionId !== sessionId) return;
              if (payload.assistant_message) applyMessage(state, payload.assistant_message);
              const current = state.permissionRequests.get(id);
              if (current && (current.status === "pending" || current.status === "resuming")) {
                state.permissionRequests.set(id, { ...current, status: "decided", decision: text(payload.decision || decision) });
              }
              if (decision === "approve" && payload.approved !== false && payload.status !== "denied") forceScrollAfterApproval = true;
            } catch (error) {
              if (config.sessionId === sessionId) permissionErrors.set(id, error.message || String(error));
            } finally {
              permissionBusy.delete(busyKey);
              render();
            }
          });
          actions.appendChild(button);
        }
        row.appendChild(actions);
        if (permissionErrors.has(id)) addText(row, "div", permissionErrors.get(id), "permission-error");
        approvalDrawerItems.appendChild(row);
      }
    };

    const renderMainActivityBox = (trace) => {
      if (!trace || !trace.mainBox || trace.mainBox.events.length === 0) return null;
      const box = trace.mainBox;
      const key = `${trace.turnId}:main`;
      const node = document.createElement("details");
      node.className = "main-activity-box";
      const isRunning = box.status === "running" || (!trace.isCompleted && !trace.isFailed && !trace.isCancelled);

      const getBoxOpen = () => {
        if (newPendingApproval(box)) {
          setBoxOpen(true);
          return true;
        }
        if (activityBoxState.has(key)) {
          return activityBoxState.get(key);
        }
        for (const alias of (trace.aliases || [])) {
          const aliasKey = `${alias}:main`;
          if (activityBoxState.has(aliasKey)) {
            return activityBoxState.get(aliasKey);
          }
        }
        // First render: initialize from isRunning and persist so new activities don't close it
        const initial = isRunning || box.events.some((ev) => isActionApproval(ev) && (state.permissionRequests.get(approvalId(ev)) || {}).status === "pending");
        setBoxOpen(initial);
        return initial;
      };

      const setBoxOpen = (openVal) => {
        activityBoxState.set(key, openVal);
        for (const alias of (trace.aliases || [])) {
          activityBoxState.set(`${alias}:main`, openVal);
        }
      };

      const isOpen = getBoxOpen();
      node.open = isOpen;
      node.addEventListener("toggle", () => {
        setBoxOpen(node.open);
      });

      const summary = document.createElement("summary");
      const titleWrap = document.createElement("div");
      titleWrap.className = "activity-box-title";
      const icon = document.createElement("span");
      icon.textContent = isRunning ? "⚡" : box.status === "failed" ? "⚠️" : "✓";
      const titleText = document.createElement("span");
      const count = box.events.length;
      titleText.textContent = `Main Activity (${count} ${count === 1 ? "operation" : "operations"})`;
      titleWrap.appendChild(icon);
      titleWrap.appendChild(titleText);
      summary.appendChild(titleWrap);

      const statusBadge = document.createElement("span");
      const st = box.status || (isRunning ? "running" : "completed");
      statusBadge.className = `activity-box-badge ${st}`;
      statusBadge.textContent = st;
      summary.appendChild(statusBadge);
      node.appendChild(summary);

      const stream = document.createElement("div");
      stream.className = "activity-stream";
      stream.dataset.activityStreamKey = key;
      stream.dataset.scrollPositionReady = "false";
      stream.addEventListener("scroll", () => rememberActivityStreamPosition(stream, key));

      const showAllKey = `${key}:show_all`;
      const showAll = activityBoxState.get(showAllKey) === true;
      const maxEvents = 100;
      const allEvents = box.events.filter((ev) => {
        const request = state.permissionRequests.get(approvalId(ev));
        return !isPermissionRequest(ev) || !request || request.status === "pending";
      });
      const eventsToRender = (showAll || allEvents.length <= maxEvents) ? allEvents : allEvents.filter((ev, index) => {
        const request = state.permissionRequests.get(approvalId(ev));
        return index >= allEvents.length - maxEvents || (isActionApproval(ev) && request && request.status === "pending");
      });

      if (allEvents.length > maxEvents && !showAll) {
        const expandBtn = document.createElement("button");
        expandBtn.type = "button";
        expandBtn.className = "activity-expand-history-btn";
        expandBtn.style.cssText = "font-size:11px;color:#93c5fd;background:none;border:none;cursor:pointer;padding:4px;text-align:left;";
        expandBtn.textContent = `▲ Show ${allEvents.length - maxEvents} earlier operations...`;
        expandBtn.addEventListener("click", (e) => {
          e.preventDefault();
          activityBoxState.set(showAllKey, true);
          render();
        });
        stream.appendChild(expandBtn);
      }

      // Group consecutive operations with the same title/tool_name and status
      const groups = [];
      for (const ev of eventsToRender) {
        const lastGroup = groups[groups.length - 1];
        const evTitle = ev.tool_name || ev.title || ev.type;
        if (!isActionApproval(ev) && lastGroup && !isActionApproval(lastGroup.events[0]) && lastGroup.title === evTitle && lastGroup.status === ev.status) {
          lastGroup.events.push(ev);
        } else {
          groups.push({ title: evTitle, status: ev.status, events: [ev] });
        }
      }

      for (const grp of groups) {
        const row = document.createElement("div");
        row.className = "activity-event-row";

        const header = document.createElement("div");
        header.className = "activity-event-header";

        const titleEl = document.createElement("div");
        titleEl.className = "activity-event-title";
        const statusIcon = grp.status === "running" ? "⏳" : grp.status === "completed" || grp.status === "success" ? "✓" : grp.status === "failed" ? "✗" : "•";
        const iconSpan = document.createElement("span");
        iconSpan.textContent = statusIcon;
        const nameSpan = document.createElement("span");
        nameSpan.textContent = grp.events.length > 1 ? `${grp.title} (x${grp.events.length})` : grp.title;
        titleEl.appendChild(iconSpan);
        titleEl.appendChild(nameSpan);
        header.appendChild(titleEl);

        const latestEv = grp.events[grp.events.length - 1];
        if (latestEv.durationMs != null) {
          const durSpan = document.createElement("span");
          durSpan.className = "meta";
          durSpan.style.fontSize = "11px";
          durSpan.textContent = formatDuration(latestEv.durationMs);
          header.appendChild(durSpan);
        }
        row.appendChild(header);

        if (latestEv.detail) {
          const detailDiv = document.createElement("div");
          detailDiv.className = "meta";
          detailDiv.style.color = "#94a3b8";
          detailDiv.style.marginTop = "2px";
          detailDiv.textContent = latestEv.detail;
          row.appendChild(detailDiv);
        }

        const hasDetails = grp.events.some((e) => (e.metadata && (e.metadata.arguments || e.metadata.args || e.metadata.args_summary)) || (e.detail && e.detail.length > 80) || (e.metadata && e.metadata.error));
        if (hasDetails) {
          const det = document.createElement("details");
          det.className = "activity-event-details";
          const detKey = `${key}:det:${grp.title}:${grp.status}`;
          det.open = activityBoxState.get(detKey) === true;
          det.addEventListener("toggle", () => {
            activityBoxState.set(detKey, det.open);
          });
          addText(det, "summary", "Details");
          for (const e of grp.events) {
            const meta = e.metadata || {};
            const args = meta.arguments || meta.args || meta.args_summary;
            if (args) addText(det, "div", `Arguments: ${typeof args === "string" ? args : JSON.stringify(args)}`, "logs");
            if (e.detail && e.detail.length > 80) addText(det, "div", `Output: ${e.detail}`, "logs");
            if (meta.error) addText(det, "div", `Error: ${meta.error}`, "logs error");
          }
          row.appendChild(det);
        }

        stream.appendChild(row);
      }
      node.appendChild(stream);
      if (node.open) {
        if (stream.lastElementChild) {
          stream.lastElementChild.classList.add("activity-latest");
        }
        restoreActivityStreamPosition(stream, key);
      }
      return node;
    };

    const renderCodingActivityBox = (trace) => {
      if (!trace || !trace.codingBox) return null;
      const box = trace.codingBox;
      const hasEvents = box.events.length > 0;
      const hasCommands = box.commands.length > 0;
      const hasFiles = box.changedFiles && box.changedFiles.size > 0;
      if (!hasEvents && !hasCommands && !hasFiles) return null;

      const key = `${trace.turnId}:coding`;
      const node = document.createElement("details");
      node.className = "coding-activity-box";
      const isRunning = box.status === "running" || (!trace.isCompleted && !trace.isFailed && !trace.isCancelled);

      const getBoxOpen = () => {
        if (newPendingApproval(box)) {
          setBoxOpen(true);
          return true;
        }
        if (activityBoxState.has(key)) {
          return activityBoxState.get(key);
        }
        for (const alias of (trace.aliases || [])) {
          const aliasKey = `${alias}:coding`;
          if (activityBoxState.has(aliasKey)) {
            return activityBoxState.get(aliasKey);
          }
        }
        // First render: initialize from isRunning and persist so new activities don't close it
        const initial = isRunning || box.events.some((ev) => isActionApproval(ev) && (state.permissionRequests.get(approvalId(ev)) || {}).status === "pending");
        setBoxOpen(initial);
        return initial;
      };

      const setBoxOpen = (openVal) => {
        activityBoxState.set(key, openVal);
        for (const alias of (trace.aliases || [])) {
          activityBoxState.set(`${alias}:coding`, openVal);
        }
      };

      const isOpen = getBoxOpen();
      node.open = isOpen;
      node.addEventListener("toggle", () => {
        setBoxOpen(node.open);
      });

      const summary = document.createElement("summary");
      const titleWrap = document.createElement("div");
      titleWrap.className = "activity-box-title";
      const icon = document.createElement("span");
      icon.textContent = "💻";
      const titleText = document.createElement("span");
      const backendLabel = box.backend ? ` · ${box.backend}` : "";
      const modelLabel = box.model ? ` (${box.model})` : "";
      const totalOps = Math.max(box.events.length, box.commands.length);
      titleText.textContent = `Coding Activity${backendLabel}${modelLabel} (${totalOps} ${totalOps === 1 ? "operation" : "operations"})`;
      titleWrap.appendChild(icon);
      titleWrap.appendChild(titleText);
      summary.appendChild(titleWrap);

      const statusBadge = document.createElement("span");
      const st = box.status || (isRunning ? "running" : "completed");
      statusBadge.className = `activity-box-badge ${st}`;
      statusBadge.textContent = st;
      summary.appendChild(statusBadge);
      node.appendChild(summary);

      if (box.changedFiles && box.changedFiles.size > 0) {
        const filesContainer = document.createElement("div");
        filesContainer.className = "coding-files-chips";
        for (const f of box.changedFiles) {
          const chip = document.createElement("span");
          chip.className = "coding-file-chip";
          chip.textContent = `📄 ${f}`;
          chip.title = f;
          filesContainer.appendChild(chip);
        }
        node.appendChild(filesContainer);
      }

      if (box.events.length > 0) {
        const stream = document.createElement("div");
        stream.className = "activity-stream";
        stream.dataset.activityStreamKey = key;
        stream.dataset.scrollPositionReady = "false";
        stream.addEventListener("scroll", () => rememberActivityStreamPosition(stream, key));

        const showAllKey = `${key}:show_all`;
        const showAll = activityBoxState.get(showAllKey) === true;
        const maxEvents = 100;
        const allEvents = box.events.filter((ev) => {
          const request = state.permissionRequests.get(approvalId(ev));
          return !isPermissionRequest(ev) || !request || request.status === "pending";
        });
        const eventsToRender = (showAll || allEvents.length <= maxEvents) ? allEvents : allEvents.filter((ev, index) => {
          const request = state.permissionRequests.get(approvalId(ev));
          return index >= allEvents.length - maxEvents || (isActionApproval(ev) && request && request.status === "pending");
        });

        if (allEvents.length > maxEvents && !showAll) {
          const expandBtn = document.createElement("button");
          expandBtn.type = "button";
          expandBtn.className = "activity-expand-history-btn";
          expandBtn.style.cssText = "font-size:11px;color:#93c5fd;background:none;border:none;cursor:pointer;padding:4px;text-align:left;";
          expandBtn.textContent = `▲ Show ${allEvents.length - maxEvents} earlier operations...`;
          expandBtn.addEventListener("click", (e) => {
            e.preventDefault();
            activityBoxState.set(showAllKey, true);
            render();
          });
          stream.appendChild(expandBtn);
        }

        const groups = [];
        for (const ev of eventsToRender) {
          const lastGroup = groups[groups.length - 1];
          const evTitle = ev.tool_name || ev.title || ev.type;
          if (!isActionApproval(ev) && lastGroup && !isActionApproval(lastGroup.events[0]) && lastGroup.title === evTitle && lastGroup.status === ev.status) {
            lastGroup.events.push(ev);
          } else {
            groups.push({ title: evTitle, status: ev.status, events: [ev] });
          }
        }

        for (const grp of groups) {
          const row = document.createElement("div");
          row.className = "activity-event-row";

          const header = document.createElement("div");
          header.className = "activity-event-header";

          const titleEl = document.createElement("div");
          titleEl.className = "activity-event-title";
          const statusIcon = grp.status === "running" ? "⏳" : grp.status === "completed" || grp.status === "success" ? "✓" : grp.status === "failed" ? "✗" : "•";
          const iconSpan = document.createElement("span");
          iconSpan.textContent = statusIcon;
          const nameSpan = document.createElement("span");
          nameSpan.textContent = grp.events.length > 1 ? `${grp.title} (x${grp.events.length})` : grp.title;
          titleEl.appendChild(iconSpan);
          titleEl.appendChild(nameSpan);
          header.appendChild(titleEl);

          const latestEv = grp.events[grp.events.length - 1];
          if (latestEv.durationMs != null) {
            const durSpan = document.createElement("span");
            durSpan.className = "meta";
            durSpan.style.fontSize = "11px";
            durSpan.textContent = formatDuration(latestEv.durationMs);
            header.appendChild(durSpan);
          }
          row.appendChild(header);

          if (latestEv.detail) {
            const detailDiv = document.createElement("div");
            detailDiv.className = "meta";
            detailDiv.style.color = "#94a3b8";
            detailDiv.style.marginTop = "2px";
            detailDiv.textContent = latestEv.detail;
            row.appendChild(detailDiv);
          }

          const hasDetails = grp.events.some((e) => (e.metadata && (e.metadata.arguments || e.metadata.args || e.metadata.args_summary || e.metadata.diff || e.metadata.patch)) || (e.detail && e.detail.length > 80) || (e.metadata && e.metadata.error));
          if (hasDetails) {
            const det = document.createElement("details");
            det.className = "activity-event-details";
            const detKey = `${key}:det:${grp.title}:${grp.status}`;
            det.open = activityBoxState.get(detKey) === true;
            det.addEventListener("toggle", () => {
              activityBoxState.set(detKey, det.open);
            });
            addText(det, "summary", "Details");
            for (const e of grp.events) {
              const meta = e.metadata || {};
              const args = meta.arguments || meta.args || meta.args_summary;
              if (args) addText(det, "div", `Arguments: ${typeof args === "string" ? args : JSON.stringify(args)}`, "logs");
              if (meta.diff || meta.patch) addText(det, "pre", meta.diff || meta.patch, "logs");
              if (e.detail && e.detail.length > 80) addText(det, "div", `Output: ${e.detail}`, "logs");
              if (meta.error) addText(det, "div", `Error: ${meta.error}`, "logs error");
            }
            row.appendChild(det);
          }

          stream.appendChild(row);
        }
        node.appendChild(stream);
        if (node.open) {
          if (stream.lastElementChild) {
            stream.lastElementChild.classList.add("activity-latest");
          }
          restoreActivityStreamPosition(stream, key);
        }
      }

      if (box.commands && box.commands.length > 0) {
        const pre = document.createElement("pre");
        pre.className = "coding-logs-pre";
        pre.textContent = box.commands.map((cmd) => `$ ${cmd}`).join("\n");
        node.appendChild(pre);
      }

      if (!isRunning && (hasEvents || hasFiles || hasCommands)) {
        const summaryCard = document.createElement("div");
        summaryCard.className = "coding-summary-card";
        summaryCard.style.cssText = "margin-top:6px;padding:6px 10px;background:#0d1d2e;border-left:3px solid #22c55e;border-radius:4px;font-size:12px;color:#cbd5e1;";
        const filesCount = box.changedFiles ? box.changedFiles.size : 0;
        const opsCount = Math.max(box.events.length, box.commands.length);
        const filesText = filesCount > 0 ? `${filesCount} ${filesCount === 1 ? "file" : "files"} changed` : "Code inspected";
        summaryCard.textContent = `✓ Coding complete · ${filesText} (${opsCount} operations)`;
        node.appendChild(summaryCard);
      }

      return node;
    };

    const render = () => {
      renderApprovalDrawer();
      const nearBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 80;
      for (const stream of timeline.querySelectorAll(".activity-stream[data-activity-stream-key]")) {
        rememberActivityStreamPosition(stream, stream.dataset.activityStreamKey);
      }
      timeline.replaceChildren();
      const rows = [];
      for (const message of state.messages.values()) rows.push({ kind: "message", time: message.created_at || "", sequence: 0, value: message });
      for (const activity of state.activities.values()) {
        const type = eventType(activity);
        if (type.startsWith("media_generation_") || type === "canvas.createSurface") {
          rows.push({ kind: "activity", time: eventTime(activity), sequence: activity.sequence || 0, value: activity });
        }
      }
      rows.sort((a, b) => a.time.localeCompare(b.time) || a.sequence - b.sequence);

      const renderedTraces = new Set();
      for (const row of rows) {
        if (row.kind === "message") {
          const item = row.value;
          const node = document.createElement("div");
          node.className = `message ${item.role === "user" ? "user" : "assistant"} ${item.status === "failed" ? "failed" : ""} ${item.optimistic ? "sending" : ""}`;
          addText(node, "div", item.content || (item.status === "streaming" ? "…" : ""));
          if (item.attachments && item.attachments.length > 0) {
            const attContainer = document.createElement("div");
            attContainer.className = "message-attachments";
            for (const att of item.attachments) {
              const chip = document.createElement("div");
              chip.className = "message-attachment-card";
              const fname = text(att.filename || "file");
              const sz = formatSize(att.size_bytes || 0);
              const cat = text(att.category || "");
              chip.textContent = `📎 ${fname} (${cat ? `${cat}, ` : ""}${sz})`;
              chip.title = `${fname} (${att.mime_type || ""})`;
              attContainer.appendChild(chip);
            }
            node.appendChild(attContainer);
          }
          const meta = item.error ? `${item.status} · ${item.error}` : item.optimistic ? "sending…" : item.status === "streaming" ? "streaming…" : "";
          if (meta) addText(node, "div", meta, `meta ${item.error ? "error" : ""}`);
          timeline.appendChild(node);

          if (item.role === "user") {
            const trace = resolveTrace(state, item.execution_id || item.message_id || item.id);
            if (trace) {
              renderedTraces.add(trace);
              const mb = renderMainActivityBox(trace);
              if (mb) timeline.appendChild(mb);
              const cb = renderCodingActivityBox(trace);
              if (cb) timeline.appendChild(cb);
            }
          }
        } else if (row.kind === "activity") {
          const event = row.value;
          const node = document.createElement("div");
          const type = eventType(event);
          const metadata = metaOf(event);
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
          timeline.appendChild(node);
        }
      }

      for (const trace of state.executionTraces.values()) {
        if (!renderedTraces.has(trace)) {
          renderedTraces.add(trace);
          const mb = renderMainActivityBox(trace);
          if (mb) timeline.appendChild(mb);
          const cb = renderCodingActivityBox(trace);
          if (cb) timeline.appendChild(cb);
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
      if (nearBottom || forceScrollAfterApproval) {
        forceScrollAfterApproval = false;
        timeline.scrollTop = timeline.scrollHeight;
        if (typeof requestAnimationFrame === "function") {
          requestAnimationFrame(() => {
            timeline.scrollTop = timeline.scrollHeight;
          });
        }
      }
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
      surfacedApprovalIds.clear();
      drawerSeenIds.clear();
      permissionErrors.clear();
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
      const validAttachments = pendingAttachments.filter((a) => !a.uploading && !a.error);
      if ((!content && validAttachments.length === 0) || state.submitting || !state.socketReady) return;
      if (pendingAttachments.some((a) => a.uploading)) return;
      const id = `client_${Date.now().toString(36)}_${crypto.randomUUID().replaceAll("-", "")}`;
      const attsToSend = validAttachments.map((a) => ({ ...a }));
      reduce(state, {
        type: "optimistic",
        message: {
          message_id: id,
          content,
          attachments: attsToSend,
          created_at: new Date().toISOString(),
        }
      });
      input.value = "";
      pendingAttachments.length = 0;
      renderPreviewBar();
      render();
      try {
        const response = await fetch(`${config.apiBase}/api/v1/conversations/${encodeURIComponent(config.sessionId)}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) },
          body: JSON.stringify({ content, attachments: attsToSend, client_message_id: id, root: config.root }),
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

  return { createState, reduce, snapshot, init, classifyRuntimePhase, isCodingTool };
});

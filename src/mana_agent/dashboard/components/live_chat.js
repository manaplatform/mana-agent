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
    const type = eventType(event);
    const metadata = metaOf(event);
    const permissionRequestId = text(metadata.permission_request_id);
    if (type === "computer.waiting_permission" && permissionRequestId) {
      state.permissionRequests.set(permissionRequestId, {
        requestId: permissionRequestId,
        scope: text(metadata.permission_scope),
        preview: text(metadata.preview || event.title || eventSummary(event)),
        status: "pending",
        decision: "",
      });
    } else if (type === "computer.permission_decided" && permissionRequestId) {
      const previous = state.permissionRequests.get(permissionRequestId) || {
        requestId: permissionRequestId,
      };
      state.permissionRequests.set(permissionRequestId, {
        ...previous,
        status: "decided",
        decision: text(metadata.decision),
      });
    }
    if (type === "message.accepted" || type === "turn.started" && metaOf(event).role === "user") {
      applyAcceptedMessage(state, event);
    } else if (type.startsWith("tool.") || type.startsWith("command.")) {
      applyTool(state, event);
    } else if (type === "assistant.started" || type === "assistant.delta" || type === "turn.finished") {
      applyAssistant(state, event);
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
        @media(max-width:520px){.message{max-width:96%}.shell{border-radius:7px}.timeline{padding:8px}}
      </style>
      <div class="shell"><div class="status"><span class="dot"></span><span class="statusText">Connecting to live events…</span></div>
      <div class="timeline"></div><form><textarea aria-label="Chat message" placeholder="Message this conversation"></textarea><button type="submit">Send</button></form></div>`;
    const timeline = mount.querySelector(".timeline");
    const form = mount.querySelector("form");
    const input = mount.querySelector("textarea");
    const submitButton = form.querySelector("button");
    const dot = mount.querySelector(".dot");
    const statusText = mount.querySelector(".statusText");
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
        } else if (row.kind === "tool") {
          const tool = row.value;
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
          if (type === "computer.waiting_permission" && permissionRequestId) {
            node.className = "activity permission";
            const request = state.permissionRequests.get(permissionRequestId) || {};
            addText(node, "div", "Computer permission required");
            addText(node, "div", request.preview || metadata.preview || event.title, "logs");
            addText(node, "div", `Scope: ${request.scope || metadata.permission_scope || ""}`, "meta");
            if (request.status === "decided") {
              addText(node, "div", `Decision: ${request.decision || "completed"}`, "permission-state");
            } else {
              const actions = document.createElement("div");
              actions.className = "permission-actions";
              const choices = [
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
                decisionButton.disabled = permissionBusy.has(permissionRequestId);
                decisionButton.addEventListener("click", async () => {
                  permissionBusy.add(permissionRequestId);
                  permissionErrors.delete(permissionRequestId);
                  render();
                  try {
                    const response = await fetch(
                      `${config.apiBase}/api/v1/conversations/${encodeURIComponent(config.sessionId)}/computer-permissions/${encodeURIComponent(permissionRequestId)}`,
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) },
                        body: JSON.stringify({ decision, root: config.root }),
                      },
                    );
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
                    state.permissionRequests.set(permissionRequestId, {
                      ...request,
                      requestId: permissionRequestId,
                      status: "decided",
                      decision,
                    });
                  } catch (error) {
                    permissionErrors.set(permissionRequestId, error.message || String(error));
                  } finally {
                    permissionBusy.delete(permissionRequestId);
                    render();
                  }
                });
                actions.appendChild(decisionButton);
              }
              node.appendChild(actions);
              if (permissionErrors.has(permissionRequestId)) {
                addText(node, "div", permissionErrors.get(permissionRequestId), "permission-error");
              }
            }
          } else {
            node.className = "activity";
            addText(node, "div", `${event.title || type} · ${eventStatus(event)}`);
            if (eventSummary(event)) addText(node, "div", eventSummary(event), "logs");
          }
          timeline.appendChild(node);
        }
      }
      submitButton.disabled = !state.socketReady || state.submitting;
      dot.classList.toggle("ok", state.socketReady);
      statusText.textContent = state.socketReady
        ? state.submitting || state.runStatus === "running" || state.runStatus === "starting" ? "Agent is working · live" : "Live events connected"
        : "Reconnecting to live events…";
      if (nearBottom) timeline.scrollTop = timeline.scrollHeight;
    };

    const dispatchEvent = (event) => { reduce(state, { type: "event", event }); render(); };
    const socketUrl = () => `${config.wsBase}/api/v1/ws/conversations/${encodeURIComponent(config.sessionId)}?root=${encodeURIComponent(config.root)}&replay_limit=1000&after_sequence=${state.lastSequence}`;
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
      if ([...state.tools.values()].some((tool) => tool.status === "running")) render();
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

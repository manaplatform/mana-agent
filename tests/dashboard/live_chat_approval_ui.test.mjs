import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const { init } = createRequire(import.meta.url)("../../src/mana_agent/dashboard/components/live_chat.js");

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.classList = { add() {}, toggle() {} };
    this.handlers = {};
    this.scrollHeight = 1000;
    this.clientHeight = 100;
    this.scrollTop = 0;
    this.hidden = false;
  }
  appendChild(child) { if (child.parentNode) child.remove(); this.children.push(child); child.parentNode = this; return child; }
  insertBefore(child, reference) { if (child === reference) return child; if (child.parentNode) child.remove(); const index = reference ? this.children.indexOf(reference) : this.children.length; this.children.splice(index, 0, child); child.parentNode = this; return child; }
  remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter((node) => node !== this); this.parentNode = null; } }
  get attributes() { return Object.entries(this.dataset).map(([key, value]) => ({ name: `data-${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`, value })); }
  setAttribute(name, value) { if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = value; else this[name] = value; }
  querySelector(selector) { return this.children.find((child) => selector.startsWith(".") && child.className.split(" ").includes(selector.slice(1))) || null; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  get lastElementChild() { return this.children.at(-1); }
  querySelectorAll() { return []; }
}

function fixture(events, response) {
  const selectors = new Map();
  for (const selector of [".timeline", ".permission-notice", ".approval-drawer", ".approval-drawer > summary", ".approval-drawer-items", "form", "textarea", ".attachment-preview-bar", ".dot", ".statusText", ".contextMeter", "button[type=submit]", ".attach-btn", "#mana-file-input"]) {
    selectors.set(selector, new Element());
  }
  const mount = new Element();
  mount.querySelector = (selector) => selectors.get(selector);
  selectors.get("form").querySelector = (selector) => selectors.get(selector);
  globalThis.document = { getElementById: () => mount, createElement: (tag) => new Element(tag), addEventListener() {}, removeEventListener() {} };
  globalThis.window = { addEventListener() {} };
  let socket;
  globalThis.WebSocket = class { constructor() { socket = this; } close() {} };
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return response; };
  const chat = init({ mountId: "chat", sessionId: "session", apiBase: "", wsBase: "", root: "/repo", messages: [], events });
  return { chat, timeline: selectors.get(".timeline"), notice: selectors.get(".permission-notice"), drawer: selectors.get(".approval-drawer"), items: selectors.get(".approval-drawer-items"), calls,
    emit: (event) => socket.onmessage({ data: JSON.stringify({ type: "event", event }) }) };
}

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
const approval = (id, sequence, backend = "") => ({
  event_id: `event-${id}`, sequence, type: "action.approval.required", execution_id: "run",
  status: "waiting_approval", title: "Action approval required", backend,
  metadata: { inbox_item_id: id, permission_scope: "transactional_action.once", preview: `Preview ${id}` },
});

test("computer permissions retain their scopes in the drawer and disappear after a decision", async () => {
  const event = { event_id: "computer", sequence: 1, type: "computer.waiting_permission", execution_id: "run",
    status: "waiting_approval", metadata: { permission_request_id: "pc-1", permission_scope: "screen", preview: "Capture screen" } };
  const { chat, timeline, drawer, items, calls } = fixture([event], { ok: true, json: async () => ({ ok: true, decision: "allow_session" }) });
  try {
    assert.deepEqual(descendants(items).filter((n) => n.tagName === "button").map((n) => n.textContent), ["Deny", "Allow once", "This session", "Always"]);
    assert.ok(!descendants(timeline).some((n) => n.className === "activity permission"));
    await descendants(items).find((n) => n.textContent === "This session").handlers.click();
    assert.match(calls[0].url, /computer-permissions\/pc-1$/);
    assert.equal(JSON.parse(calls[0].options.body).decision, "allow_session");
    assert.equal(drawer.hidden, true);
    assert.ok(!descendants(items).some((n) => n.textContent === "Capture screen"));
  } finally { chat.close(); }
});

test("computer approval does not hide a separate exact-action confirmation requirement", async () => {
  const event = { event_id: "computer", sequence: 1, type: "computer.waiting_permission", execution_id: "run",
    metadata: { permission_request_id: "pc-1", preview: "Permission" } };
  const { chat, notice, drawer, items } = fixture([event], { ok: true, json: async () => ({ ok: true, executed: false,
    result: { error_code: "confirmation_required", message: "Confirm the action", preview: "Exact action", confirmation_request_id: "confirm-1" } }) });
  try {
    await descendants(items).find((n) => n.textContent === "Allow once").handlers.click();
    assert.equal(drawer.hidden, true);
    assert.equal(notice.hidden, false);
    assert.match(notice.textContent, /Confirm the action.*\nExact action\nConfirmation request: confirm-1/);
  } finally { chat.close(); }
});

test("external decisions and replayed request events do not reopen a request", () => {
  const { chat, drawer, items, emit } = fixture([approval("one", 1), approval("two", 2)], { ok: true, json: async () => ({}) });
  try {
    emit({ event_id: "decision", sequence: 3, type: "action.approval.denied", metadata: { inbox_item_id: "one", decision: "deny" } });
    emit({ ...approval("one", 4), event_id: "repeat" });
    assert.equal(chat.state.permissionRequests.get("one").status, "decided");
    assert.ok(!descendants(items).some((n) => n.textContent === "Preview one"));
    assert.ok(descendants(items).some((n) => n.textContent === "Preview two"));
    emit({ event_id: "decision-two", sequence: 5, type: "action.approval.granted", metadata: { inbox_item_id: "two" } });
    assert.equal(drawer.hidden, true);
  } finally { chat.close(); }
});

test("a pending decision cannot be submitted twice and a failed request can be retried", async () => {
  let resolveFetch;
  const { chat, drawer, items, calls } = fixture([approval("one", 1)], { ok: true, json: async () => ({}) });
  globalThis.fetch = (url, options) => { calls.push({ url, options }); return new Promise((resolve) => { resolveFetch = resolve; }); };
  try {
    const button = descendants(items).find((n) => n.textContent === "Approve once");
    const first = button.handlers.click();
    await button.handlers.click();
    assert.equal(calls.length, 1);
    resolveFetch({ ok: false, status: 403, json: async () => ({ detail: "Not allowed" }) });
    await first;
    assert.equal(chat.state.permissionRequests.get("one").status, "pending");
    assert.equal(descendants(items).find((n) => n.textContent === "Approve once").disabled, false);
  } finally { chat.close(); }
});

test("unrelated events preserve drawer controls and a new request adds a second drawer item", () => {
  const { chat, drawer, items, emit } = fixture([approval("one", 1)], { ok: true, json: async () => ({}) });
  try {
    const button = descendants(items).find((n) => n.textContent === "Approve once");
    emit({ event_id: "log", sequence: 2, type: "log.info", status: "running", execution_id: "run" });
    assert.equal(descendants(items).find((n) => n.textContent === "Approve once"), button);
    emit(approval("two", 3));
    assert.equal(descendants(items).filter((n) => n.textContent === "Approve once").length, 2);
  } finally { chat.close(); }
});

test("transactional approvals share the drawer and resolve independently", async () => {
  const { chat, timeline, drawer, items, calls } = fixture([approval("one", 1), approval("two", 2, "internal")], {
    ok: true, json: async () => ({ approved: true, status: "completed", assistant_message: { message_id: "reply", role: "assistant", content: "Done" } }),
  });
  try {
    let nodes = descendants(items);
    assert.equal(nodes.filter((n) => n.className === "activity permission").length, 0);
    assert.equal(nodes.filter((n) => n.textContent === "Approve once").length, 2);
    assert.equal(drawer.hidden, false);
    assert.ok(descendants(timeline).some((n) => n.className === "main-activity-box"));
    assert.ok(descendants(timeline).some((n) => n.className === "coding-activity-box"));
    assert.equal(descendants(timeline).filter((n) => n.textContent === "Approve once").length, 0);
    assert.equal(calls.length, 0); // Rendering does not grant approval.
    timeline.scrollTop = 0;
    await nodes.find((n) => n.textContent === "Approve once").handlers.click();
    assert.match(calls[0].url, /transactional-actions\/one$/);
    assert.equal(JSON.parse(calls[0].options.body).decision, "approve");
    assert.equal(chat.state.permissionRequests.get("two").status, "pending");
    assert.equal(timeline.scrollTop, timeline.scrollHeight);
    nodes = descendants(items);
    assert.equal(nodes.filter((n) => n.textContent === "Approve once").length, 1);
    assert.ok(!nodes.some((n) => n.textContent === "Preview one"));
    assert.ok(descendants(timeline).some((n) => n.textContent === "Done"));
  } finally { chat.close(); }
});

test("failed approval remains pending and does not force scrolling", async () => {
  const { chat, timeline, drawer, items } = fixture([approval("one", 1)], { ok: false, status: 403, json: async () => ({ detail: "Not allowed" }) });
  try {
    timeline.scrollTop = 0;
    await descendants(items).find((n) => n.textContent === "Approve once").handlers.click();
    assert.equal(chat.state.permissionRequests.get("one").status, "pending");
    assert.equal(timeline.scrollTop, 0);
    assert.ok(descendants(items).some((n) => n.textContent === "Not allowed"));
    assert.ok(descendants(items).some((n) => n.textContent === "Approve once"));
  } finally { chat.close(); }
});

test("pending approval remains accessible after activity history is truncated", () => {
  const later = Array.from({ length: 110 }, (_, i) => ({ event_id: `later-${i}`, sequence: i + 2, type: "log.info", execution_id: "run", status: "running" }));
  const { chat, items } = fixture([approval("old", 1), ...later], { ok: true, json: async () => ({}) });
  try {
    assert.ok(descendants(items).some((n) => n.textContent === "Approve once"));
  } finally { chat.close(); }
});

test("a pending approval is not repeated as an activity row", () => {
  const { chat, timeline, items } = fixture([approval("one", 1)], { ok: true, json: async () => ({}) });
  try {
    // The request has its own drawer row, so the activity box must not show a
    // duplicate approval row.
    assert.ok(descendants(items).some((n) => n.textContent === "Preview one"));
    assert.equal(descendants(timeline).filter((n) => n.textContent === "Preview one").length, 0);
  } finally { chat.close(); }
});

for (const [type, endpoint] of [["server.waiting_approval", "server-approvals"], ["api.waiting_approval", "api-approvals"]]) {
  test(`${type} is decided in the drawer through its own endpoint`, async () => {
    const event = { event_id: type, sequence: 1, type, execution_id: "run", status: "waiting_approval", title: type,
      metadata: { permission_request_id: "req-1", preview: "Review action" } };
    const { chat, timeline, drawer, items, calls } = fixture([event], { ok: true, json: async () => ({ status: "denied" }) });
    try {
      assert.equal(descendants(timeline).filter((n) => n.className === "activity permission").length, 0);
      await descendants(items).find((n) => n.textContent === "Deny").handlers.click();
      assert.match(calls[0].url, new RegExp(`/${endpoint}/req-1$`));
      assert.equal(JSON.parse(calls[0].options.body).decision, "deny");
      assert.equal(chat.state.permissionRequests.get("req-1").status, "decided");
      assert.equal(drawer.hidden, true);
    } finally { chat.close(); }
  });
}

test("expired approval request is not shown in drawer", () => {
  const expiredEvent = {
    event_id: "expired-appr", sequence: 1, type: "action.approval.required", execution_id: "run",
    status: "waiting_approval", title: "Action approval required",
    metadata: {
      inbox_item_id: "exp-item-1", permission_scope: "transactional_action.once", preview: "Preview exp-1",
      expires_at: new Date(Date.now() - 30000).toISOString(),
    },
  };
  const { chat, drawer, items } = fixture([expiredEvent], { ok: true, json: async () => ({}) });
  try {
    assert.equal(drawer.hidden, true);
    assert.equal(descendants(items).filter((n) => n.textContent === "Approve once").length, 0);
  } finally { chat.close(); }
});

test("inbox.item.expired removes drawer item immediately", () => {
  const { chat, drawer, items, emit } = fixture([approval("live-1", 1)], { ok: true, json: async () => ({}) });
  try {
    assert.equal(drawer.hidden, false);
    assert.equal(descendants(items).filter((n) => n.textContent === "Approve once").length, 1);
    emit({
      event_id: "exp-event-1", sequence: 2, type: "inbox.item.expired", execution_id: "run",
      status: "expired", metadata: { inbox_item_id: "live-1" },
    });
    assert.equal(drawer.hidden, true);
    assert.equal(descendants(items).filter((n) => n.textContent === "Approve once").length, 0);
  } finally { chat.close(); }
});

test("approval action returning terminal or expired error dismisses drawer gracefully without stuck error", async () => {
  const { chat, drawer, items } = fixture([approval("term-1", 1)], {
    ok: false,
    status: 409,
    json: async () => ({ ok: false, detail: "inbox item is already terminal: expired" }),
  });
  try {
    assert.equal(drawer.hidden, false);
    const approveBtn = descendants(items).find((n) => n.textContent === "Approve once");
    assert.ok(approveBtn);
    await approveBtn.handlers.click();
    assert.equal(drawer.hidden, true);
    assert.equal(chat.state.permissionRequests.get("term-1").status, "expired");
  } finally { chat.close(); }
});


test("main and internal coding activity render without separate archive controls", () => {
  const events = [
    { event_id: "planning", sequence: 1, type: "agent.planning", execution_id: "run", status: "running",
      title: "Reviewing requirements", agent_id: "MainAgent", operation_id: "planning" },
    { event_id: "read", sequence: 2, type: "tool.call.started", execution_id: "run", status: "running",
      title: "Reading implementation", backend: "internal", agent_id: "CodingAgent", operation_id: "read",
      metadata: { tool_name: "read_file" } },
  ];
  const { chat, timeline, calls } = fixture(events, { ok: true, json: async () => ({ events: [] }) });
  try {
    const nodes = descendants(timeline);
    assert.equal(nodes.filter((node) => node.className === "main-activity-box").length, 1);
    assert.equal(nodes.filter((node) => node.className === "coding-activity-box").length, 1);
    assert.ok(nodes.some((node) => String(node.textContent || "").includes("Reviewing requirements")));
    assert.ok(nodes.some((node) => String(node.textContent || "").includes("Reading implementation")));
    assert.ok(!nodes.some((node) => /earlier activity/i.test(String(node.textContent || ""))));
    assert.equal(calls.length, 0);
  } finally { chat.close(); }
});

test("coding operation DOM and expanded details survive streaming updates", async () => {
  const owner = { task_id: "task", agent_id: "worker", parent_agent_id: "coding", execution_id: "run", operation_id: "operation", operation_attempt_id: "invocation", backend: "internal" };
  const initial = { ...owner, type: "tool.started", event_id: "start", sequence: 1, status: "running", metadata: { command: "python streams.py", cwd: "/workspace" } };
  const { chat, timeline, emit } = fixture([initial], { ok: true, json: async () => ({}) });
  try {
    const box = descendants(timeline).find((node) => node.className === "coding-activity-box");
    const row = descendants(box).find((node) => node.className === "activity-event-row");
    const details = descendants(row).find((node) => node.className === "activity-event-details");
    details.open = true;
    emit({ ...owner, type: "tool.stdout", event_id: "chunk", sequence: 2, status: "running", output_preview: "progress\n", metadata: { stream_channel: "stdout", stream_offset: 0, content_length: 9 } });
    await new Promise((resolve) => setTimeout(resolve, 150));
    const updatedBox = descendants(timeline).find((node) => node.className === "coding-activity-box");
    assert.equal(updatedBox, box);
    assert.equal(descendants(updatedBox).find((node) => node.className === "activity-event-row"), row);
    assert.equal(details.open, true);
    assert.ok(descendants(updatedBox).some((node) => String(node.textContent || "").includes("progress")));
  } finally { chat.close(); }
});

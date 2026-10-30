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
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  get lastElementChild() { return this.children.at(-1); }
  querySelectorAll() { return []; }
}

function fixture(events, response) {
  const selectors = new Map();
  for (const selector of [".timeline", "form", "textarea", ".attachment-preview-bar", ".dot", ".statusText", ".contextMeter", "button[type=submit]", ".attach-btn", "#mana-file-input"]) {
    selectors.set(selector, new Element());
  }
  const mount = new Element();
  mount.querySelector = (selector) => selectors.get(selector);
  selectors.get("form").querySelector = (selector) => selectors.get(selector);
  globalThis.document = { getElementById: () => mount, createElement: (tag) => new Element(tag) };
  globalThis.window = { addEventListener() {} };
  globalThis.WebSocket = class { close() {} };
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return response; };
  const chat = init({ mountId: "chat", sessionId: "session", apiBase: "", wsBase: "", root: "/repo", messages: [], events });
  return { chat, timeline: selectors.get(".timeline"), calls };
}

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
const approval = (id, sequence, backend = "") => ({
  event_id: `event-${id}`, sequence, type: "action.approval.required", execution_id: "run",
  status: "waiting_approval", title: "Action approval required", backend,
  metadata: { inbox_item_id: id, permission_scope: "transactional_action.once", preview: `Preview ${id}` },
});

test("transactional approvals are actionable only within their activity box", async () => {
  const { chat, timeline, calls } = fixture([approval("one", 1), approval("two", 2, "internal")], {
    ok: true, json: async () => ({ approved: true, status: "completed", assistant_message: { message_id: "reply", role: "assistant", content: "Done" } }),
  });
  try {
    let nodes = descendants(timeline);
    assert.equal(nodes.filter((n) => n.className === "activity permission").length, 0);
    assert.equal(nodes.filter((n) => n.textContent === "Approve once").length, 2);
    assert.ok(nodes.some((n) => n.className === "main-activity-box"));
    assert.ok(nodes.some((n) => n.className === "coding-activity-box"));
    assert.equal(calls.length, 0); // Rendering does not grant approval.
    timeline.scrollTop = 0;
    await nodes.find((n) => n.textContent === "Approve once").handlers.click();
    assert.match(calls[0].url, /transactional-actions\/one$/);
    assert.equal(JSON.parse(calls[0].options.body).decision, "approve");
    assert.equal(chat.state.permissionRequests.get("two").status, "pending");
    assert.equal(timeline.scrollTop, timeline.scrollHeight);
    nodes = descendants(timeline);
    assert.ok(nodes.some((n) => n.textContent === "Done"));
  } finally { chat.close(); }
});

test("failed approval remains pending and does not force scrolling", async () => {
  const { chat, timeline } = fixture([approval("one", 1)], { ok: false, status: 403, json: async () => ({ detail: "Not allowed" }) });
  try {
    timeline.scrollTop = 0;
    await descendants(timeline).find((n) => n.textContent === "Approve once").handlers.click();
    assert.equal(chat.state.permissionRequests.get("one").status, "pending");
    assert.equal(timeline.scrollTop, 0);
    assert.ok(descendants(timeline).some((n) => n.textContent === "Not allowed"));
  } finally { chat.close(); }
});

test("pending approval remains accessible after activity history is truncated", () => {
  const later = Array.from({ length: 110 }, (_, i) => ({ event_id: `later-${i}`, sequence: i + 2, type: "log.info", execution_id: "run", status: "running" }));
  const { chat, timeline } = fixture([approval("old", 1), ...later], { ok: true, json: async () => ({}) });
  try {
    assert.ok(descendants(timeline).some((n) => n.textContent === "Approve once"));
  } finally { chat.close(); }
});

for (const [type, endpoint] of [["server.waiting_approval", "server-approvals"], ["api.waiting_approval", "api-approvals"]]) {
  test(`${type} is decided in Activity through its own endpoint`, async () => {
    const event = { event_id: type, sequence: 1, type, execution_id: "run", status: "waiting_approval", title: type,
      metadata: { permission_request_id: "req-1", preview: "Review action" } };
    const { chat, timeline, calls } = fixture([event], { ok: true, json: async () => ({ status: "denied" }) });
    try {
      assert.equal(descendants(timeline).filter((n) => n.className === "activity permission").length, 0);
      await descendants(timeline).find((n) => n.textContent === "Deny").handlers.click();
      assert.match(calls[0].url, new RegExp(`/${endpoint}/req-1$`));
      assert.equal(JSON.parse(calls[0].options.body).decision, "deny");
      assert.equal(chat.state.permissionRequests.get("req-1").status, "denied");
    } finally { chat.close(); }
  });
}

import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const canvas = require("../../src/mana_agent/dashboard/components/live_canvas.js");

test("snapshot and ordered incremental updates converge", () => {
  const state = canvas.createState("session_one");
  canvas.applySnapshot(state, {
    session_id: "session_one", conversation_id: "session_one", surface_id: "plan",
    last_sequence: 1, version: 1, components: [], data_model: {}, owner: { agent_id: "main" },
  });
  canvas.applyCanvasEvent(state, {
    session_id: "session_one", conversation_id: "session_one", surface_id: "plan",
    sequence: 2, event_type: "updateComponents", timestamp: "2026-01-01T00:00:00Z",
    payload: { updateComponents: { components: [{ id: "root", component: "Text", text: "Ready" }] } },
  });
  canvas.applyCanvasEvent(state, {
    session_id: "session_one", conversation_id: "session_one", surface_id: "plan",
    sequence: 3, event_type: "updateDataModel", timestamp: "2026-01-01T00:00:01Z",
    payload: { updateDataModel: { path: "/status", value: "done" } },
  });
  assert.equal(state.surfaces.get("plan").components[0].text, "Ready");
  assert.equal(state.surfaces.get("plan").data_model.status, "done");
});

test("sequence gaps are explicit and cross-session events are ignored", () => {
  const state = canvas.createState("session_one");
  canvas.applySnapshot(state, { session_id: "session_one", surface_id: "plan", last_sequence: 1 });
  canvas.applyCanvasEvent(state, { session_id: "other", surface_id: "plan", sequence: 2 });
  assert.equal(state.surfaces.get("plan").last_sequence, 1);
  canvas.applyCanvasEvent(state, {
    session_id: "session_one", surface_id: "plan", sequence: 3,
    event_type: "updateDataModel", payload: { updateDataModel: { value: {} } },
  });
  assert.match(state.error, /sequence gap/);
});

test("bindings and JSON pointer updates are deterministic", () => {
  const model = canvas.updatePath({ project: { priority: "low" } }, "/project/priority", "high");
  assert.equal(canvas.resolve({ path: "/project/priority" }, model, {}), "high");
});

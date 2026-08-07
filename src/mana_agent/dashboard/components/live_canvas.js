(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.ManaLiveCanvas = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const text = (value) => String(value == null ? "" : value);
  const clone = (value) => JSON.parse(JSON.stringify(value == null ? {} : value));
  const canvasEvent = (event) => (event.metadata || event.details || {}).canvas_event || null;

  function safeResourceUrl(value) {
    try {
      const url = new URL(text(value), document.baseURI);
      if (url.protocol === "https:") return url.href;
      if (url.protocol === "http:" && ["localhost", "127.0.0.1"].includes(url.hostname)) return url.href;
    } catch (_error) { /* Invalid URLs are blocked below. */ }
    return "";
  }

  function createState(sessionId) {
    return {
      sessionId: text(sessionId), surfaces: new Map(), selected: "", lastSequence: 0,
      socketReady: false, loading: true, error: "", mode: "surface", activities: [],
      pendingActions: new Set(), actionErrors: new Map(), fullscreen: false,
    };
  }

  function applySnapshot(state, snapshot) {
    if (!snapshot || !snapshot.surface_id) return state;
    const prior = state.surfaces.get(snapshot.surface_id) || {};
    state.surfaces.set(snapshot.surface_id, {
      ...prior, ...clone(snapshot), components: clone(snapshot.components || []),
      data_model: clone(snapshot.data_model || {}),
    });
    if (!state.selected || state.selected === snapshot.surface_id) state.selected = snapshot.surface_id;
    return state;
  }

  function applyCanvasEvent(state, envelope) {
    if (!envelope || envelope.session_id !== state.sessionId) return state;
    const id = envelope.surface_id;
    const current = state.surfaces.get(id);
    const payload = envelope.payload || {};
    if (envelope.event_type === "createSurface") {
      const body = payload.createSurface || {};
      applySnapshot(state, {
        session_id: envelope.session_id, conversation_id: envelope.conversation_id,
        surface_id: id, catalog_id: body.catalogId, owner: {
          agent_id: envelope.agent_id, task_id: envelope.task_id,
          workflow_id: envelope.workflow_id, node_id: envelope.node_id,
          automation_id: envelope.automation_id,
        }, version: 1,
        last_sequence: envelope.sequence, components: [], data_model: {}, deleted: false,
        completed: false, retain_on_complete: envelope.retain_on_complete !== false,
        created_at: envelope.timestamp, updated_at: envelope.timestamp,
      });
    } else if (!current) {
      state.error = `Update received for unknown surface ${id}.`;
      return state;
    } else if (envelope.sequence <= Number(current.last_sequence || 0)) {
      return state;
    } else if (envelope.sequence !== Number(current.last_sequence || 0) + 1) {
      state.error = `Surface ${id} needs recovery (event sequence gap).`;
      return state;
    } else if (envelope.event_type === "updateComponents") {
      const merged = new Map((current.components || []).map((item) => [item.id, item]));
      for (const item of (payload.updateComponents || {}).components || []) merged.set(item.id, clone(item));
      current.components = [...merged.values()];
    } else if (envelope.event_type === "updateDataModel") {
      const body = payload.updateDataModel || {};
      current.data_model = updatePath(current.data_model || {}, body.path || "/", body.value);
    } else if (envelope.event_type === "deleteSurface") current.deleted = true;
    else if (envelope.event_type === "streamComplete") current.completed = true;
    if (current) {
      current.last_sequence = envelope.sequence;
      current.version = Number(current.version || 0) + 1;
      current.updated_at = envelope.timestamp;
    }
    return state;
  }

  function applyHubEvent(state, event) {
    if (!event || typeof event !== "object") return state;
    const sequence = Number(event.sequence || 0);
    if (sequence) state.lastSequence = Math.max(state.lastSequence, sequence);
    const envelope = canvasEvent(event);
    if (envelope) applyCanvasEvent(state, envelope);
    const type = text(event.type);
    if (!type.startsWith("assistant.delta") && !type.startsWith("message.")) {
      state.activities.push(clone(event));
      if (state.activities.length > 250) state.activities.shift();
    }
    return state;
  }

  function isUnsafeKey(token) {
    return token === "__proto__" || token === "constructor" || token === "prototype";
  }

  function updatePath(model, pointer, value) {
    if (!pointer || pointer === "/") return clone(value || {});
    const result = clone(model || {});
    const tokens = pointer.slice(1).split("/").map((part) => part.replaceAll("~1", "/").replaceAll("~0", "~"));
    if (tokens.some(isUnsafeKey)) return result;
    let cursor = result;
    for (const token of tokens.slice(0, -1)) {
      if (isUnsafeKey(token)) return result;
      const existing = Object.prototype.hasOwnProperty.call(cursor, token) ? cursor[token] : undefined;
      if (!existing || typeof existing !== "object") {
        const next = Object.create(null);
        Object.defineProperty(cursor, token, { value: next, writable: true, enumerable: true, configurable: true });
      }
      cursor = cursor[token];
    }
    const leaf = tokens[tokens.length - 1];
    if (isUnsafeKey(leaf) || !leaf) return result;
    if (value === undefined) {
      if (Object.prototype.hasOwnProperty.call(cursor, leaf)) delete cursor[leaf];
    } else {
      Object.defineProperty(cursor, leaf, {
        value: clone(value),
        writable: true,
        enumerable: true,
        configurable: true,
      });
    }
    return result;
  }

  function resolve(value, model, local) {
    if (!value || typeof value !== "object" || Array.isArray(value) || !Object.prototype.hasOwnProperty.call(value, "path")) return value;
    const pointer = text(value.path);
    const source = pointer.startsWith("/") ? model : local;
    const tokens = pointer.replace(/^\//, "").split("/").filter(Boolean).map((part) => part.replaceAll("~1", "/").replaceAll("~0", "~"));
    return tokens.reduce((current, token) => {
      if (current == null || isUnsafeKey(token) || !Object.prototype.hasOwnProperty.call(current, token)) return undefined;
      return current[token];
    }, source);
  }

  function actionFor(component, family) {
    return (component.actions || []).find((item) => item.name === family || item.name.endsWith(`.${family}`));
  }

  function generationExpired(surface, timeoutSeconds, now = Date.now()) {
    const created = Date.parse((surface || {}).created_at || "");
    return Number.isFinite(created) && now - created > Number(timeoutSeconds || 30) * 1000;
  }

  function init(config) {
    const mount = document.getElementById(config.mountId);
    const state = createState(config.sessionId);
    state.selected = text(config.surfaceId);
    let socket = null;
    let stopped = false;
    let reconnects = 0;
    const headers = () => ({ "Content-Type": "application/json", ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}) });

    mount.innerHTML = `<style>
      #mana-live-canvas,#mana-live-canvas *{box-sizing:border-box}#mana-live-canvas{font:14px ui-sans-serif,system-ui;color:#e8eaed;background:#0f1115}
      .canvas-shell{height:${Number(config.height || 760)}px;border:1px solid #ffffff22;border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
      .canvas-shell.fullscreen{position:fixed;inset:0;height:100vh;z-index:99999;border-radius:0}.canvas-bar{display:flex;align-items:center;gap:8px;padding:9px 11px;border-bottom:1px solid #ffffff1a;background:#171a20;flex-wrap:wrap}
      .canvas-bar button,.canvas-bar select{min-height:34px;background:#252a33;color:#fff;border:1px solid #ffffff25;border-radius:8px;padding:6px 10px}.canvas-bar button[aria-pressed=true]{background:#2563eb}
      .connection{margin-left:auto;color:#aeb4bd}.connection:before{content:'●';color:#f59e0b;margin-right:6px}.connection.ok:before{color:#22c55e}.canvas-main{flex:1;overflow:auto;padding:16px}.surface{max-width:1100px;margin:auto}.surface-grid{display:flex;gap:12px}.a2-column{display:flex;flex-direction:column;gap:10px}.a2-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
      .a2-card{border:1px solid #ffffff20;border-radius:12px;background:#181b22;padding:14px}.a2-heading{margin:.2em 0}.a2-markdown{white-space:pre-wrap;line-height:1.55}.a2-divider{border:0;border-top:1px solid #ffffff24;width:100%}.a2-button{background:#2563eb;color:white;border:0;border-radius:8px;padding:9px 14px;min-height:40px;font-weight:600}.a2-button:disabled{opacity:.5}.a2-field{display:grid;gap:5px;min-width:180px}.a2-field input,.a2-field textarea,.a2-field select{background:#101319;color:#fff;border:1px solid #ffffff2c;border-radius:8px;padding:9px;min-height:40px}.a2-field textarea{min-height:90px}.a2-badge{display:inline-flex;border-radius:999px;background:#334155;padding:4px 9px}.a2-progress{width:100%}.a2-image{max-width:100%;height:auto;border-radius:8px}.a2-artifact{color:#93c5fd}.a2-error,.unsupported{border:1px solid #ef444477;background:#450a0a55;border-radius:8px;padding:10px}.a2-empty{color:#9ca3af;padding:18px;text-align:center}.a2-table{width:100%;border-collapse:collapse;overflow:auto}.a2-table th,.a2-table td{padding:8px;border-bottom:1px solid #ffffff18;text-align:left}.surface-meta{color:#9ca3af;font-size:12px;margin:0 0 12px}.state-panel{max-width:900px;margin:40px auto;text-align:center;color:#aeb4bd}.action-error{color:#fca5a5;margin-top:8px}.workflow-item{border-left:3px solid #64748b;padding:8px 10px;margin:8px 0;background:#171a20}.workflow-item.failed{border-color:#ef4444}.workflow-item.success{border-color:#22c55e}
      @media(max-width:620px){.canvas-main{padding:9px}.a2-row{align-items:stretch;flex-direction:column}.a2-field{width:100%}.connection{width:100%;margin-left:0}}
    </style><div class="canvas-shell"><div class="canvas-bar"><select aria-label="Canvas surface"></select><button class="surfaceMode" aria-pressed="true">Surface</button><button class="workflowMode" aria-pressed="false">Workflow</button><button class="fullscreen" aria-label="Toggle full screen">Full screen</button><span class="connection" role="status">Connecting…</span></div><main class="canvas-main" aria-live="polite"></main></div>`;
    const shell = mount.querySelector(".canvas-shell");
    const main = mount.querySelector(".canvas-main");
    const picker = mount.querySelector("select");
    const connection = mount.querySelector(".connection");

    const submitAction = async (surface, component, declaration) => {
      const actionId = `canvas_action_${crypto.randomUUID().replaceAll("-", "")}`;
      state.pendingActions.add(component.id);
      state.actionErrors.delete(component.id);
      render();
      const context = {};
      for (const [key, value] of Object.entries(declaration.context || {})) context[key] = resolve(value, surface.data_model || {}, surface.data_model || {});
      try {
        const response = await fetch(`${config.apiBase}/api/v1/conversations/${encodeURIComponent(state.sessionId)}/canvas/surfaces/${encodeURIComponent(surface.surface_id)}/actions`, {
          method: "POST", headers: headers(), body: JSON.stringify({
            action_id: actionId, version: "v0.9", source_component_id: component.id,
            name: declaration.name, correlation_id: actionId,
            timestamp: new Date().toISOString(), context, root: config.root,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
      } catch (error) {
        state.actionErrors.set(component.id, error.message || String(error));
      } finally {
        state.pendingActions.delete(component.id);
        render();
      }
    };

    function renderComponent(surface, id, trail, local) {
      const component = (surface.components || []).find((item) => item.id === id);
      if (!component) return unsupported(`Missing component ${id}`);
      if (trail.has(id)) return unsupported(`Cyclic component ${id}`);
      const nextTrail = new Set(trail); nextTrail.add(id);
      const model = surface.data_model || {};
      const node = document.createElement("div");
      node.dataset.componentId = id;
      const children = () => (component.children || []).map((child) => renderComponent(surface, child, nextTrail, local));
      const appendChildren = (parent) => { for (const child of children()) parent.appendChild(child); };
      const value = (name, fallback = "") => resolve(component[name], model, local) ?? fallback;
      if (component.component === "Text") { node.textContent = text(value("text")); }
      else if (component.component === "Heading") { const level = Math.min(6, Math.max(1, Number(component.level || 2))); const heading = document.createElement(`h${level}`); heading.className="a2-heading"; heading.textContent=text(value("text")); node.appendChild(heading); }
      else if (component.component === "Markdown") { node.className="a2-markdown"; renderMarkdown(node,text(value("text"))); }
      else if (component.component === "Column" || component.component === "Row") { node.className=component.component === "Column" ? "a2-column" : "a2-row"; appendChildren(node); }
      else if (component.component === "Card") { node.className="a2-card"; if(component.child) node.appendChild(renderComponent(surface,component.child,nextTrail,local)); else appendChildren(node); }
      else if (component.component === "Divider") { const hr=document.createElement("hr");hr.className="a2-divider";node.appendChild(hr); }
      else if (component.component === "Button") { const button=document.createElement("button");button.className="a2-button";button.textContent=text(value("label"));const action=actionFor(component,"press");button.type=action?"button":"submit";button.disabled=Boolean(value("disabled",false))||state.pendingActions.has(id);button.addEventListener("click",()=>action&&submitAction(surface,component,action));node.appendChild(button); }
      else if (["TextField","TextArea","Select","Checkbox","RadioGroup"].includes(component.component)) renderInput(node,surface,component,value);
      else if (component.component === "Form") { const form=document.createElement("form");form.className="a2-column";appendChildren(form);const action=actionFor(component,"submit");form.addEventListener("submit",(event)=>{event.preventDefault();if(action)submitAction(surface,component,action);});node.appendChild(form); }
      else if (component.component === "Badge") { const badge=document.createElement("span");badge.className="a2-badge";badge.textContent=text(value("text"));node.appendChild(badge); }
      else if (component.component === "Progress") { const progress=document.createElement("progress");progress.className="a2-progress";progress.max=Number(component.max||100);progress.value=Number(value("value",0));progress.setAttribute("aria-label",text(component.label||"Progress"));node.appendChild(progress); }
      else if (component.component === "Image") { const image=document.createElement("img");const url=safeResourceUrl(value("url"));if(!url)return unsupported("Blocked image URL");image.className="a2-image";image.src=url;image.alt=text(value("description"));image.loading="lazy";node.appendChild(image); }
      else if (component.component === "Artifact") { const link=document.createElement("a");link.className="a2-artifact";link.textContent=text(value("label"));const url=safeResourceUrl(value("url"));link.href=url||"#";link.rel="noopener noreferrer";link.target="_blank";node.appendChild(link); }
      else if (component.component === "Table") renderTable(node,value("columns",[]),value("rows",[]));
      else if (component.component === "List") { const list=document.createElement("ul");for(const item of value("items",[])){const li=document.createElement("li");li.textContent=text(typeof item==="object"?item.label||item.text||JSON.stringify(item):item);list.appendChild(li);}node.appendChild(list); }
      else if (component.component === "Tabs") { const tabs=document.createElement("div");tabs.className="a2-row";for(const tab of component.tabs||[]){const button=document.createElement("button");button.type="button";button.textContent=text(tab.label);button.addEventListener("click",()=>{node.querySelector(".tab-content").replaceChildren(renderComponent(surface,tab.child,nextTrail,local));});tabs.appendChild(button);}node.appendChild(tabs);const content=document.createElement("div");content.className="tab-content";if((component.tabs||[])[0])content.appendChild(renderComponent(surface,component.tabs[0].child,nextTrail,local));node.appendChild(content); }
      else if (component.component === "ErrorState") { node.className="a2-error";node.setAttribute("role","alert");node.textContent=text(value("message")); }
      else if (component.component === "EmptyState") { node.className="a2-empty";node.textContent=text(value("message")); }
      else return unsupported(`Unsupported component: ${component.component}`);
      const error = state.actionErrors.get(id);if(error){const e=document.createElement("div");e.className="action-error";e.setAttribute("role","alert");e.textContent=error;node.appendChild(e);}
      return node;
    }

    function renderMarkdown(node,source){let list=null;for(const line of source.split("\n")){const heading=line.match(/^(#{1,6})\s+(.+)$/);if(heading){list=null;const h=document.createElement(`h${heading[1].length}`);h.textContent=heading[2];node.appendChild(h);continue;}const bullet=line.match(/^[-*]\s+(.+)$/);if(bullet){if(!list){list=document.createElement("ul");node.appendChild(list);}const li=document.createElement("li");li.textContent=bullet[1];list.appendChild(li);continue;}list=null;const p=document.createElement("div");p.textContent=line||" ";node.appendChild(p);}}

    function renderInput(node,surface,component,value){
      node.className="a2-field";const label=document.createElement("label");label.htmlFor=`field-${component.id}`;label.textContent=text(value("label"));node.appendChild(label);let input;
      if(component.component==="RadioGroup"){const current=text(value("value"));for(const option of value("options",[])){const wrap=document.createElement("label");const radio=document.createElement("input");radio.type="radio";radio.name=`radio-${component.id}`;radio.value=text(option.value??option);radio.checked=radio.value===current;radio.addEventListener("change",()=>{const binding=component.value&&component.value.path;if(binding&&radio.checked)surface.data_model=updatePath(surface.data_model,binding,radio.value);});wrap.appendChild(radio);wrap.appendChild(document.createTextNode(text(option.label??option)));node.appendChild(wrap);}return;}
      if(component.component==="TextArea")input=document.createElement("textarea");else if(component.component==="Select")input=document.createElement("select");else input=document.createElement("input");
      input.id=`field-${component.id}`;
      if(component.component==="Checkbox")input.type="checkbox";else if(component.component==="RadioGroup")input.type="radio";else input.type="text";
      if(component.component==="Select")for(const option of value("options",[])){const o=document.createElement("option");o.value=text(option.value??option);o.textContent=text(option.label??option);input.appendChild(o);}
      const binding=component.value&&component.value.path;if(component.component==="Checkbox")input.checked=Boolean(value("value"));else input.value=text(value("value"));
      input.addEventListener("input",()=>{if(binding&&binding.startsWith("/")){surface.data_model=updatePath(surface.data_model,binding,component.component==="Checkbox"?input.checked:input.value);}});node.appendChild(input);
    }

    function renderTable(node,columns,rows){const wrap=document.createElement("div");wrap.style.overflow="auto";const table=document.createElement("table");table.className="a2-table";const head=document.createElement("thead");const hr=document.createElement("tr");for(const col of columns||[]){const th=document.createElement("th");th.textContent=text(col.label??col.key??col);hr.appendChild(th);}head.appendChild(hr);table.appendChild(head);const body=document.createElement("tbody");for(const row of rows||[]){const tr=document.createElement("tr");for(const col of columns||[]){const td=document.createElement("td");const key=col.key??col;td.textContent=text(typeof row==="object"?row[key]:row);tr.appendChild(td);}body.appendChild(tr);}table.appendChild(body);wrap.appendChild(table);node.appendChild(wrap);}
    function unsupported(message){const node=document.createElement("div");node.className="unsupported";node.setAttribute("role","status");node.textContent=message;return node;}
    function surfaceFailure(){const node=document.createElement("div");node.className="state-panel a2-error";node.setAttribute("role","alert");node.textContent="Surface generation did not complete. Ask the agent to retry.";return node;}

    function render() {
      const active = document.activeElement && document.activeElement.id;
      picker.replaceChildren();
      const surfaces = [...state.surfaces.values()].filter((item) => !item.deleted);
      for (const surface of surfaces) { const option=document.createElement("option");option.value=surface.surface_id;option.textContent=`${surface.surface_id}${surface.completed?" · complete":""}`;option.selected=surface.surface_id===state.selected;picker.appendChild(option); }
      picker.disabled=!surfaces.length;
      main.replaceChildren();
      if(state.loading){const p=document.createElement("div");p.className="state-panel";p.textContent="Restoring Live Canvas…";main.appendChild(p);}
      else if(state.error){const p=document.createElement("div");p.className="state-panel a2-error";p.setAttribute("role","alert");p.textContent=state.error;main.appendChild(p);}
      else if(state.mode==="workflow")renderWorkflow();
      else { const surface=state.surfaces.get(state.selected)||surfaces[0];if(!surface){const p=document.createElement("div");p.className="state-panel";p.textContent="No active surfaces in this session.";main.appendChild(p);}else{state.selected=surface.surface_id;const section=document.createElement("section");section.className="surface";section.setAttribute("aria-label",`Canvas surface ${surface.surface_id}`);const provenance=document.createElement("div");provenance.className="surface-meta";provenance.textContent=`A2UI ${surface.protocol_version} · owner ${Object.entries(surface.owner||{}).filter(([,v])=>v).map(([k,v])=>`${k}:${v}`).join(" · ")||"unknown"}`;section.appendChild(provenance);if(!(surface.components||[]).length){section.appendChild(generationExpired(surface,config.generationTimeoutSeconds)?surfaceFailure():unsupported("Waiting for surface content…"));}else{section.appendChild(renderComponent(surface,"root",new Set(),surface.data_model||{}));}main.appendChild(section);} }
      connection.classList.toggle("ok",state.socketReady);connection.textContent=state.socketReady?"Live":"Disconnected · recovering";shell.classList.toggle("fullscreen",state.fullscreen);
      if(active){const target=document.getElementById(active);if(target)target.focus();}
    }

    function renderWorkflow(){const section=document.createElement("section");section.className="surface";const title=document.createElement("h2");title.textContent="Workflow activity";section.appendChild(title);if(!state.activities.length)section.appendChild(unsupported("No workflow activity yet."));for(const event of state.activities.slice(-100)){const item=document.createElement("div");item.className=`workflow-item ${text(event.status)}`;const heading=document.createElement("div");heading.textContent=`${event.title||event.type} · ${event.status||"running"}`;item.appendChild(heading);const metadata=event.metadata||{};const details=document.createElement("small");details.textContent=[metadata.agent_id,metadata.task_id,metadata.tool_name].filter(Boolean).join(" · ");item.appendChild(details);section.appendChild(item);}main.appendChild(section);}

    async function hydrate(){state.loading=true;render();try{const response=await fetch(`${config.apiBase}/api/v1/conversations/${encodeURIComponent(state.sessionId)}/canvas/surfaces?root=${encodeURIComponent(config.root)}`);const payload=await response.json();if(!response.ok)throw new Error(payload.detail||`HTTP ${response.status}`);for(const surface of payload.surfaces||[])applySnapshot(state,surface);if(!state.selected&&payload.surfaces&&payload.surfaces[0])state.selected=payload.surfaces[0].surface_id;state.error="";}catch(error){state.error=error.message||String(error);}finally{state.loading=false;render();}}
    const socketUrl=()=>`${config.wsBase}/api/v1/ws/conversations/${encodeURIComponent(state.sessionId)}?root=${encodeURIComponent(config.root)}&replay_limit=1000&after_sequence=${state.lastSequence}${config.token?`&token=${encodeURIComponent(config.token)}`:""}`;
    function connect(){if(stopped)return;socket=new WebSocket(socketUrl());socket.onmessage=(message)=>{const packet=JSON.parse(message.data);if(packet.type==="socket.ready"){state.socketReady=true;reconnects=0;render();}else if(packet.type==="event"||packet.type==="event.replay"){applyHubEvent(state,packet.event);render();}else if(packet.type==="socket.replay_complete"&&state.error.includes("sequence gap"))hydrate();};socket.onclose=()=>{state.socketReady=false;render();if(!stopped)setTimeout(connect,Math.min(10000,400*Math.pow(2,reconnects++)));};socket.onerror=()=>socket.close();}
    picker.addEventListener("change",()=>{state.selected=picker.value;render();});mount.querySelector(".surfaceMode").addEventListener("click",()=>{state.mode="surface";mount.querySelector(".surfaceMode").setAttribute("aria-pressed","true");mount.querySelector(".workflowMode").setAttribute("aria-pressed","false");render();});mount.querySelector(".workflowMode").addEventListener("click",()=>{state.mode="workflow";mount.querySelector(".surfaceMode").setAttribute("aria-pressed","false");mount.querySelector(".workflowMode").setAttribute("aria-pressed","true");render();});mount.querySelector(".fullscreen").addEventListener("click",()=>{state.fullscreen=!state.fullscreen;render();});
    window.addEventListener("beforeunload",()=>{stopped=true;if(socket)socket.close();},{once:true});
    hydrate();connect();
    return { state, applySnapshot:(value)=>applySnapshot(state,value), applyEvent:(value)=>applyCanvasEvent(state,value), render, close:()=>{stopped=true;if(socket)socket.close();} };
  }

  return { createState, applySnapshot, applyCanvasEvent, applyHubEvent, updatePath, resolve, generationExpired, init };
});

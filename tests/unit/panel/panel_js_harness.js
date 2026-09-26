// Runs a panel page's inline <script> against a fake DOM, fetch, WebSocket and virtual timers.
// Usage: node panel_js_harness.js <page.html> <scenario>; prints one JSON object of observations.
"use strict";
const fs = require("fs");
const vm = require("vm");

const [, , pagePath, scenario] = process.argv;
const html = fs.readFileSync(pagePath, "utf8");
const source = html.match(/<script>([\s\S]*)<\/script>/)[1];

// --- virtual time ---------------------------------------------------------------------------
let now = 0;
let seq = 0;
const timers = new Map();
const setTimeoutV = (fn, ms) => { const id = ++seq; timers.set(id, { at: now + Math.max(0, ms || 0), fn, every: 0 }); return id; };
const setIntervalV = (fn, ms) => { const id = ++seq; timers.set(id, { at: now + ms, fn, every: Math.max(1, ms) }); return id; };
const clearTimeoutV = (id) => { timers.delete(id); };
const flush = async () => { for (let i = 0; i < 30; i += 1) await new Promise((r) => setImmediate(r)); };
async function advance(ms) {
  const end = now + ms;
  for (;;) {
    let next = null;
    for (const [id, t] of timers) if (t.at <= end && (!next || t.at < next[1].at)) next = [id, t];
    if (!next) break;
    const [id, t] = next;
    now = t.at;
    if (t.every) t.at += t.every; else timers.delete(id);
    t.fn();
    await flush();
  }
  now = end;
  await flush();
}

// --- fake DOM -------------------------------------------------------------------------------
class Node {}
class ClassList {
  constructor() { this.items = new Set(); }
  add(c) { this.items.add(c); }
  remove(c) { this.items.delete(c); }
  contains(c) { return this.items.has(c); }
  toggle(c, on) { const v = on === undefined ? !this.items.has(c) : !!on; if (v) this.items.add(c); else this.items.delete(c); return v; }
}
class Text extends Node {
  constructor(text) { super(); this.textContent = String(text); this.parent = null; }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
}
class Element extends Node {
  constructor(tag, id = "") {
    super();
    this.tagName = String(tag).toUpperCase();
    this.id = id;
    this.children = [];
    this.parent = null;
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.classList = new ClassList();
    this.style = { cssText: "", setProperty: (k, v) => { this.style[k] = v; } };
    this.hidden = false; this.value = ""; this.disabled = false; this.open = false; this.className = "";
    this.isContentEditable = false;
  }
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.replaceChildren(new Text(v)); }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, ev) { for (const fn of this.listeners[type] || []) fn(ev); }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  _adopt(kids) { return kids.map((k) => { const n = k instanceof Node ? k : new Text(k); if (n.parent) n.remove(); n.parent = this; return n; }); }
  append(...kids) { this.children.push(...this._adopt(kids)); }
  prepend(...kids) { this.children.unshift(...this._adopt(kids)); }
  replaceChildren(...kids) { for (const c of this.children) c.parent = null; this.children = this._adopt(kids); }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
  querySelector() { return null; }
  focus() {}
  showModal() { this.open = true; }
  setPointerCapture() {}
}
const byId = new Map();
const docListeners = {};
const document = {
  getElementById(id) {
    if (!byId.has(id)) { const el = new Element("div", id); el.append(new Text("")); byId.set(id, el); }
    return byId.get(id);
  },
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => new Text(text),
  querySelectorAll: () => [],
  addEventListener(type, fn) { (docListeners[type] = docListeners[type] || []).push(fn); },
  get activeElement() { return null; },
};
const key = (code, extra = {}) => {
  const ev = { code, key: code.slice(3).toLowerCase(), repeat: false, ctrlKey: false, metaKey: false, altKey: false,
    target: new Element("body"), preventDefault() {}, ...extra };
  for (const fn of docListeners.keydown || []) fn(ev);
};

// --- fake network ---------------------------------------------------------------------------
const fetches = [];
const routes = {};
function respond(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}
async function fakeFetch(url, init = {}) {
  const u = new URL(url, "http://127.0.0.1:8770");
  const entry = { url: String(url), path: u.pathname, method: init.method || "GET", mode: init.mode || "cors",
    body: init.body ? JSON.parse(init.body) : null, at: now };
  fetches.push(entry);
  const handler = routes[u.pathname];
  if (handler === "hang") {
    return new Promise((_, reject) => {
      if (init.signal) init.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    });
  }
  if (typeof handler === "function") return handler(entry);
  return respond({ ok: true });
}
const sockets = [];
class FakeWebSocket {
  constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  emit(type, ev) { for (const fn of this.listeners[type] || []) fn(ev); }
  close() {}
  send() {}
}

function run(href, storage = {}) {
  const location = new URL(href);
  const sandbox = {
    console, URL, URLSearchParams, AbortController, JSON, Math, Date, Promise, Map, Set, Number, String, Object, Array,
    Node, document, location, fetch: fakeFetch, WebSocket: FakeWebSocket,
    setTimeout: setTimeoutV, clearTimeout: clearTimeoutV, setInterval: setIntervalV, clearInterval: clearTimeoutV,
    history: { replaceState: (a, b, url) => { location.href = new URL(url, location.href).href; } },
    sessionStorage: { data: { ...storage }, getItem(k) { return this.data[k] ?? null; }, setItem(k, v) { this.data[k] = String(v); } },
    window: { addEventListener() {} }, confirm: () => true, prompt: () => "",
  };
  vm.runInNewContext(source, sandbox, { filename: pagePath });
  return sandbox;
}

// --- DOM queries ----------------------------------------------------------------------------
function findAll(el, pred, out = []) {
  for (const child of el.children || []) {
    if (child instanceof Element) {
      if (pred(child)) out.push(child);
      findAll(child, pred, out);
    }
  }
  return out;
}
const buttons = (el) => findAll(el, (n) => n.tagName === "BUTTON");
const buttonTexts = (el) => buttons(el).map((b) => b.textContent);
const click = (el, text) => {
  const btn = buttons(el).find((b) => b.textContent === text);
  if (!btn) throw new Error(`no button ${text}: ${buttonTexts(el)}`);
  btn.dispatch("click", {});
};
const posted = (path) => fetches.filter((f) => f.path === path && f.method === "POST").map((f) => f.body);

const CONFIG = { emergency_url: "http://127.0.0.1:8779", emergency_token: "EMERG", characters: ["pailin"],
  default_character: "pailin", memory: false, ops: false, moderation: false };
const STATE = { ok: true, control: { muted: false, mic_mode: "open", characters: { pailin: { state: "speaking" } } },
  health: [], states: { pailin: "speaking" }, recent: {} };

async function panelScenario(cmdRoute) {
  routes["/api/config"] = () => respond(CONFIG);
  routes["/api/state"] = () => respond(STATE);
  routes["/api/traces"] = () => respond({ ok: true, rows: [], badges: {}, alarms: [] });
  routes["/api/chat"] = () => respond({ ok: false }, 404);
  routes["/api/cmd"] = cmdRoute;
  run("http://127.0.0.1:8770/?token=PANEL");
  await advance(10);
  const t0 = now;
  key("KeyF");
  await advance(299);
  const before = fetches.filter((f) => f.path === "/hardkill").length;
  await advance(2);
  const kills = fetches.filter((f) => f.path === "/hardkill");
  await advance(6000);
  return {
    token_header_used: fetches.some((f) => f.path === "/api/config"),
    freeze_posted: fetches.filter((f) => f.path === "/api/cmd").map((f) => f.body),
    hardkill_before_300ms: before,
    hardkills: kills.map((f) => ({ url: f.url, method: f.method, mode: f.mode, after_ms: f.at - t0 })),
    hardkills_total: fetches.filter((f) => f.path === "/hardkill").length,
    alarm: document.getElementById("alarm-text").textContent,
    ws_url: sockets.length ? sockets[0].url : null,
  };
}

async function keysScenario() {
  routes["/api/config"] = () => respond(CONFIG);
  routes["/api/state"] = () => respond(STATE);
  run("http://127.0.0.1:8770/#token=PANEL");
  await advance(10);
  key("KeyS", { key: "ห" });                      // Thai layout: physical S
  key("KeyM", { key: "ท" });
  key("KeyS", { target: new Element("textarea") }); // typing: ignored
  key("KeyF", { ctrlKey: true });                  // modifiers: ignored
  await advance(50);
  return { commands: fetches.filter((f) => f.path === "/api/cmd").map((f) => f.body.kind) };
}

async function captionsScenario() {
  run("http://127.0.0.1:8770/overlay/captions?token=PANEL&character=pailin");
  await advance(1);
  const ws = sockets[0];
  const box = document.getElementById("caption");
  const frame = (events, t) => ws.emit("message", { data: JSON.stringify({ kind: "events", now: t, dropped: 0, events }) });
  const out = { ws_url: ws.url };
  frame([{ type: "UtteranceStarted", utt_id: "u1", stimulus_id: "s1", character: "pailin" },
    { type: "SegmentStarted", utt_id: "u1", seq: 0, t_audible: 100.5, duration_s: 1.0, caption: "สวัสดีค่ะ", character: "pailin" },
    { type: "SegmentStarted", utt_id: "u9", seq: 0, t_audible: 100.0, duration_s: 1.0, caption: "twin", character: "twin" }], 100.0);
  await advance(400);
  out.before_audible = { text: box.textContent, shown: box.classList.contains("show") };
  await advance(200);
  out.at_audible = { text: box.textContent, shown: box.classList.contains("show") };
  frame([{ type: "Filtered", direction: "out", tier: "tier0", category: "slur", rule: "r", ref: null, character: "pailin" }], 101.0);
  await advance(10);
  // §4.10: the clean segment still playing (until 101.5) keeps its caption
  out.during_playing = { text: box.textContent, red: box.classList.contains("filtered") };
  frame([{ type: "SegmentStarted", utt_id: "u1", seq: 1, t_audible: 101.02, duration_s: 1.0, caption: "late", character: "pailin" }], 101.01);
  await advance(10);
  out.after_filtered_segment = box.textContent;
  await advance(490);
  out.filtered = { text: box.textContent, shown: box.classList.contains("show"), red: box.classList.contains("filtered") };
  frame([{ type: "SegmentStarted", utt_id: "u1", seq: 2, t_audible: 101.6, duration_s: 0.8, caption: "Filtered.", character: "pailin" }], 101.52);
  await advance(100);
  out.canned = { text: box.textContent, red: box.classList.contains("filtered") };
  frame([{ type: "UtteranceDone", utt_id: "u1", heard_text: "สวัสดีค่ะ", cancelled: true, reason: "filtered", filtered: true, character: "pailin" }], 102.4);
  await advance(5000);
  out.hidden_later = !box.classList.contains("show");
  return out;
}

async function reloadWedgedScenario() {
  // The page reloads while the core is wedged: /api/config never answers.
  routes["/api/config"] = "hang";
  routes["/api/cmd"] = "hang";
  const storage = {
    "aivtube.panel.token": "PANEL",
    "aivtube.panel.emergency": JSON.stringify({ emergency_url: "http://127.0.0.1:8779", emergency_token: "EMERG" }),
  };
  run("http://127.0.0.1:8770/", storage);
  await advance(4100);  // the config request times out; boot retries later
  const bootAlarm = document.getElementById("alarm-text").textContent;
  const bootButton = !document.getElementById("alarm-hardkill").hidden;
  const t0 = now;
  key("KeyF");
  await advance(301);
  const kills = fetches.filter((f) => f.path === "/hardkill");
  return {
    config_requests: fetches.filter((f) => f.path === "/api/config").length,
    boot_alarm: bootAlarm,
    alarm_hardkill_button: bootButton,
    alarm: document.getElementById("alarm-text").textContent,
    hardkills: kills.map((f) => ({ url: f.url, after_ms: f.at - t0 })),
  };
}

async function approvalsScenario() {
  const state = JSON.parse(JSON.stringify(STATE));
  state.control.approvals = [{ id: "ap1", character: "pailin", tool: "timeout_user", args: { user: "troll", seconds: 60 }, age_s: 3.2 }];
  state.control.tools = { mode: "off", resume_mode: "live", tools: { timeout_user: { enabled: true }, play_sound: { enabled: false } } };
  routes["/api/config"] = () => respond(CONFIG);
  routes["/api/state"] = () => respond(state);
  routes["/api/cmd"] = () => respond({ ok: true, detail: "timeout_user approved", latency_ms: 1 });
  run("http://127.0.0.1:8770/?token=PANEL");
  await advance(10);
  const card = document.getElementById("approvals-card");
  const list = document.getElementById("approval-list");
  const out = { visible: !card.hidden, text: list.textContent, buttons: buttonTexts(list) };
  click(list, "อนุมัติ");
  await advance(10);
  out.tools = buttonTexts(document.getElementById("tool-list"));
  out.resume_note = document.getElementById("tools-resume").textContent;
  click(document.getElementById("tool-list"), "play_sound");
  await advance(10);
  out.commands = posted("/api/cmd");
  state.control.approvals = [];
  await advance(2100);
  out.hidden_after = card.hidden;
  return out;
}

async function moderationScenario() {
  const item = { id: null, key: "k1", ts: 1000.5, character: "pailin", direction: "in", source: "twitch", tier: "tier0",
    category: "slur", rule: "r1", verdict: "drop", text: "คำต้องห้าม นะ", ref: "abcdef0123456789", platform: "twitch", user_id: "u-9" };
  let items = [];
  const polls = [];
  routes["/api/config"] = () => respond({ ...CONFIG, moderation: true, moderation_log: true });
  routes["/api/state"] = () => respond(STATE);
  routes["/api/moderation"] = (entry) => {
    if (entry.method === "POST") return respond({ ok: true, detail: "", latency_ms: 1 });
    polls.push(entry.url);
    return respond({ ok: true, source: "live", items });
  };
  run("http://127.0.0.1:8770/?token=PANEL");
  await advance(10);
  const ws = sockets[0];
  items = [item];
  ws.emit("message", { data: JSON.stringify({ kind: "events", now: 1, dropped: 0, events: [
    { type: "Filtered", direction: "in", tier: "tier0", category: "slur", rule: "r1", ref: "abcdef0123456789", character: "pailin" },
    { type: "ChatDropped", message_id: "m1", reason: "duplicate", character: "pailin" },
  ] }) });
  await advance(299);
  const feed = document.getElementById("mod-feed");
  const out = { before_poll: feed.textContent };
  await advance(2);
  out.after_poll = feed.textContent;
  out.buttons = buttonTexts(feed);
  await advance(5000);  // the periodic poll returns the same item: shown once
  out.items = feed.children.filter((c) => c.tagName === "LI").length;
  out.last_poll = polls[polls.length - 1];
  click(feed, "ปิดเสียงผู้ใช้ 10 นาที");
  await advance(10);
  out.mute = posted("/api/moderation");
  return out;
}

async function memoryScenario() {
  const items = [
    { id: 1, kind: "fact", text: "ชอบแมว", status: "quarantined", importance: 3, source: "model", origin: "chat", pinned: false, locked: false },
    { id: 2, kind: "core", slot: 1, text: "ชื่อไพลิน", status: "active", importance: 5, source: "operator", origin: "", pinned: true, locked: true },
  ];
  routes["/api/config"] = () => respond({ ...CONFIG, memory: true, memory_edit_fields: ["text", "importance", "locked"] });
  routes["/api/state"] = () => respond(STATE);
  routes["/api/memory"] = () => respond({ ok: true, items });
  routes["/api/memory/1"] = () => respond({ ok: true, results: [], item: items[0] });
  run("http://127.0.0.1:8770/?token=PANEL");
  await advance(10);
  const rows = document.getElementById("mem-rows");
  const out = { rows: rows.children.length, quarantined: buttonTexts(rows.children[0]), core: buttonTexts(rows.children[1]) };
  click(rows.children[0], "อนุมัติ");
  await advance(10);
  out.patch = fetches.filter((f) => f.path === "/api/memory/1").map((f) => [f.method, f.body]);
  return out;
}

(async () => {
  let result;
  if (scenario === "freeze_hang") result = await panelScenario("hang");
  else if (scenario === "freeze_fast") result = await panelScenario(() => respond({ ok: true, detail: "", latency_ms: 3 }));
  else if (scenario === "keys") result = await keysScenario();
  else if (scenario === "captions") result = await captionsScenario();
  else if (scenario === "reload_wedged") result = await reloadWedgedScenario();
  else if (scenario === "approvals") result = await approvalsScenario();
  else if (scenario === "moderation") result = await moderationScenario();
  else if (scenario === "memory") result = await memoryScenario();
  else throw new Error(`unknown scenario ${scenario}`);
  process.stdout.write(JSON.stringify(result));
})().catch((err) => { console.error(err); process.exit(1); });

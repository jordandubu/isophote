// blender.js — the local Blender relay (127.0.0.1:8778): polls for channel
// queues pushed from the add-on, sends painted maps / animated frames back.
// Wire protocol is unchanged from before the UI rewrite.
import { $, h } from "./dom.js";
import { b64ToBlob, bitmapToB64, CANCELLED } from "./engine.js";

const BRIDGE_PORT = 8778;
const url = (path) => `http://127.0.0.1:${BRIDGE_PORT}${path}`;

export const KIND_LABEL = {
  color: "base color", normal: "normal map", rough: "roughness", metal: "metallic",
  emission: "emission", orm: "ORM", other: "custom", height: "impasto height",
  preview: "relief preview", image: "image",
};
const kindList = (kinds) => kinds.map((k) => KIND_LABEL[k] || k).join(" + ");

// app: { source(), result(), activeKind(), setKind(k), loadQueue(kinds, blobs),
//        ensureFinal(), anim, store, toast(msg) }
export function createBlender(app) {
  const B = { enabled: false, connected: null, timer: null, inFlight: false, pending: false, lastHash: 0, msg: "" };

  const dot = h("span", { class: "conn", title: "" });
  const msg = h("div", { class: "bmsg" });
  const seg = h("div", { class: "seg", hidden: true });
  const sendBtn = h("button", { class: "primary", type: "button", disabled: true, onclick: () => guard(sendCurrent) }, "Send to Blender");
  const animBtn = h("button", { class: "ghost", type: "button", disabled: true, onclick: () => guard(sendAnimated),
    title: "Uses Frames and Boil from the Animation section" }, "Send animated");
  const section = h("details", { class: "sec", id: "sec-blender", open: true },
    h("summary", {}, "Blender", dot),
    h("div", { class: "secbody" }, msg, seg, h("div", { class: "btnrow" }, sendBtn, animBtn)));
  section.hidden = true;
  $("blenderSlot").append(section);

  function say(text) {
    if (text === B.msg) return;      // the relay repeats its status on every poll
    B.msg = text; msg.textContent = text; app.toast(text);
  }

  function setConnected(on) {
    if (B.connected === on) return;
    B.connected = on;
    dot.className = "conn " + (on ? "on" : "off");
    dot.title = on ? "Connected to Blender" : "Blender relay offline";
  }

  // channel tabs + enable state follow the current result
  function refresh() {
    const src = app.source(), res = app.result();
    const ready = !!res;
    sendBtn.disabled = !ready;
    animBtn.disabled = !ready;
    const kinds = res && src && src.multi ? res.kinds : [];
    seg.hidden = kinds.length < 2;
    seg.replaceChildren(...kinds.map((k) =>
      h("button", { type: "button", class: k === app.activeKind() ? "on" : "", onclick: () => app.setKind(k) }, KIND_LABEL[k] || k)));
  }

  async function guard(fn) {
    try { await fn(); }
    catch (e) { if (e !== CANCELLED) say("send failed: " + (e.message || e)); }
  }

  const sendable = (res) => res.kinds.map((k, i) => [k, i]).filter(([k]) => k !== "preview");

  async function sendCurrent() {
    const res = await app.ensureFinal();
    if (!res) return;
    if (app.source().multi) {
      const items = sendable(res);
      const pngs = [];
      for (const [, i] of items) pngs.push(await bitmapToB64(res.bitmaps[i]));
      const kinds = items.map(([k]) => k);
      await post({ kinds, pngs, height_scale: res.heightScale });
      say(`sent ${kinds.length} maps (${kindList(kinds)}) — now click 'Apply from Isophote' inside Blender`);
    } else {
      await post({ png: await bitmapToB64(res.bitmaps[0]), kind: "normal" });
      say("sent — now click 'Apply from Isophote' inside Blender");
    }
  }

  async function sendAnimated() {
    const res = await app.ensureFinal();
    if (!res) return;
    const overlay = $("animProgress"), bar = $("animBar"), count = $("animCount");
    overlay.hidden = false;
    bar.style.width = "0%";
    try {
      const frames = await app.anim.ensure((i, n) => {
        bar.style.width = Math.round(i / n * 100) + "%";
        count.textContent = `${i} / ${n}`;
      });
      count.textContent = "sending to Blender…";
      const multi = app.source().multi;
      if (multi) {
        const items = sendable(frames[0]);
        const kinds = items.map(([k]) => k);
        const anim = [];
        for (const f of frames) anim.push(await Promise.all(items.map(([, i]) => bitmapToB64(f.bitmaps[i]))));
        await post({ kinds, anim, height_scale: frames[0].heightScale });
        say(`sent ${frames.length} frames × ${kinds.length} maps (${kindList(kinds)}) — now click 'Apply from Isophote' (Animated checked) inside Blender`);
      } else {
        const anim = [];
        for (const f of frames) anim.push(await bitmapToB64(f.bitmaps[0]));
        await post({ anim });
        say(`sent ${frames.length} frames — now click 'Apply from Isophote' (Animated checked) inside Blender`);
      }
    } finally {
      overlay.hidden = true;
    }
  }

  async function post(body) {
    await fetch(url("/send_to_blender"), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  }

  async function poll() {
    if (B.inFlight) return;
    B.inFlight = true;
    try {
      const r = await fetch(url("/poll"), { method: "POST", body: "{}" });
      let d = await r.json();
      setConnected(true);
      // a single Blender image is a one-kind queue: same path, so impasto
      // relief (height map + preview) works for normal-only sends too
      if (d.kind === "image" && d.bgr_png) d = { kind: "multi", kinds: [d.map_kind || "normal"], pngs: [d.bgr_png], status: d.status };
      if (d.status) say(d.status);
      if (d.kind === "multi" && Array.isArray(d.kinds) && Array.isArray(d.pngs) && d.pngs.length === d.kinds.length && !B.pending) {
        let hash = 0;
        for (const png of d.pngs) for (let i = 0; i < png.length; i += 4096) hash = (hash + (png.charCodeAt(i) + i)) | 0;
        if (hash === B.lastHash) return;
        B.lastHash = hash;          // claim immediately: blocks the sample load
        B.pending = true;
        try {
          const blobs = {};
          d.kinds.forEach((k, i) => { blobs[k] = b64ToBlob(d.pngs[i], "image/png"); });
          say(`received ${kindList(d.kinds)} — painting…`);
          await app.loadQueue(d.kinds, blobs);
          fetch(url("/ack"), { method: "POST", body: "{}" }).catch(() => {});
          say(`received ${kindList(d.kinds)} — adjust sliders freely; ALL channels repaint on Send`);
        } finally {
          B.pending = false;
        }
      }
    } catch (e) {
      setConnected(false);
      if (B.enabled) { B.msg = "bridge offline — restart Blender (runs the relay on 127.0.0.1:8778)"; msg.textContent = B.msg; }
    } finally {
      B.inFlight = false;
    }
  }

  return {
    refresh,
    setEnabled(on) {
      B.enabled = on;
      section.hidden = !on;
      clearInterval(B.timer); B.timer = null;
      if (on) { poll(); B.timer = setInterval(poll, 1500); refresh(); }
    },
    // true once the relay delivered (or is about to deliver) a Blender queue
    claimed: () => B.lastHash !== 0 || B.pending,
  };
}

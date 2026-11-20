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
  // channel shown on the canvas (base color, normal map, impasto height, relief preview...)
  const sel = h("select", { class: "chsel", hidden: true, title: "Channel shown on the canvas", "aria-label": "Channel",
    onchange: (e) => app.setKind(e.target.value) });
  const section = h("details", { class: "sec", id: "sec-blender", open: true },
    h("summary", {}, "Blender", dot),
    h("div", { class: "secbody" }, sel));
  section.hidden = true;
  $("blenderSlot").append(section);

  function say(text, ms) {
    if (text === B.msg) return;      // the relay repeats its status on every poll
    B.msg = text; app.toast(text, ms);
  }
  const SENT_MS = 3000;              // "sent" confirmations are brief

  function setConnected(on) {
    if (B.connected === on) return;
    B.connected = on;
    dot.className = "conn " + (on ? "on" : "off");
    dot.title = on ? "Connected to Blender" : "Blender relay offline";
  }

  // send button enable state follows the current result
  function refresh() {
        const ready = !!app.result();
    $("sendBtn").disabled = !ready;
    const res = app.result(), src = app.source();
    const kinds = res && src && src.multi ? res.kinds : [];
    sel.hidden = kinds.length < 2;
    sel.replaceChildren(...kinds.map((k) => h("option", { value: k }, KIND_LABEL[k] || k)));
    sel.value = app.activeKind();
  }

  async function send(animated) {
    try { await (animated ? sendAnimated() : sendCurrent()); }
    catch (e) { if (e !== CANCELLED) say("send failed: " + (e.message || e)); }
  }

  // the relief preview is display-only; impasto height is opt-in (Send menu)
  const sendable = (res) => res.kinds.map((k, i) => [k, i])
    .filter(([k]) => k !== "preview" && (k !== "height" || app.sendHeight()));

  async function sendCurrent() {
    const res = await app.ensureFinal();
    if (!res) return;
    if (app.source().multi) {
      const items = sendable(res);
      const pngs = [];
      for (const [, i] of items) pngs.push(await bitmapToB64(res.bitmaps[i]));
      const kinds = items.map(([k]) => k);
      await post({ kinds, pngs, height_scale: res.heightScale });
      say(`sent ${kinds.length} maps (${kindList(kinds)}) — now click 'Apply from Isophote' inside Blender`, SENT_MS);
    } else {
      await post({ png: await bitmapToB64(res.bitmaps[0]), kind: "normal" });
      say("sent — now click 'Apply from Isophote' inside Blender", SENT_MS);
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
        say(`sent ${frames.length} frames × ${kinds.length} maps (${kindList(kinds)}) — now click 'Apply from Isophote' (Animated checked) inside Blender`, SENT_MS);
      } else {
        const anim = [];
        for (const f of frames) anim.push(await bitmapToB64(f.bitmaps[0]));
        await post({ anim });
        say(`sent ${frames.length} frames — now click 'Apply from Isophote' (Animated checked) inside Blender`, SENT_MS);
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
          say(`received ${kindList(d.kinds)} — adjust sliders freely; ALL channels repaint on Send`, 4000);
        } finally {
          B.pending = false;
        }
      }
    } catch (e) {
      setConnected(false);
      if (B.enabled) say("bridge offline — restart Blender (runs the relay on 127.0.0.1:8778)");
    } finally {
      B.inFlight = false;
    }
  }

  return {
    refresh, send,
    setEnabled(on) {
      B.enabled = on;
      section.hidden = !on;
      $("sendCaret").hidden = !on;
      clearInterval(B.timer); B.timer = null;
      if (on) { poll(); B.timer = setInterval(poll, 1500); refresh(); }
    },
    // true once the relay delivered (or is about to deliver) a Blender queue
    claimed: () => B.lastHash !== 0 || B.pending,
  };
}

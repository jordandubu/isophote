// anim.js — boil animation: paints frames on demand, caches them, loops them
// on the canvas, and hands the same frames to "Send animated". Playback
// starts as soon as two frames exist; the rest paint while it loops.
import { h, icon } from "./dom.js";
import { CANCELLED } from "./engine.js";

// hooks: paintFrame(i) -> Result, key() -> string, show(result),
//        restore() (show the still painting again), settle() -> Promise
export function createAnim({ strip, store, hooks }) {
  const A = { frames: [], key: null, job: null, token: 0, playing: false, timer: null, idx: 0, progress: new Set() };

  const playBtn = h("button", { class: "play", type: "button", title: "Play / pause (Space)", "aria-label": "Play boil preview" });
  const track = h("div", { class: "track" });
  const info = h("div", { class: "ainfo" });
  strip.append(playBtn, track, info);

  const total = () => store.values.frames;

  function render() {
    playBtn.replaceChildren(icon(A.playing ? "pause" : "play"), h("span", {}, A.playing ? "Pause" : "Boil"));
    playBtn.classList.toggle("on", A.playing);
    const n = total();
    track.replaceChildren(...Array.from({ length: n }, (_, i) =>
      h("button", { type: "button", class: "dot" + (i < A.frames.length ? " ready" : "") + (i === A.idx && A.frames.length ? " cur" : ""),
        title: "Frame " + (i + 1), "aria-label": "Frame " + (i + 1),
        onclick: () => scrub(i) })));
    const generating = A.job && A.frames.length < n;
    info.textContent = generating ? `painting ${A.frames.length + 1}/${n}…`
      : `${n} frames · ${store.values.fps} fps · boil ${store.values.boil}%`;
  }

  function stopTimer() { clearInterval(A.timer); A.timer = null; }
  function startTimer() {
    stopTimer();
    if (!A.playing || A.frames.length < 2) return;
    A.timer = setInterval(() => {
      A.idx = (A.idx + 1) % A.frames.length;
      hooks.show(A.frames[A.idx]);
      render();
    }, 1000 / store.values.fps);
  }

  function scrub(i) {
    if (i >= A.frames.length) return;
    pause();
    A.idx = i;
    hooks.show(A.frames[i]);
    render();
  }

  function ensure(onProgress) {
    const key = hooks.key();
    if (A.job && A.key === key) {
      if (onProgress) A.progress.add(onProgress);
      return A.job;
    }
    const wasPlaying = A.playing;
    invalidate(false);
    A.playing = wasPlaying;
    if (onProgress) A.progress.add(onProgress);
    A.key = key;
    const token = A.token;
    const n = total();
    const job = (async () => {
      for (let i = 0; i < n; i++) {
        const f = await hooks.paintFrame(i);
        if (token !== A.token) { f.close(); throw CANCELLED; }
        A.frames.push(f);
        if (i === 0) { A.idx = 0; hooks.show(f); }
        if (i === 1) startTimer();
        A.progress.forEach((p) => p(i + 1, n));
        render();
      }
      return A.frames;
    })();
    A.job = job;
    job.catch((e) => {
      if (token !== A.token) return;
      A.job = null; A.key = null;
      if (e !== CANCELLED) { pause(); hooks.error(e); }
    }).finally(() => { if (token === A.token) A.progress.clear(); render(); });
    render();
    return job;
  }

  function pause() { A.playing = false; stopTimer(); render(); }

  async function play() {
    A.playing = true;
    render();
    await hooks.settle();               // wait for the still painting to finish first
    if (!A.playing) return;
    ensure().catch(() => {});
    startTimer();
  }

  function invalidate(restore = true) {
    A.token++;
    stopTimer();
    A.playing = false;
    A.progress.clear();
    if (restore && A.frames.length) hooks.restore();
    A.frames.forEach((f) => f.close());
    A.frames = []; A.key = null; A.job = null; A.idx = 0;
    render();
  }

  playBtn.onclick = () => { if (A.playing) pause(); else play(); };
  render();

  return {
    ensure, invalidate, render,
    toggle: () => playBtn.click(),
    restartTimer: () => { if (A.playing) startTimer(); render(); },
    get playing() { return A.playing; },
    get hasFrames() { return A.frames.length > 0; },
  };
}

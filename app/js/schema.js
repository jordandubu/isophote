// schema.js — the single source of truth for every setting: the panel, presets,
// defaults, copy/paste JSON and the paint-vs-animation split all derive from it.

// Output size slider: absolute output long-edge in px; -1 = source size.
export const IMGSCALE_PX = [256, 512, 1024, 2048, 4096, -1];
export const IMGSCALE_LABELS = ["256px", "512px", "1024px", "2048px", "4096px", "MAX"];

// Stroke scale slider stores a symmetric position -4..4 (0 = 1x); the
// multiplier is 2^(pos/2) so 1x sits centred with equal range both ways.
export const sscaleFromPos = (pos) => Math.pow(2, pos / 2);
export const sscaleToPos = (m) => Math.round(Math.log2(m) * 2 / 0.05) * 0.05;

const fixed = (d) => (v) => (+v).toFixed(d);

export const GROUPS = [
  { id: "brush", label: "Brush" },
  { id: "surface", label: "Surface" },
  { id: "output", label: "Output" },
  { id: "anim", label: "Animation" },
];

// Impasto brushes. The position in this list is the index sent to the core:
// keep the order of BRUSHES in rust/src/pipeline.rs.
export const BRUSHES = [
  { id: "bristle", label: "Bristle" },
  { id: "knife", label: "Palette knife (flat slabs)" },
  { id: "dry", label: "Dry brush (scratchy)" },
  { id: "filbert", label: "Filbert (soft, rounded)" },
  { id: "rake", label: "Rake (coarse grooves)" },
  { id: "loaded", label: "Loaded (thick, heavy ridges)" },
];

export const SETTINGS = [
  { key: "sscale", group: "brush", label: "Stroke scale", min: -4, max: 4, step: 0.05, def: 0,
    help: "Scales every stroke dimension together. 1× is the default look.",
    fmt: (v) => sscaleFromPos(v).toFixed(2).replace(/\.?0+$/, "") + "×",
    parse: (t) => sscaleToPos(parseFloat(t)) },
  { key: "len", group: "brush", label: "Stroke length", min: 10, max: 90, step: 1, def: 40,
    help: "How far each stroke walks along the contours." },
  { key: "width", group: "brush", label: "Stroke width", min: 2, max: 30, step: 1, def: 5,
    help: "Brush thickness." },
  { key: "layers", group: "brush", label: "Layers", min: 1, max: 6, step: 1, def: 2,
    help: "Paint passes. More layers = denser coverage." },
  { key: "flow_smooth", group: "brush", label: "Flow smoothing", min: 2, max: 24, step: 1, def: 8,
    help: "Higher = strokes follow broader, more uniform directions instead of fine detail." },

  { key: "base_radius", group: "surface", label: "Underpaint smoothing", min: 2, max: 16, step: 1, def: 6,
    help: "Edge-preserving flatten of the base layer under the strokes." },
  { key: "relief", group: "surface", label: "Canvas weave", min: 0, max: 0.6, step: 0.05, def: 0.05,
    help: "Strength of the canvas texture lighting. 0 = flat.", fmt: fixed(2) },
  { key: "impasto", group: "surface", label: "Impasto (paint ridges)", min: 0, max: 2, step: 0.1, def: 0.1,
    help: "Turns strokes into thick slabs of paint that catch light. 0 = off.", fmt: fixed(2) },
  { key: "brush", group: "surface", type: "select", label: "Brush shape", options: BRUSHES, def: "knife",
    help: "The relief each stroke leaves. Shows when Impasto is above 0." },

  { key: "imgscale", group: "output", label: "Output size", min: 0, max: 5, step: 1, def: 1,
    help: "Long edge of the painted image. MAX = same size as the source.",
    fmt: (v) => IMGSCALE_LABELS[Math.min(v, IMGSCALE_LABELS.length - 1)], readonly: true },
  { key: "seed", group: "output", label: "Seed", min: 1, max: 200, step: 1, def: 42, dice: true,
    help: "Same seed + settings = the identical painting." },

  { key: "frames", group: "anim", label: "Frames", min: 2, max: 24, step: 1, def: 5,
    help: "Number of frames in the loop." },
  { key: "fps", group: "anim", label: "Speed (fps)", min: 1, max: 30, step: 1, def: 8 },
  { key: "boil", group: "anim", label: "Boil", min: 0, max: 100, step: 5, def: 30,
    help: "Share of strokes repainted each frame. 100% = a new seed per frame.",
    fmt: (v) => v + "%" },
];

export const DEFAULTS = Object.fromEntries(SETTINGS.map((s) => [s.key, s.def]));
export const PAINT_KEYS = SETTINGS.filter((s) => s.group !== "anim").map((s) => s.key);
export const ANIM_KEYS = SETTINGS.filter((s) => s.group === "anim").map((s) => s.key);

// A preset is a whole look: the keys it names, every other look key at its default.
// "default" names nothing, so it IS the defaults. Output size and seed are not part of a look.
export const LOOK_KEYS = PAINT_KEYS.filter((k) => k !== "seed" && k !== "imgscale");
export const PRESETS = {
  default: { label: "Default" },
  art:    { label: "Art",     len: 40, width: 9,  layers: 2, flow_smooth: 8, base_radius: 6, relief: 0.10, impasto: 0,  brush: "bristle" },
  medium: { label: "Medium",  len: 30, width: 4,  layers: 3, flow_smooth: 8, base_radius: 6, relief: 0.15, impasto: 0,  brush: "bristle" },
  bold:   { label: "Impasto", len: 44, width: 26, layers: 2, flow_smooth: 8, base_radius: 8, relief: 0.10, impasto: 1,  brush: "bristle" },
  pastel: { label: "Pastel",  len: 30, width: 5,  layers: 3, flow_smooth: 8, base_radius: 9, relief: 0.00, impasto: 0,  brush: "bristle" },
};

export function presetPatch(id) {
  const { label, ...patch } = PRESETS[id];
  return { ...Object.fromEntries(LOOK_KEYS.map((k) => [k, DEFAULTS[k]])), ...patch };
}

// the preset the current values match exactly, or null = custom
export function activePreset(v) {
  for (const id of Object.keys(PRESETS)) {
    const p = presetPatch(id);
    if (LOOK_KEYS.every((k) => v[k] === p[k])) return id;
  }
  return null;
}

export const byKey = Object.fromEntries(SETTINGS.map((s) => [s.key, s]));

export function clampSnap(def, x) {
  const c = Math.min(def.max, Math.max(def.min, x));
  return +(Math.round((c - def.min) / def.step) * def.step + def.min).toFixed(4);
}

// Clipboard JSON: sscale as the multiplier and imgscale as its label, as ever.
export function exportSettings(v) {
  const o = {};
  for (const s of SETTINGS) {
    if (s.group === "anim") continue;
    if (s.key === "sscale") o.sscale = sscaleFromPos(v.sscale);
    else if (s.key === "imgscale") o.imgscale = IMGSCALE_LABELS[Math.min(v.imgscale, IMGSCALE_LABELS.length - 1)];
    else o[s.key] = v[s.key];
  }
  return o;
}

export function importSettings(obj) {
  const patch = {};
  for (const s of SETTINGS) {
    if (s.group === "anim" || !(s.key in obj)) continue;
    let x = obj[s.key];
    if (s.type === "select") { if (s.options.some((o) => o.id === x)) patch[s.key] = x; continue; }
    if (s.key === "sscale") x = sscaleToPos(+x);
    else if (s.key === "imgscale") x = IMGSCALE_LABELS.indexOf(x);
    else x = +x;
    if (!Number.isFinite(x) || x < 0 && s.key === "imgscale") continue;
    patch[s.key] = clampSnap(s, x);
  }
  // settings copied before the brush list existed carry a "knife" on/off instead
  if (!("brush" in patch) && "knife" in obj) patch.brush = obj.knife ? "knife" : "bristle";
  return patch;
}

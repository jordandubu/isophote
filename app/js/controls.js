// controls.js — renders the side panel from the settings schema.
// One DOM, used as a sidebar on desktop and a bottom sheet on phones.
import { h, icon } from "./dom.js";
import { SETTINGS, GROUPS, PRESETS, PAINT_KEYS, DEFAULTS, activePreset, presetPatch, clampSnap } from "./schema.js";

export function buildPanel(root, store, { onCopy, onPaste }) {
  const syncers = [];

  function slider(def) {
    const { key } = def;
    const range = h("input", { type: "range", min: def.min, max: def.max, step: def.step, "aria-label": def.label });
    const val = h("input", { class: "val", type: "text", inputmode: "decimal", spellcheck: "false",
      readonly: def.readonly, "aria-label": def.label + " value" });
    const reset = h("button", { class: "reset", type: "button", title: "Reset to default",
      "aria-label": "Reset " + def.label, onclick: () => store.apply({ [key]: def.def }) }, icon("reset"));
    const dice = def.dice && h("button", { class: "reset dice", type: "button", title: "Random seed (R)",
      "aria-label": "Random seed",
      onclick: () => store.apply({ [key]: def.min + Math.floor(Math.random() * (def.max - def.min + 1)) }) }, icon("dice"));

    range.addEventListener("input", () => { store.begin(); store.set({ [key]: +range.value }); });
    range.addEventListener("change", () => store.end());
    val.addEventListener("change", () => {
      const x = def.parse ? def.parse(val.value) : parseFloat(val.value);
      if (Number.isFinite(x)) store.apply({ [key]: clampSnap(def, x) });
      sync();
    });
    val.addEventListener("keydown", (e) => { if (e.key === "Enter") val.blur(); });

    const el = h("div", { class: "slider", title: def.help },
      h("div", { class: "row" },
        h("label", {}, def.label),
        h("span", { class: "vals" }, dice, reset, val)),
      range);
    function sync() {
      const x = store.values[key];
      range.value = x;
      val.value = def.fmt ? def.fmt(x) : String(x);
      el.classList.toggle("mod", x !== def.def);
    }
    syncers.push(sync);
    return el;
  }

  function select(def) {
    const sel = h("select", { class: "chsel", "aria-label": def.label,
      onchange: () => store.apply({ [def.key]: sel.value }) },
      def.options.map((o) => h("option", { value: o.id }, o.label)));
    syncers.push(() => { sel.value = store.values[def.key]; });
    return h("div", { class: "slider", title: def.help }, h("div", { class: "row" }, h("label", {}, def.label)), sel);
  }

  function section(id, title, kids, open = true) {
    const d = h("details", { class: "sec", id: "sec-" + id, open }, h("summary", {}, title), h("div", { class: "secbody" }, kids));
    return d;
  }

  root.append(h("div", { id: "blenderSlot" }));

  const chips = Object.entries(PRESETS).map(([id, p]) =>
    h("button", { class: "chip", type: "button", "data-p": id, onclick: () => store.apply(presetPatch(id)) }, p.label));
  // no preset matches = custom: selected automatically, nothing to click
  const customChip = h("span", { class: "chip static", title: "The settings differ from every preset" }, "Custom");
  chips.push(customChip);
  syncers.push(() => {
    const on = activePreset(store.values);
    chips.forEach((c) => c.classList.toggle("on", c === customChip ? on === null : c.dataset.p === on));
  });
  const resetAll = () => store.apply(Object.fromEntries(PAINT_KEYS.filter((k) => k !== "seed").map((k) => [k, DEFAULTS[k]])));
  root.append(section("look", "Look", [
    h("div", { class: "chips" }, chips),
    h("div", { class: "btnrow" },
      h("button", { class: "ghost", type: "button", onclick: onCopy, title: "Copy all settings to the clipboard" }, "Copy"),
      h("button", { class: "ghost", type: "button", onclick: onPaste, title: "Paste settings from the clipboard (or press Ctrl+V)" }, "Paste"),
      h("button", { class: "ghost", type: "button", onclick: resetAll, title: "Back to default settings (keeps the seed)" }, "Reset")),
  ]));

  for (const g of GROUPS) {
    const kids = SETTINGS.filter((s) => s.group === g.id).map((s) => (s.type === "select" ? select(s) : slider(s)));
    root.append(section(g.id, g.label, kids, g.id !== "anim"));
  }

  const refresh = () => syncers.forEach((f) => f());
  store.subscribe(refresh);
  refresh();
}

// store.js — settings values + change subscription + undo/redo.
// A "gesture" (slider drag, preset click, paste…) is one undo step:
// begin() snapshots, end() commits the snapshot if anything changed.
export function createStore(defaults) {
  let v = { ...defaults };
  const subs = new Set();
  const undoStack = [], redoStack = [];
  let gesture = null;
  const diff = (a, b) => Object.keys(b).filter((k) => a[k] !== b[k]);
  const emit = (changed) => { for (const f of subs) f(changed); };
  const jump = (snap) => {
    const changed = diff(v, snap);
    v = { ...snap };
    if (changed.length) emit(changed);
  };

  const api = {
    get values() { return v; },
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
    set(patch) {
      const changed = Object.keys(patch).filter((k) => v[k] !== patch[k]);
      if (!changed.length) return;
      v = { ...v, ...patch };
      emit(changed);
    },
    begin() { if (!gesture) gesture = { ...v }; },
    end() {
      if (!gesture) return;
      if (diff(gesture, v).length) {
        undoStack.push(gesture);
        if (undoStack.length > 100) undoStack.shift();
        redoStack.length = 0;
      }
      gesture = null;
    },
    apply(patch) { api.begin(); api.set(patch); api.end(); },
    undo() {
      api.end();
      if (!undoStack.length) return;
      redoStack.push({ ...v });
      jump(undoStack.pop());
    },
    redo() {
      api.end();
      if (!redoStack.length) return;
      undoStack.push({ ...v });
      jump(redoStack.pop());
    },
  };
  return api;
}

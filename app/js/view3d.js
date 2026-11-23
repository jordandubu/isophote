// view3d.js — WebGL2 preview: the Blender mesh wearing the painted maps.
// Drag orbits; Shift-drag / middle / right button pans; wheel or pinch zooms;
// double-click resets the view. Renders on demand (no animation loop).

const VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUv;
uniform mat4 uViewProj;
out vec3 vPos;
out vec3 vNrm;
out vec2 vUv;
void main() {
  vPos = aPos; vNrm = aNrm; vUv = aUv;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;

// uHas bits: 1 color, 2 normal, 4 rough, 8 metal, 16 orm, 32 emission, 64 height
const FS = `#version 300 es
precision highp float;
in vec3 vPos;
in vec3 vNrm;
in vec2 vUv;
uniform sampler2D uColor, uNormal, uRough, uMetal, uOrm, uEmis, uHeight, uFlat;
uniform int uHas;
uniform int uShowFlat;  // 1: show uFlat alone, unlit (a single channel picked in the dropdown)
uniform int uTangent;   // 1: the normal map is tangent-space, 0: object-space
uniform mat3 uObjN;     // object -> world normal matrix
uniform float uBump;    // world height of a white height texel
uniform vec3 uEye;
uniform vec3 uLight;
out vec4 o;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);                // bitmaps upload top row first
  if (uShowFlat == 1) { o = vec4(texture(uFlat, uv).rgb, 1.0); return; }
  vec3 N = normalize(vNrm);
  if (!gl_FrontFacing) N = -N;
  if ((uHas & 2) != 0) {
    vec3 t = texture(uNormal, uv).xyz * 2.0 - 1.0;
    if (uTangent == 1) {                              // cotangent frame from derivatives (B follows +v)
      vec3 dp1 = dFdx(vPos), dp2 = dFdy(vPos);
      vec2 d1 = dFdx(vUv), d2 = dFdy(vUv);
      vec3 a = cross(dp2, N), b = cross(N, dp1);
      vec3 T = a * d1.x + b * d2.x, B = a * d1.y + b * d2.y;
      float inv = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
      N = normalize(mat3(T * inv, B * inv, N) * t);
    } else {
      N = normalize(uObjN * t);
    }
  }
  if ((uHas & 64) != 0) {                             // derivative bump, after the normal map like the Blender chain
    float hgt = texture(uHeight, uv).r * uBump;
    vec3 sx = dFdx(vPos), sy = dFdy(vPos);
    vec3 r1 = cross(sy, N), r2 = cross(N, sx);
    float det = dot(sx, r1);
    N = normalize(abs(det) * N - sign(det) * (dFdx(hgt) * r1 + dFdy(hgt) * r2));
  }
  vec3 base = (uHas & 1) != 0 ? texture(uColor, uv).rgb : vec3(0.42);
  float rough = (uHas & 4) != 0 ? texture(uRough, uv).r : (uHas & 16) != 0 ? texture(uOrm, uv).g : 0.5;
  float metal = (uHas & 8) != 0 ? texture(uMetal, uv).r : (uHas & 16) != 0 ? texture(uOrm, uv).b : 0.0;
  rough = clamp(rough, 0.05, 1.0);
  vec3 V = normalize(uEye - vPos), L = uLight, H = normalize(L + V);
  float nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3), nh = max(dot(N, H), 0.0), vh = max(dot(V, H), 0.0);
  float a2 = pow(rough, 4.0), dd = nh * nh * (a2 - 1.0) + 1.0, D = a2 / (3.14159 * dd * dd);
  float k = (rough + 1.0) * (rough + 1.0) / 8.0;
  float G = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
  vec3 F0 = mix(vec3(0.04), base, metal), F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
  vec3 col = (base * (1.0 - metal) / 3.14159 + D * G * F / (4.0 * nl * nv + 1e-4)) * nl * 3.0;
  vec3 lo = vec3(0.10, 0.09, 0.08), hi = vec3(0.34, 0.36, 0.40);   // ground / sky, Z up
  col += base * (1.0 - metal) * mix(lo, hi, N.z * 0.5 + 0.5);
  col += F0 * mix(lo, hi, reflect(-V, N).z * 0.5 + 0.5) * (1.0 - 0.5 * rough);
  if ((uHas & 32) != 0) col += texture(uEmis, uv).rgb;
  o = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}`;

// [result kind, uHas bit, sRGB texture, sampler uniform]; texture unit = index
const SLOTS = [
  ["color", 1, true, "uColor"], ["normal", 2, false, "uNormal"], ["rough", 4, false, "uRough"],
  ["metal", 8, false, "uMetal"], ["orm", 16, false, "uOrm"], ["emission", 32, true, "uEmis"],
  ["height", 64, false, "uHeight"],
];
const FOV = 35 * Math.PI / 180;
const IMPASTO_DEPTH = 0.5;    // same factor as the add-on's Bump distance

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// column-major 4x4 product a * b
function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

// inverse-transpose of a row-major 3x3, returned column-major for uniformMatrix3fv
function normalMatrix(m) {
  const c = [
    m[4] * m[8] - m[5] * m[7], m[5] * m[6] - m[3] * m[8], m[3] * m[7] - m[4] * m[6],
    m[2] * m[7] - m[1] * m[8], m[0] * m[8] - m[2] * m[6], m[1] * m[6] - m[0] * m[7],
    m[1] * m[5] - m[2] * m[4], m[2] * m[3] - m[0] * m[5], m[0] * m[4] - m[1] * m[3],
  ];                                                    // cofactors, row-major
  const det = m[0] * c[0] + m[1] * c[1] + m[2] * c[2];
  if (!(Math.abs(det) > 1e-12)) return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  return new Float32Array([c[0], c[3], c[6], c[1], c[4], c[7], c[2], c[5], c[8]].map((v) => v / det));
}

const STUB = { supported: false, setMesh() {}, show() {}, resetView() {} };

export function createView3d(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) return STUB;

  function shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  } catch (e) {
    console.error("3D preview disabled:", e);
    return STUB;
  }
  gl.useProgram(prog);
  const U = {};
  for (const n of ["uViewProj", "uHas", "uShowFlat", "uTangent", "uObjN", "uBump", "uEye", "uLight"]) U[n] = gl.getUniformLocation(prog, n);
  const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
  // 1x1 until a result is shown (every sampler stays complete)
  const makeTex = (name, unit) => {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT));
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.uniform1i(gl.getUniformLocation(prog, name), unit);
    return t;
  };
  // one texture per slot, plus the single channel shown unlit
  const tex = SLOTS.map(([, , , name], unit) => makeTex(name, unit));
  const flatTex = makeTex("uFlat", SLOTS.length);
  gl.enable(gl.DEPTH_TEST);

  let vao = null, bufs = [], count = 0;
  let center = [0, 0, 0], radius = 1, uvWorld = 1, objN = normalMatrix([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let target = [0, 0, 0], yaw = 0, pitch = 0, dist = 1;
  let shown = null, has = 0, heightScale = 0, tangent = false, useHeight = false;
  let flatKind = null, showFlat = false;
  let raf = 0;

  const dirty = () => { if (!raf) raf = requestAnimationFrame(render); };

  // camera basis: eye, forward, right, up (world Z is up)
  function camera() {
    const cp = Math.cos(pitch);
    const eye = [target[0] + dist * cp * Math.cos(yaw), target[1] + dist * cp * Math.sin(yaw), target[2] + dist * Math.sin(pitch)];
    const f = norm(sub(target, eye));
    const r = norm(cross(f, [0, 0, 1]));
    return { eye, f, r, u: cross(r, f) };
  }

  function render() {
    raf = 0;
    const w = canvas.width, h = canvas.height;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.11, 0.11, 0.12, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!vao || !w || !h) return;
    const { eye, f, r, u } = camera();
    const view = new Float32Array([
      r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0,
      -dot(r, eye), -dot(u, eye), dot(f, eye), 1,
    ]);
    const near = Math.max(dist - 2 * radius, radius * 1e-3), far = dist + 2 * radius;
    const t = 1 / Math.tan(FOV / 2);
    const proj = new Float32Array([
      t / (w / h), 0, 0, 0, 0, t, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0,
    ]);
    gl.uniformMatrix4fv(U.uViewProj, false, mul4(proj, view));
    gl.uniform1i(U.uHas, useHeight && !tangent ? has : has & ~64);
    gl.uniform1i(U.uShowFlat, showFlat ? 1 : 0);
    gl.uniform1i(U.uTangent, tangent ? 1 : 0);
    gl.uniformMatrix3fv(U.uObjN, false, objN);
    gl.uniform1f(U.uBump, uvWorld * heightScale * IMPASTO_DEPTH);
    gl.uniform3fv(U.uEye, eye);
    gl.uniform3fv(U.uLight, norm([-0.5 * r[0] + 0.6 * u[0] - 0.65 * f[0], -0.5 * r[1] + 0.6 * u[1] - 0.65 * f[1], -0.5 * r[2] + 0.6 * u[2] - 0.65 * f[2]]));
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, count);
  }

  function resetView() {
    target = center.slice();
    yaw = -Math.PI / 2 + 0.5;
    pitch = 0.3;
    dist = radius / Math.sin(FOV / 2) * 1.1;
    dirty();
  }

  // mesh: parseMesh() result (engine.js) or null
  function setMesh(mesh) {
    if (vao) { gl.deleteVertexArray(vao); bufs.forEach((b) => gl.deleteBuffer(b)); }
    vao = null; bufs = []; count = 0; shown = null; has = 0; showFlat = false;
    if (!mesh) { dirty(); return; }
    const n = mesh.ntri * 3;
    const pos = new Float32Array(n * 3);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        const v = mesh.vpos[mesh.triVert[i] * 3 + c];
        pos[i * 3 + c] = v;
        if (v < lo[c]) lo[c] = v;
        if (v > hi[c]) hi[c] = v;
      }
    }
    // world units per UV unit (same measure as the add-on's Bump distance)
    let area3 = 0, areaUv = 0;
    for (let t = 0; t < mesh.ntri; t++) {
      const p = (k) => [pos[t * 9 + k * 3], pos[t * 9 + k * 3 + 1], pos[t * 9 + k * 3 + 2]];
      const c = cross(sub(p(1), p(0)), sub(p(2), p(0)));
      area3 += Math.hypot(c[0], c[1], c[2]);
      const q = mesh.triUv, o = t * 6;
      areaUv += Math.abs((q[o + 2] - q[o]) * (q[o + 5] - q[o + 1]) - (q[o + 4] - q[o]) * (q[o + 3] - q[o + 1]));
    }
    uvWorld = areaUv > 0 ? Math.sqrt(area3 / areaUv) : 1;
    center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    radius = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 1e-6);
    objN = normalMatrix(mesh.matrix);
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    [[pos, 3], [mesh.triNrm, 3], [mesh.triUv, 2]].forEach(([data, size], loc) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      bufs.push(b);
    });
    count = n;
    resetView();
  }

  // result: engine.js Result; uploads its bitmaps now (the caller may close them afterwards).
  // opts: { kind: channel picked in the dropdown, tangent: the normal map is tangent-space,
  //         useHeight: bump from the impasto height }
  // The reference channel (result.kinds[0], top of the dropdown) shows the lit material built
  // from every map; any other channel is shown alone, unlit, exactly as on the 2D canvas.
  function show(result, opts) {
    if (result !== shown) {
      shown = result;
      has = 0;
      SLOTS.forEach(([kind, bit, srgb], unit) => {
        const i = result.kinds.indexOf(kind);
        if (i < 0) return;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex[unit]);
        gl.texImage2D(gl.TEXTURE_2D, 0, srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, result.bitmaps[i]);
        gl.generateMipmap(gl.TEXTURE_2D);
        has |= bit;
      });
      heightScale = result.heightScale || 0;
      flatKind = null;
    }
    tangent = !!opts.tangent;
    useHeight = !!opts.useHeight;
    const flat = opts.kind !== result.kinds[0] && result.kinds.includes(opts.kind) ? opts.kind : null;
    if (flat && flat !== flatKind) {
      gl.activeTexture(gl.TEXTURE0 + SLOTS.length);
      gl.bindTexture(gl.TEXTURE_2D, flatTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, result.bitmap(flat));
      gl.generateMipmap(gl.TEXTURE_2D);
      flatKind = flat;
    }
    showFlat = !!flat;
    dirty();
  }

  /* ---- input ---- */
  const ptrs = new Map();
  let pinch = 0;
  const pinchDist = () => { const [a, b] = [...ptrs.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };
  const zoom = (factor) => { dist = Math.min(20 * radius, Math.max(0.05 * radius, dist * factor)); dirty(); };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 1 || e.button === 2 });
    if (ptrs.size === 2) pinch = pinchDist();
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = ptrs.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (ptrs.size === 2) {
      const d = pinchDist();
      if (pinch > 0 && d > 0) zoom(pinch / d);
      pinch = d;
    } else if (p.pan) {
      const { r, u } = camera();
      const k = 2 * dist * Math.tan(FOV / 2) / (canvas.clientHeight || 1);
      for (let c = 0; c < 3; c++) target[c] += (-dx * r[c] + dy * u[c]) * k;
      dirty();
    } else {
      yaw -= dx * 0.008;
      pitch = Math.min(1.55, Math.max(-1.55, pitch + dy * 0.008));
      dirty();
    }
  });
  const up = (e) => { ptrs.delete(e.pointerId); };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", resetView);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
    dirty();
  }).observe(canvas);

  return { supported: true, setMesh, show, resetView };
}

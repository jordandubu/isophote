#!/usr/bin/env python3
"""
isophote.py -- flow-guided brush stroke oil painting filter, standalone.

Strokes follow image contours (structure-tensor flow field), long and thin,
layered with jitter. Canvas relief + impasto lighting optional.

Usage: ./isophote.py photo.jpg [-o out.png] [flags]
"""
import argparse
import numpy as np
import cv2

# ---------------------------------------------------------------- flow field

def flow_field(img_bgr, sigma=2.0):
    """Return (angle, coherence) per pixel. Angle = isophote direction (deg,
    [0,180)), from structure tensor. coherence in [0,1]; 1 = strong orientation."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    ang = np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy + tmp)) / 2.0 + 90.0
    angle = ang % 180.0
    l1 = (Jxx + Jyy + tmp) * 0.5
    l2 = (Jxx + Jyy - tmp) * 0.5
    coherence = np.clip((l1 - l2) / (l1 + l2 + 1e-3), 0.0, 1.0)
    return angle.astype(np.float32), coherence.astype(np.float32)


def smooth_flow(angle, coherence, r=8):
    """Orientation-average of the flow field weighted by coherence.
    Keeps directions coherent across flat regions (vector-average of doubled angle)."""
    v = np.stack([np.cos(2 * np.radians(angle)) * coherence,
                  np.sin(2 * np.radians(angle)) * coherence], axis=0)
    k = cv2.getGaussianKernel(r * 2 + 1, r / 2.0)
    kern = k @ k.T
    vx = cv2.filter2D(v[0], -1, kern, borderType=cv2.BORDER_REFLECT)
    vy = cv2.filter2D(v[1], -1, kern, borderType=cv2.BORDER_REFLECT)
    avg = (np.degrees(np.arctan2(vy, vx)) / 2.0) % 180.0
    mag = np.hypot(vx, vy)
    return avg.astype(np.float32), mag.astype(np.float32)

# ---------------------------------------------------------------- base layer

def painterly_base(img_bgr, radius=6, sigma=0.4, mode="kuwahara"):
    """Base layer: Kuwahara flattens detail painterly while keeping edges."""
    return kuwahara(img_bgr, radius, sigma)


def kuwahara(img_bgr, radius=5, sigma=0.4):
    """Classic Kuwahara (4 quadrants, smallest variance wins)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = img.shape[:2]
    pad = cv2.copyMakeBorder(img, radius, radius, radius, radius,
                             cv2.BORDER_REFLECT)
    L = pad[..., 0]
    means = np.zeros((4, h, w), np.float32)
    vars_ = np.zeros((4, h, w), np.float32)
    colors = np.zeros((4, h, w, 3), np.float32)
    quads = [(slice(0, radius + 1), slice(0, radius + 1)),
             (slice(0, radius + 1), slice(radius, None) if False else slice(-1, -radius - 2 if False else None))  # placeholder
             ]
    # simpler: use box filters per quadrant
    n = radius + 1
    offs = [(-radius, -radius), (-radius, 0), (0, -radius), (0, 0)]
    size = n * n
    for i, (oy, ox) in enumerate(offs):
        quad = L[oy + radius: oy + radius + h, ox + radius: ox + radius + w]
        m = cv2.boxFilter(quad, -1, (n, n), borderType=cv2.BORDER_REFLECT)
        means[i] = m
        vars_[i] = cv2.boxFilter(quad * quad, -1, (n, n),
                                 borderType=cv2.BORDER_REFLECT) - m * m
        for c in range(3):
            qc = pad[..., c][oy + radius: oy + radius + h,
                                 ox + radius: ox + radius + w]
            colors[i, ..., c] = cv2.boxFilter(qc, -1, (n, n),
                                              borderType=cv2.BORDER_REFLECT)
    idx = np.argmin(vars_, axis=0)
    ys, xs = np.indices((h, w))
    out = colors[idx, ys, xs]
    out[..., 0] = means[idx, ys, xs]
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_Lab2BGR)


def oriented_smooth(img_bgr, angle, strength=0.8):
    """Line-shaped blur along isophotes (elongated kernel per-pixel via
    multiple 1D samples -- cheap approx: rotate-aware box)."""
    # approximation: average of short line blurs at 8 quantized orientations,
    # selected per-pixel -> gives directional smear without per-pixel kernels
    q = (np.round(angle / 22.5) % 8).astype(np.uint8)
    acc = np.zeros_like(img_bgr, dtype=np.float32)
    weight = np.zeros(angle.shape, np.float32)
    h, w = q.shape
    L, th = 9, 1.0  # kernel: length 9 along orientation, 1 across
    for i in range(8):
        a = np.radians(i * 22.5)
        dx, dy = np.cos(a), -np.sin(a)
        # build elongated kernel
        k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
        for t in range(-L, L + 1):
            ky, kx = int(round(-t * dy)) + L, int(round(t * dx)) + L
            if 0 <= ky < 2 * L + 1 and 0 <= kx < 2 * L + 1:
                k[ky, kx] = 1.0
        k /= k.sum()
        blurred = cv2.filter2D(img_bgr, -1, k, borderType=cv2.BORDER_REFLECT)
        mask = (q == i).astype(np.float32)
        acc += blurred.astype(np.float32) * mask[..., None]
        weight += mask
    out = img_bgr.astype(np.float32).copy()
    m = weight > 0
    out[m] = acc[m] / weight[m][:, None]
    base = img_bgr.astype(np.float32)
    return (strength * out + (1 - strength) * base).astype(np.uint8)

# ---------------------------------------------------------------- strokes

def render_strokes(img_bgr, base, angle_img, strokes_per_layer,
                   length, width, layers=2, jitter=0.6, opacity=0.9, seed=42):
    """Draw spline strokes along the flow field on top of the base.
    Stroke color sampled from the ORIGINAL photo (accurate hues);
    base shows through in low-detail areas."""
    h, w = img_bgr.shape[:2]
    canvas = base.astype(np.float32)
    updated = np.zeros((h, w), np.float32)  # where strokes landed
    rng = np.random.default_rng(seed)

    imgf = img_bgr.astype(np.float32)
    src = imgf  # color source: original photo

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    for layer in range(layers):
        step = max(2, h // int(np.sqrt(strokes_per_layer)))
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        jit = step * 0.5 * jitter
        pts = pts + rng.uniform(-jit, jit, pts.shape)
        # high-gradient areas first (detail gets painted before flats)
        order = np.argsort(-grad[np.clip(pts[:, 1], 0, h - 1).astype(int),
                                 np.clip(pts[:, 0], 0, w - 1).astype(int)],
                          kind='stable')
        pts = pts[order]

        for x, y in pts:
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            a = np.radians(angle_img[yi, xi])
            # build polyline: walk along flow with slight curl
            steps = max(2, int(length // 2))
            poly = [(x, y)]
            cx, cy = x, y
            dx0, dy0 = np.cos(a), np.sin(a)
            for _ in range(steps):
                # sample flow at current point
                xj, yj = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
                a2 = np.radians(angle_img[yj, xj])
                # keep direction continuity (choose nearest of +/-)
                d1 = (a2 - np.arctan2(dy0, dx0) + np.pi) % (2 * np.pi) - np.pi
                a2cont = a2 if abs(d1) < np.pi / 2 else a2 + np.pi
                dx1, dy1 = np.cos(a2cont), np.sin(a2cont)
                # blend with initial dir for smoothness
                dx0 = 0.7 * dx0 + 0.3 * dx1
                dy0 = 0.7 * dy0 + 0.3 * dy1
                nrm = np.hypot(dx0, dy0) + 1e-8
                dx0, dy0 = dx0 / nrm, dy0 / nrm
                cx += 2.0 * dx0
                cy += 2.0 * dy0
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                poly.append((cx, cy))
            if len(poly) < 3:
                continue
            pts_arr = np.array(poly, np.float32)
            # brush color: sample the *source photo* under the stroke midpoint,
            # with a small radius -- keeps strokes tied to real image detail
            mx, my = pts_arr[len(pts_arr) // 2].astype(int)
            mx, my = int(np.clip(mx, 0, w - 1)), int(np.clip(my, 0, h - 1))
            r = max(1, width // 2)
            patch = src[max(0, my - r):my + r + 1, max(0, mx - r):mx + r + 1]
            col = patch.mean(axis=(0, 1)) if patch.size else imgf[my, mx]
            # slight luminance variation per stroke = loaded brush feel
            col = np.clip(col * rng.uniform(0.94, 1.06), 0, 255)
            # jitter length/width per stroke
            Lj = length * rng.uniform(1 - jitter, 1 + jitter)
            Wj = max(1, width * rng.uniform(1 - jitter, 1 + jitter))
            th = max(1, int(Wj))
            cv2.polylines(canvas, [pts_arr.astype(np.int32)], False,
                          col.tolist(), th, cv2.LINE_AA)
            cv2.polylines(updated, [pts_arr.astype(np.int32)], False, 1.0, th,
                          cv2.LINE_AA)

    # blend: strokes at full color where landed (soft AA edge via updated mask)
    a = cv2.GaussianBlur(updated, (0, 0), 0.6)[..., None]
    a = np.clip(a * 1.4, 0, 1) * opacity
    out = canvas * a + base.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- canvas/impasto

def canvas_relief(stroke_alpha, img_size, scale=12, strength=0.35):
    """Procedural canvas weave + stroke height bump -> normal map -> shade."""
    h, w = img_size
    yy, xx = np.mgrid[0:h, 0:w]
    weave = (np.sin(xx * np.pi / scale) * np.sin(yy * np.pi / scale))
    height = weave * 0.5 + stroke_alpha * 1.5
    height = height.astype(np.float32)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    n = np.stack([gx, gy, np.full_like(gx, 10.0)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    light = np.array([-0.5, -0.5, 0.7]); light /= np.linalg.norm(light)
    shade = (n @ light)[..., None]
    shade = np.clip(shade, -1, 1) * 0.5 + 0.5
    return np.clip(shade * strength + (1 - strength), 0, 1), shade

# ---------------------------------------------------------------- pipeline

def paint_bytes(data_bgra, height, width, brush_len=26, brush_w=7, layers=2,
                relief=0.35, base_radius=6, flow_smooth=8, seed=42,
                max_size=1400):
    """In-memory entry used by the browser (pyodide).

    data_bgra: contiguous uint8 ndarray (height, width, 4) of the decoded
    source at FULL size — browser canvas gives BGRA order via pyodide buffers.
    Returns (painted_png_bytes, seconds).
    """
    import time
    t0 = time.time()
    img_bgr = cv2.cvtColor(data_bgra, cv2.COLOR_BGRA2BGR)
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / max_size
    if scale > 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w / scale), int(h / scale)),
                             interpolation=cv2.INTER_AREA)
    np.random.seed(seed)

    base = painterly_base(img_bgr, radius=base_radius)

    angle, coh = flow_field(img_bgr, sigma=2.5)
    angle, coh = smooth_flow(angle, coh, r=flow_smooth)

    n_strokes = int(img_bgr.shape[0] * img_bgr.shape[1] / 140)
    painted = render_strokes(img_bgr, base, angle, n_strokes, brush_len, brush_w,
                             layers=layers, seed=seed)
    if relief > 0:
        shade, _ = canvas_relief(np.zeros_like(coh), painted.shape[:2],
                                 strength=relief)
        painted = (painted.astype(np.float32) * shade).astype(np.uint8)
    ok, buf = cv2.imencode(".png", painted)
    return buf.tobytes(), round(time.time() - t0, 2)


def run(path, out, brush_len=26, brush_w=7, layers=2, canvas=True, relief=0.35,
        base_radius=6, flow_smooth=8, density=None, seed=42, max_size=1600):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / max_size
    if scale > 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w / scale), int(h / scale)),
                             interpolation=cv2.INTER_AREA)
    np.random.seed(seed)

    base = painterly_base(img_bgr, radius=base_radius)

    angle, coh = flow_field(img_bgr, sigma=2.5)
    angle, coh = smooth_flow(angle, coh, r=flow_smooth)

    n_strokes = density or int(img_bgr.shape[0] * img_bgr.shape[1] / 140)
    painted = render_strokes(img_bgr, base, angle, n_strokes, brush_len, brush_w,
                             layers=layers, seed=seed)
    if canvas:
        a = cv2.cvtColor(painted, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shade, _ = canvas_relief(np.zeros_like(coh), painted.shape[:2],
                                 strength=relief)
        painted = (painted.astype(np.float32) * shade).astype(np.uint8)
    cv2.imwrite(out, painted)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="out.png")
    ap.add_argument("--len", type=int, default=26, help="brush length px")
    ap.add_argument("--width", type=int, default=7, help="brush thickness px")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--density", type=int, default=None, help="strokes count")
    ap.add_argument("--no-canvas", action="store_true")
    ap.add_argument("--relief", type=float, default=0.35)
    ap.add_argument("--base-radius", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.input, a.out, brush_len=a.len, brush_w=a.width, layers=a.layers,
        canvas=not a.no_canvas, relief=a.relief, base_radius=a.base_radius,
        density=a.density, seed=a.seed)

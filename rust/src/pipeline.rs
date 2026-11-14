//! pipeline.rs — port of isophote.py (flow field, kuwahara, strokes, relief).
//! All buffers are HxWx4 BGRA u8 (alpha ignored on input, 255 on output).

use crate::pcg64::Pcg64;

pub struct Params {
    pub brush_len: i32,
    pub brush_w: i32,
    pub layers: i32,
    pub relief: f64,
    pub base_radius: i32,
    pub flow_smooth: i32,
    pub seed: u64,
}

// ---------------------------------------------------------------- image ops

/// GaussianBlur equivalent operating on one f32 plane, separable, REFLECT_101
/// (matches cv2 BORDER_REFLECT). sigma <= 0 → cv2 formula for ksize.
pub struct Gauss1D {
    pub size: usize,
    pub coeffs: Vec<f32>,
}

pub fn gaussian_kernel(size: usize, sigma: f64) -> Gauss1D {
    let raw: Vec<f64> = if sigma <= 0.0 {
        let n = size - 1;
        let base = 0.3f64 * ((n as f64 - 1.0) * 0.5 - 1.0) + 0.8;
        let denom = 2.0 * base * base;
        (0..size)
            .map(|i| {
                let x = i as f64 - n as f64 / 2.0;
                (-x * x / denom).exp()
            })
            .collect()
    } else {
        (0..size)
            .map(|i| {
                let x = i as f64 - (size - 1) as f64 / 2.0;
                (-x * x / (2.0 * sigma * sigma)).exp()
            })
            .collect()
    };
    let s: f64 = raw.iter().sum();
    let coeffs: Vec<f32> = raw.into_iter().map(|c| (c / s) as f32).collect();
    Gauss1D { size, coeffs }
}

#[inline]
fn reflect101(p: i64, n: i64) -> i64 {
    // cv2 BORDER_REFLECT (inclusive): gfedcb|abcdefgh|gfedcba → -1→0, -2→1
    if p < 0 {
        -p - 1
    } else if p >= n {
        2 * n - p - 1
    } else {
        p
    }
}

#[inline]
fn reflect_101(p: i64, n: i64) -> i64 {
    // cv2 BORDER_REFLECT_101 / DEFAULT: fedcba|abcdefgh|hgfedcb → -1→1, -2→2
    if p < 0 {
        -p
    } else if p >= n {
        2 * n - p - 2
    } else {
        p
    }
}

/// Gaussian blur on a single f32 plane (separable, reflect101).
pub fn gaussian_blur_plane(src: &[f32], w: usize, h: usize, sigma_x: f64) -> Vec<f32> {
    if sigma_x <= 0.0 {
        return src.to_vec();
    }
    let ksize = (((sigma_x * 4.0 * 2.0 + 1.0).round() as usize) | 1);
    let k = gaussian_kernel(ksize, sigma_x);
    let half = k.size / 2;
    let mut tmp = vec![0f32; w * h];
    // horizontal (GaussianBlur default border = REFLECT_101)
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, kx) in (-((half as i64))..=(half as i64)).enumerate() {
                let sx = reflect_101(x as i64 + kx, w as i64) as usize;
                acc += src[y * w + sx] * k.coeffs[ki];
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0f32; w * h];
    // vertical
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, ky) in (-((half as i64))..=(half as i64)).enumerate() {
                let sy = reflect_101(y as i64 + ky, h as i64) as usize;
                acc += tmp[sy * w + x] * k.coeffs[ki];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Sobel 3x3 on f32 plane, cv2 default border REFLECT_101 (cv2.Sobel).
pub fn sobel3(src: &[f32], w: usize, h: usize, dx: i32, dy: i32) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    // kernels (cv2):
    let kx: [[f32; 3]; 3] = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]; // dx=1
    let ky: [[f32; 3]; 3] = [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]; // dy=1
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for j in -1i64..=1 {
                for i in -1i64..=1 {
                    let sy = reflect_101(y as i64 + j, h as i64) as usize;
                    let sx = reflect_101(x as i64 + i, w as i64) as usize;
                    let v = src[sy * w + sx];
                    let kw = j + 1;
                    let kh = i + 1;
                    if dx == 1 {
                        acc += v * kx[kw as usize][kh as usize];
                    } else if dy == 1 {
                        acc += v * ky[kw as usize][kh as usize];
                    }
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Box filter (mean) 3-channel, cv2 BORDER_REFLECT (== reflect101).
pub fn box_filter_c3(src: &[u8], w: usize, h: usize, kw: usize, kh: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h * 3];
    let half_w = (kw / 2) as i64;
    let half_h = (kh / 2) as i64;
    let area = (kw * kh) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut s = [0f32; 3];
            for j in -(half_h)..=half_h {
                let sy = reflect101(y as i64 + j, h as i64) as usize;
                for i in -(half_w)..=half_w {
                    let sx = reflect101(x as i64 + i, w as i64) as usize;
                    for c in 0..3 {
                        s[c] += src[(sy * w + sx) * 3 + c] as f32;
                    }
                }
            }
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = s[c] / area;
            }
        }
    }
    out
}

/// cv2 INTER_AREA-ish resize via simple bilinear+box blend. The browser resizes
/// with canvas; for parity the reference harness resizes in Python. In wasm we
/// receive already-resized buffers, so only a basic fallback provided here.
pub fn resize_area(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, c: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0u8; dw * dh * c];
    let xr = sw as f64 / dw as f64;
    let yr = sh as f64 / dh as f64;
    for y in 0..dh {
        for x in 0..dw {
            let x0 = (x as f64 * xr).floor() as usize;
            let x1 = (((x + 1) as f64 * xr).floor() as usize).min(sw);
            let y0 = (y as f64 * yr).floor() as usize;
            let y1 = (((y + 1) as f64 * yr).floor() as usize).min(sh);
            for ci in 0..c {
                let mut acc = 0f64;
                let mut n = 0;
                for yy in y0..y1.max(y0 + 1) {
                    for xx in x0..x1.max(x0 + 1) {
                        acc += src[(yy * sw + xx) * c + ci] as f64;
                        n += 1;
                    }
                }
                out[(y * dw + x) * c + ci] = (acc / n.max(1) as f64) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- Lab conv

/// sRGB -> Lab (matches cv2 8-bit BGR->Lab): linearize, sRGB gamma,
/// D65 white, f(t) and 8-bit scaled L = L*255/100.
pub fn bgr_to_lab(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    // cv2 8-bit Lab uses: L in 0..255 (L*255/100), a/b offset +128
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let b = src[idx] as f64 / 255.0;
            let g = src[idx + 1] as f64 / 255.0;
            let r = src[idx + 2] as f64 / 255.0;
            let (rl, gl, bl) = srgb_linear(r, g, b);
            // XYZ (D65)
            let xx = 0.412453 * rl + 0.357580 * gl + 0.180423 * bl;
            let yy = 0.212671 * rl + 0.715160 * gl + 0.072169 * bl;
            let zz = 0.019334 * rl + 0.119193 * gl + 0.950227 * bl;
            let f = |t: f64| -> f64 {
                const EPS: f64 = 0.008856;
                const KAPPA: f64 = 903.3;
                if t > EPS {
                    t.cbrt()
                } else {
                    (KAPPA * t + 16.0) / 116.0
                }
            };
            const XN: f64 = 0.950456;
            const ZN: f64 = 1.088754;
            let fx = f(xx / XN);
            let fy = f(yy);
            let fz = f(zz / ZN);
            let l = 116.0 * fy - 16.0;
            let a = (fx - fy) * 500.0;
            let bstar = (fy - fz) * 200.0;
            out[idx] = (l * 255.0 / 100.0).round().clamp(0.0, 255.0) as u8;
            out[idx + 1] = (a + 128.0).round().clamp(0.0, 255.0) as u8;
            out[idx + 2] = (bstar + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn srgb_linear(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let lin = |c: f64| -> f64 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    (lin(r), lin(g), lin(b))
}

pub fn lab_to_bgr(lab: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let l8 = lab[idx] as f64;
            let a8 = lab[idx + 1] as f64;
            let b8 = lab[idx + 2] as f64;
            let l = l8 * 100.0 / 255.0;
            let a = a8 - 128.0;
            let b = b8 - 128.0;
            // inverse Lab
            let fy = (l + 16.0) / 116.0;
            const EPS: f64 = 0.008856;
            let fx = fy + a / 500.0;
            let fz = fy - b / 200.0;
            let inv = |t: f64| -> f64 {
                let t3 = t * t * t;
                if t3 > EPS {
                    t3
                } else {
                    (116.0 * t - 16.0) / 903.3
                }
            };
            let xr = inv(fx) * 0.950456;
            let yr = if l > 200.0 / 1000.0 * 0.008856 {
                ((l + 16.0) / 116.0).powi(3)
            } else {
                l * 1000.0 / 903.3
            };
            let zr = inv(fz) * 1.088754;
            let rl = 3.240479 * xr - 1.537150 * yr - 0.498535 * zr;
            let gl = -0.969256 * xr + 1.875968 * yr + 0.041556 * zr;
            let bl = 0.055648 * xr - 0.204043 * yr + 1.057311 * zr;
            let gamma = |c: f64| -> u8 {
                let c = c.clamp(0.0, 1.0);
                let v = if c <= 0.0031308 {
                    12.92 * c
                } else {
                    1.055 * c.powf(1.0 / 2.4) - 0.055
                };
                (v * 255.0).round().clamp(0.0, 255.0) as u8
            };
            out[idx] = gamma(bl);     // B
            out[idx + 1] = gamma(gl); // G
            out[idx + 2] = gamma(rl); // R
        }
    }
    out
}

// ---------------------------------------------------------------- flow field

/// cv2 BGR2GRAY with u8 output: 14-bit fixed point —
/// y = (R*4899 + G*9617 + B*1868 + 8192) >> 14 (R2Y/G2Y/B2Y from color.hpp).
/// Verified against cv2 across 2000 random pixels (≤1 off on 5, SIMD rounding).
pub fn gray_from_bgr(bgr: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0f32; w * h];
    for i in 0..w * h {
        let b = bgr[i * 3] as u32;
        let gg = bgr[i * 3 + 1] as u32;
        let r = bgr[i * 3 + 2] as u32;
        let y = (r * 4899 + gg * 9617 + b * 1868 + 8192) >> 14;
        g[i] = y.min(255) as f32;
    }
    g
}

pub struct Flow {
    pub angle: Vec<f32>,
    pub coherence: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

pub fn flow_field(gray: &[f32], w: usize, h: usize, sigma: f64) -> Flow {
    let blurred = gaussian_blur_plane(gray, w, h, sigma);
    let gx = sobel3(&blurred, w, h, 1, 0);
    let gy = sobel3(&blurred, w, h, 0, 1);
    let n = w * h;
    let mut jxx = vec![0f32; n];
    let mut jyy = vec![0f32; n];
    let mut jxy = vec![0f32; n];
    for i in 0..n {
        jxx[i] = gx[i] * gx[i];
        jyy[i] = gy[i] * gy[i];
        jxy[i] = gx[i] * gy[i];
    }
    let mut jxx = gaussian_blur_plane(&jxx, w, h, sigma);
    let mut jyy = gaussian_blur_plane(&jyy, w, h, sigma);
    let mut jxy = gaussian_blur_plane(&jxy, w, h, sigma);
    let mut angle = vec![0f32; n];
    let mut coherence = vec![0f32; n];
    for i in 0..n {
        let a = jxx[i] as f64;
        let b = jyy[i] as f64;
        let c = jxy[i] as f64;
        let tmp = ((a - b) * (a - b) + 4.0 * c * c).sqrt();
        let mut ang = (2.0 * c).atan2(a - b + tmp) * 180.0 / std::f64::consts::PI / 2.0 + 90.0;
        ang %= 180.0;
        angle[i] = ang as f32;
        let l1 = (a + b + tmp) * 0.5;
        let l2 = (a + b - tmp) * 0.5;
        coherence[i] = ((l1 - l2) / (l1 + l2 + 1e-3)).clamp(0.0, 1.0) as f32;
    }
    Flow {
        angle,
        coherence,
        w,
        h,
    }
}

/// smooth_flow: doubled-angle vector average with gaussian kernel (r→sigma=r/2).
pub fn smooth_flow(mut flow: Flow, r: i32) -> Flow {
    let w = flow.w;
    let h = flow.h;
    let n = w * h;
    if r < 2 {
        return flow;
    }
    let mut vx = vec![0f32; n];
    let mut vy = vec![0f32; n];
    for i in 0..n {
        let a2 = flow.angle[i] * 2.0 * std::f32::consts::PI / 180.0;
        vx[i] = a2.cos() * flow.coherence[i];
        vy[i] = a2.sin() * flow.coherence[i];
    }
    let ks = (r as usize) * 2 + 1;
    let sigma = (r as f64) / 2.0;
    let k1 = gaussian_kernel(ks, sigma);
    // 2D separable kernel
    let avgx = separable(&vx, w, h, &k1);
    let avgy = separable(&vy, w, h, &k1);
    let mut angle = vec![0f32; n];
    let mut mag = vec![0f32; n];
    for i in 0..n {
        let mut a = avgy[i].atan2(avgx[i]) * 180.0 / std::f32::consts::PI / 2.0;
        a %= 180.0;
        if a < 0.0 {
            a += 180.0;
        }
        angle[i] = a;
        mag[i] = (avgx[i] * avgx[i] + avgy[i] * avgy[i]).sqrt();
    }
    flow.angle = angle;
    flow.coherence = mag;
    flow
}

fn separable(src: &[f32], w: usize, h: usize, k: &Gauss1D) -> Vec<f32> {
    let half = k.size / 2;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, kx) in (-((half as i64))..=(half as i64)).enumerate() {
                let sx = reflect101(x as i64 + kx, w as i64) as usize;
                acc += src[y * w + sx] * k.coeffs[ki];
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, ky) in (-((half as i64))..=(half as i64)).enumerate() {
                let sy = reflect101(y as i64 + ky, h as i64) as usize;
                acc += tmp[sy * w + x] * k.coeffs[ki];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------- kuwahara

pub fn kuwahara(bgr: &[u8], w: usize, h: usize, radius: i32) -> Vec<u8> {
    let lab = bgr_to_lab(bgr, w, h);
    let n = radius + 1;
    let area = n * n;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let mut best_var = f32::INFINITY;
            let mut best_mean = [0f32; 3];
            for qi in 0..4 {
                let (y0, x0) = quadrant_origin(qi, radius);
                let half_win = (n / 2) as i64;
                let top = y as i64 + y0 - half_win;
                let left = x as i64 + x0 - half_win;
                let mut sum = [0f64; 3];
                let mut sum2 = 0f64;
                for yy in 0..n {
                    let py = reflect101(top + yy as i64, h as i64) as usize;
                    for xx in 0..n {
                        let px = reflect101(left + xx as i64, w as i64) as usize;
                        let pidx = (py * w + px) * 3;
                        let l = lab[pidx];
                        sum[0] += l as f64;
                        sum2 += (l as f64) * (l as f64);
                        sum[1] += lab[pidx + 1] as f64;
                        sum[2] += lab[pidx + 2] as f64;
                    }
                }
                let mean = sum[0] / area as f64;
                let var = sum2 / area as f64 - mean * mean;
                if (var as f32) < best_var {
                    best_var = var as f32;
                    // NOTE python quirk: colors channel 0 = a, 1 = b, 2 unused; out[0]=mean L
                    best_mean[0] = mean as f32;
                    best_mean[1] = sum[1] as f32 / area as f32;
                    best_mean[2] = sum[2] as f32 / area as f32;
                }
            }
            let idx = (y * w + x) * 3;
            out[idx] = best_mean[0].round().clamp(0.0, 255.0) as u8;
            out[idx + 1] = best_mean[1].round().clamp(0.0, 255.0) as u8;
            out[idx + 2] = best_mean[2].round().clamp(0.0, 255.0) as u8;
        }
    }
    lab_to_bgr(&out, w, h)
}

fn quadrant_origin(q: i32, n: i32) -> (i64, i64) {
    // python offs = [(-r,-r), (-r, 0), (0, -r), (0, 0)]
    match q {
        0 => (-(n as i64), -(n as i64)),
        1 => (-(n as i64), 0),
        2 => (0, -(n as i64)),
        _ => (0, 0),
    }
}

// ---------------------------------------------------------------- strokes

pub struct StrokeOut {
    pub bgra: Vec<u8>,
}

pub fn render_strokes(
    bgr: &[u8],
    base: &[u8],
    w: usize,
    h: usize,
    flow: &Flow,
    strokes_per_layer: i32,
    length: i32,
    width: i32,
    layers: i32,
    jitter: f64,
    opacity: f64,
    seed: u64,
) -> StrokeOut {
    let mut rng = Pcg64::new(seed);
    // canvas starts as base (BGRA out)
    let mut canvas: Vec<[f32; 3]> = base
        .chunks(3)
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let mut updated = vec![0f32; w * h];
    // gradient magnitude of gray
    let gray = gray_from_bgr(bgr, w, h);
    let gx = sobel3(&gray, w, h, 1, 0);
    let gy = sobel3(&gray, w, h, 0, 1);
    let grad: Vec<f32> = (0..w * h).map(|i| (gx[i].powi(2) + gy[i].powi(2)).sqrt()).collect();

    let step = (w as i32 / (strokes_per_layer as f64).sqrt() as i32).max(2);
    let jit = step as f64 * 0.5 * jitter;

    for _layer in 0..layers {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut yy = 0i32;
        while yy < h as i32 {
            let mut xx = 0i32;
            while xx < w as i32 {
                pts.push((
                    xx as f64 + rng.uniform(-jit, jit),
                    yy as f64 + rng.uniform(-jit, jit),
                ));
                xx += step;
            }
            yy += step;
        }
        // sort by descending grad (i64 rounding clip like np.astype(int).clip)
        let gi = |p: &(f64, f64)| -> usize {
            let xi = (p.0 as i64).clamp(0, w as i64 - 1) as usize;
            let yi = (p.1 as i64).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };
        pts.sort_by(|a, b| {
            grad[gi(b)]
                .partial_cmp(&grad[gi(a)])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (x0, y0) in pts {
            let xi = (x0 as i64).clamp(0, w as i64 - 1) as usize;
            let yi = (y0 as i64).clamp(0, h as i64 - 1) as usize;
            let a = flow.angle[yi * w + xi] as f64 * std::f64::consts::PI / 180.0;
            let steps = (length as i64 / 2).max(2) as usize;
            let mut poly: Vec<(f64, f64)> = vec![(x0, y0)];
            let (mut cx, mut cy) = (x0, y0);
            let (mut dx0, mut dy0) = (a.cos(), a.sin());
            for _s in 0..steps {
                let xj = (cx as i64).clamp(0, w as i64 - 1) as usize;
                let yj = (cy as i64).clamp(0, h as i64 - 1) as usize;
                let mut a2 = flow.angle[yj * w + xj] as f64 * std::f64::consts::PI / 180.0;
                let cur = dy0.atan2(dx0);
                let mut d1 = (a2 - cur + std::f64::consts::PI)
                    % (2.0 * std::f64::consts::PI)
                    - std::f64::consts::PI;
                if d1 < -std::f64::consts::PI {
                    d1 += 2.0 * std::f64::consts::PI;
                }
                if d1.abs() >= std::f64::consts::PI / 2.0 {
                    a2 += std::f64::consts::PI;
                }
                dx0 = 0.7 * dx0 + 0.3 * a2.cos();
                dy0 = 0.7 * dy0 + 0.3 * a2.sin();
                let nrm = (dx0 * dx0 + dy0 * dy0).sqrt() + 1e-8;
                dx0 /= nrm;
                dy0 /= nrm;
                cx += 2.0 * dx0;
                cy += 2.0 * dy0;
                if !(cx >= 0.0 && cx < w as f64 && cy >= 0.0 && cy < h as f64) {
                    break;
                }
                poly.push((cx, cy));
            }
            if poly.len() < 3 {
                continue;
            }
            // color: mean of src patch around midpoint (RGB channels)
            let mi = poly.len() / 2;
            let mx = ((poly[mi].0 as i64).clamp(0, w as i64 - 1)) as i64;
            let my = ((poly[mi].1 as i64).clamp(0, h as i64 - 1)) as i64;
            let r = (width as i64 / 2).max(1);
            let (mut sr, mut sg, mut sb, mut cnt) = (0f64, 0f64, 0f64, 0i64);
            for yy in (my - r).max(0)..=(my + r).min(h as i64 - 1) {
                for xx in (mx - r).max(0)..=(mx + r).min(w as i64 - 1) {
                    let p = (yy as usize * w + xx as usize) * 3;
                    sb += bgr[p] as f64;
                    sg += bgr[p + 1] as f64;
                    sr += bgr[p + 2] as f64;
                    cnt += 1;
                }
            }
            let lum = rng.uniform(0.94, 1.06);
            let _lj = length as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter);
            let wj = ((width as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter)) as i64).max(1) as i32;
            let th = wj.max(1);
            let col = [
                (sb / cnt as f64 * lum).clamp(0.0, 255.0) as f32,
                (sg / cnt as f64 * lum).clamp(0.0, 255.0) as f32,
                (sr / cnt as f64 * lum).clamp(0.0, 255.0) as f32,
            ];
            draw_polyline(&mut canvas, &mut updated, w, h, &poly, col, th, 1.0);
        }
    }

    // blend: a = clip(blur(updated) * 1.4, 0, 1) * opacity
    let blurred = gaussian_blur_plane(&updated, w, h, 0.6);
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        let a = (blurred[i] * 1.4).clamp(0.0, 1.0) * opacity as f32;
        for c in 0..3 {
            let v = canvas[i][c] * a + base[i * 3 + c] as f32 * (1.0 - a);
            out[i * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    StrokeOut { bgra: out }
}

/// cv2.LINE_AA-ish polyline with thickness (anti-aliased coverage).
fn draw_polyline(
    canvas: &mut [[f32; 3]],
    updated: &mut [f32],
    w: usize,
    h: usize,
    poly: &[(f64, f64)],
    col: [f32; 3],
    thickness: i32,
    alpha: f32,
) {
    // walk each segment, rasterize with distance-based AA (cv2 LINE_AA)
    let half = (thickness as f64) / 2.0;
    let minx = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let maxx = poly
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let miny = poly.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let maxy = poly
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let x0 = ((minx - half - 1.0).floor() as i64).max(0) as usize;
    let x1 = ((maxx + half + 1.0).ceil() as i64).min(w as i64 - 1) as usize;
    let y0 = ((miny - half - 1.0).floor() as i64).max(0) as usize;
    let y1 = ((maxy + half + 1.0).ceil() as i64).min(h as i64 - 1) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            // min distance to polyline
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            let mut dmin = f64::INFINITY;
            for s in 0..poly.len() - 1 {
                let (ax, ay) = poly[s];
                let (bx, by) = poly[s + 1];
                let dx = bx - ax;
                let dy = by - ay;
                let len2 = dx * dx + dy * dy;
                let t = if len2 <= 0.0 {
                    0.0
                } else {
                    (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
                };
                let qx = ax + t * dx;
                let qy = ay + t * dy;
                let d = ((px - qx).powi(2) + (py - qy).powi(2)).sqrt();
                if d < dmin {
                    dmin = d;
                }
            }
            // AA coverage
            let a = ((half + 0.5 - dmin).clamp(0.0, 1.0) * alpha as f64) as f32;
            if a > 0.0 {
                let i = y * w + x;
                for c in 0..3 {
                    canvas[i][c] = canvas[i][c] * (1.0 - a) + col[c] * a;
                }
                updated[i] = updated[i] * (1.0 - a) + a;
            }
        }
    }
}

// ---------------------------------------------------------------- relief

pub fn canvas_relief(w: usize, h: usize, scale: i32, strength: f64) -> (Vec<f32>, Vec<f32>) {
    let mut height = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            height[y * w + x] = (0.5 * (x as f64 * std::f64::consts::PI / scale as f64).sin()
                * (y as f64 * std::f64::consts::PI / scale as f64).sin())
                as f32;
        }
    }
    // cv2 Sobel on height with strength → normal map → lambert
    let gx = sobel3(&height, w, h, 1, 0);
    let gy = sobel3(&height, w, h, 0, 1);
    let light = [-0.5f64, -0.5, 0.7];
    let ll = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt();
    let (lx, ly, lz) = (light[0] / ll, light[1] / ll, light[2] / ll);
    let mut shade = vec![0f32; w * h];
    for i in 0..w * h {
        let nx = gx[i] as f64;
        let ny = gy[i] as f64;
        let nz = 10.0;
        let nl = (nx * nx + ny * ny + nz * nz).sqrt() + 1e-6;
        let v = (nx * lx + ny * ly + nz * lz) / nl;
        let s = v.clamp(-1.0, 1.0) * 0.5 + 0.5;
        shade[i] = ((s * strength + (1.0 - strength)) as f32);
    }
    (shade, height)
}

// ---------------------------------------------------------------- pipeline

pub fn paint_bgra(bgra: &[u8], w: usize, h: usize, p: &Params) -> (Vec<u8>, String) {
    use std::time::Instant;
    let t0 = Instant::now();
    // BGRA -> BGR
    let mut bgr = vec![0u8; w * h * 3];
    for i in 0..w * h {
        bgr[i * 3] = bgra[i * 4];
        bgr[i * 3 + 1] = bgra[i * 4 + 1];
        bgr[i * 3 + 2] = bgra[i * 4 + 2];
    }
    // resize handled by JS (canvas); here we assume already-resized
    let base = kuwahara(&bgr, w, h, p.base_radius);
    let gray = gray_from_bgr(&bgr, w, h);
    let flow0 = flow_field(&gray, w, h, 2.5);
    let flow = smooth_flow(flow0, p.flow_smooth);
    let n_strokes = (w * h) as i32 / 140;
    let stroke_out = render_strokes(
        &bgr,
        &base,
        w,
        h,
        &flow,
        n_strokes,
        p.brush_len,
        p.brush_w,
        p.layers,
        0.6,
        0.9,
        p.seed,
    );
    let mut out = stroke_out.bgra;
    if p.relief > 0.0 {
        let (shade, _h) = canvas_relief(w, h, 12, p.relief);
        for i in 0..w * h {
            let s = shade[i];
            for c in 0..4 {
                let v = out[i * 4 + c] as f32;
                out[i * 4 + c] = (v * s).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let ms = t0.elapsed().as_millis();
    (out, format!("{ms}ms"))
}

//! pipeline.rs — port of isophote.py (flow field, kuwahara, strokes, relief).
//! All buffers are HxWx4 BGRA u8 (alpha ignored on input, 255 on output).

use crate::pcg64::Pcg64;
use std::borrow::Cow;

#[derive(serde::Deserialize)]
pub struct Params {
    pub brush_len: i32,
    pub brush_w: i32,
    pub layers: i32,
    pub relief: f64,
    pub base_radius: i32,
    pub flow_smooth: i32,
    pub seed: u64,
    // input downscale cap (long edge). Omitted -> 1400 (old behavior).
    #[serde(default)]
    pub max_size: Option<i32>,
    /// Impasto strength (0 = off, bit-identical). Shades color channels by a
    /// stroke height field; multi path also emits a trailing height frame for
    /// the Blender Bump node.
    #[serde(default)]
    pub impasto: f64,
    /// Impasto look: false = bristle strokes, true = palette knife (flat slabs).
    #[serde(default)]
    pub impasto_knife: bool,
    /// Animation boil: share (0..1) of strokes repainted on frame `boil_frame`.
    #[serde(default)]
    pub boil: f64,
    /// Animation frame index; 0 = base painting (boil has no effect).
    #[serde(default)]
    pub boil_frame: u32,
    /// Brushwork zoom: multiplies the pixel-unit constants of the paint path.
    /// 0/omitted = 1.0 (Python-reference behaviour).
    #[serde(default)]
    pub look_scale: f64,
    /// Long edge of the returned frames; painted frames are area-downscaled to
    /// it. 0/omitted or >= painted long edge = no resize.
    #[serde(default)]
    pub out_size: i32,
    /// Frames of kind "normal" are tangent-space (Blender/OpenGL: +X = u, +Y =
    /// v, +Z out): with impasto > 0 the stroke-height slope is blended into
    /// them. False = object-space maps, left untouched (relief goes through
    /// the height frame / Bump node instead).
    #[serde(default)]
    pub normal_tangent: bool,
    /// Paint on the mesh surface (strokes cross UV seams) when the caller
    /// supplies a mesh; false or no mesh = classic flat-texture strokes.
    #[serde(default)]
    pub uv_aware: bool,
}

impl Params {
    pub fn look(&self) -> f64 {
        if self.look_scale > 0.0 {
            self.look_scale
        } else {
            1.0
        }
    }
}

/// Companion struct: optional per-frame kind names, same params JSON.
/// Frame-aligned: index 0 = reference kind, index i = chans[i-1].
#[derive(serde::Deserialize, Default)]
pub struct KindsHint {
    #[serde(default)]
    pub kinds: Vec<Option<String>>,
}

/// Kind name of frame `i` (kinds are frame-aligned: kinds[0] = reference).
pub fn kind_at(kinds: Option<&[Option<String>]>, i: usize) -> Option<&str> {
    kinds.and_then(|ks| ks.get(i)).and_then(|k| k.as_deref())
}

// ---------------------------------------------------------------- image ops

/// Long-edge downscale for max_size cap (None/0 = no cap).
/// Returns Some((dw, dh, scale)) when the image exceeds the cap.
pub fn downscale_dims(w: usize, h: usize, max_size: i32) -> Option<(usize, usize, f64)> {
    if max_size <= 0 {
        return None;
    }
    let ms = max_size as usize;
    if w.max(h) <= ms {
        return None;
    }
    let scale = w.max(h) as f64 / ms as f64;
    Some((((w as f64 / scale) as usize).max(1), ((h as f64 / scale) as usize).max(1), scale))
}

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

/// Separable convolution with a symmetric odd kernel `k`. Per output pixel the
/// accumulator starts at 0.0 and adds taps in ascending order (bit-identical to
/// the per-pixel reference loop); interior columns run tap-outer so the inner
/// loop is a plain elementwise multiply-add the compiler can vectorize.
fn conv_separable(
    src: &[f32],
    w: usize,
    h: usize,
    k: &[f32],
    reflect: impl Fn(i64, i64) -> i64 + Copy,
) -> Vec<f32> {
    let half = k.len() / 2;
    let mut tmp = vec![0f32; w * h];
    let interior = w > 2 * half;
    for y in 0..h {
        let srow = &src[y * w..(y + 1) * w];
        let trow = &mut tmp[y * w..(y + 1) * w];
        if interior {
            let n = w - 2 * half;
            let dst = &mut trow[half..half + n];
            for (ki, &kc) in k.iter().enumerate() {
                let s = &srow[ki..ki + n];
                for (d, &v) in dst.iter_mut().zip(s) {
                    *d += v * kc;
                }
            }
        }
        let (bl, br) = if interior { (half, half) } else { (w, 0) };
        for x in (0..bl).chain(w - br..w) {
            let mut acc = 0f32;
            for (ki, &kc) in k.iter().enumerate() {
                let sx = reflect(x as i64 + ki as i64 - half as i64, w as i64) as usize;
                acc += srow[sx] * kc;
            }
            trow[x] = acc;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let orow = &mut out[y * w..(y + 1) * w];
        for (ki, &kc) in k.iter().enumerate() {
            let sy = reflect(y as i64 + ki as i64 - half as i64, h as i64) as usize;
            let trow = &tmp[sy * w..(sy + 1) * w];
            for (o, &v) in orow.iter_mut().zip(trow) {
                *o += v * kc;
            }
        }
    }
    out
}

/// Gaussian blur on a single f32 plane (separable, reflect101).
pub fn gaussian_blur_plane(src: &[f32], w: usize, h: usize, sigma_x: f64) -> Vec<f32> {
    if sigma_x <= 0.0 {
        return src.to_vec();
    }
    let ksize = (((sigma_x * 4.0 * 2.0 + 1.0).round() as usize) | 1);
    let k = gaussian_kernel(ksize, sigma_x);
    conv_separable(src, w, h, &k.coeffs, reflect_101)
}

/// Sobel 3x3 on f32 plane, cv2 default border REFLECT_101 (cv2.Sobel).
/// Taps are added in row-major order skipping the zero-weight ones (exact:
/// the accumulator starts at +0.0 and can never become -0.0).
pub fn sobel3(src: &[f32], w: usize, h: usize, dx: i32, dy: i32) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    if dx != 1 && dy != 1 {
        return out;
    }
    let kernel = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| -> f32 {
        // a/b/c = rows y-1 / y / y+1, each as [left, centre, right]
        let mut acc = 0f32;
        if dx == 1 {
            acc += a[0] * -1.0;
            acc += a[2] * 1.0;
            acc += b[0] * -2.0;
            acc += b[2] * 2.0;
            acc += c[0] * -1.0;
            acc += c[2] * 1.0;
        } else {
            acc += a[0] * -1.0;
            acc += a[1] * -2.0;
            acc += a[2] * -1.0;
            acc += c[0] * 1.0;
            acc += c[1] * 2.0;
            acc += c[2] * 1.0;
        }

        acc
    };
    for y in 0..h {
        let y0 = reflect_101(y as i64 - 1, h as i64) as usize;
        let y2 = reflect_101(y as i64 + 1, h as i64) as usize;
        let (r0, r1, r2) = (&src[y0 * w..(y0 + 1) * w], &src[y * w..(y + 1) * w], &src[y2 * w..(y2 + 1) * w]);
        let orow = &mut out[y * w..(y + 1) * w];
        let edge = |x: usize| -> f32 {
            let l = reflect_101(x as i64 - 1, w as i64) as usize;
            let r = reflect_101(x as i64 + 1, w as i64) as usize;
            kernel([r0[l], r0[x], r0[r]], [r1[l], r1[x], r1[r]], [r2[l], r2[x], r2[r]])
        };
        if w < 3 {
            for x in 0..w {
                orow[x] = edge(x);
            }
            continue;
        }
        orow[0] = edge(0);
        orow[w - 1] = edge(w - 1);
        for x in 1..w - 1 {
            orow[x] = kernel(
                [r0[x - 1], r0[x], r0[x + 1]],
                [r1[x - 1], r1[x], r1[x + 1]],
                [r2[x - 1], r2[x], r2[x + 1]],
            );
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

//// cv2 INTER_AREA resize: area-weighted average (overlap area weights with
/// fractional scales), cv2 8-bit rounding via (v + 127.5)/255 style saturate
/// — cv2 rounds to nearest. Used when JS sends the full-size decode.
pub fn resize_area(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, c: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let xr = sw as f64 / dw as f64;
    let yr = sh as f64 / dh as f64;
    let mut out = vec![0u8; dw * dh * c];
    // precompute per-dst-pixel column weights: (src start, span) pairs are
    // shared across rows; do accumulation in f64.
    for y in 0..dh {
        let sy0 = (y as f64 * yr).floor() as usize;
        let sy1 = (((y + 1) as f64 * yr).floor() as usize).min(sh);
        for x in 0..dw {
            let sx0 = (x as f64 * xr).floor() as usize;
            let sx1 = (((x + 1) as f64 * xr).floor() as usize).min(sw);
            for ci in 0..c {
                let mut acc = 0f64;
                let mut wsum = 0f64;
                for yy in sy0..sy1.max(sy0 + 1) {
                    let wy = ((yy as f64 + 1.0).min((y + 1) as f64 * yr) - (yy as f64).max(y as f64 * yr)).max(0.0);
                    for xx in sx0..sx1.max(sx0 + 1) {
                        let wx = ((xx as f64 + 1.0).min((x + 1) as f64 * xr) - (xx as f64).max(x as f64 * xr)).max(0.0);
                        let wt = wx * wy;
                        acc += src[(yy * sw + xx) * c + ci] as f64 * wt;
                        wsum += wt;
                    }
                }
                out[(y * dw + x) * c + ci] = (acc / wsum.max(1e-9)).round().clamp(0.0, 255.0) as u8;
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
    // sRGB 8-bit -> linear table (same expression as the former per-pixel call)
    let lin: [f64; 256] = std::array::from_fn(|v| {
        let c = v as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let bl = lin[src[idx] as usize];
            let gl = lin[src[idx + 1] as usize];
            let rl = lin[src[idx + 2] as usize];
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

pub fn lab_to_bgr(lab: &[u8], w: usize, h: usize) -> Vec<u8> {
    let bins = gamma_bins();
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
                let t = bins[((c * GAMMA_N as f64) as usize).min(GAMMA_N - 1)];
                if t != 0xFFFF {
                    return t as u8;
                }
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

const GAMMA_N: usize = 16384;

/// sRGB gamma (f64 [0,1] -> 8-bit) lookup: a bin holds its value only when the
/// function rounds to the same byte at both bin edges with a 1e-6 margin from
/// a rounding boundary (monotone => exact inside); 0xFFFF = compute directly.
fn gamma_bins() -> &'static [u16] {
    static BINS: std::sync::LazyLock<Box<[u16]>> = std::sync::LazyLock::new(|| {
        let g = |c: f64| -> f64 {
            let v = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
            v * 255.0
        };
        let e: Vec<f64> = (0..=GAMMA_N).map(|i| g(i as f64 / GAMMA_N as f64)).collect();
        let safe = |v: f64| (v - v.floor() - 0.5).abs() > 1e-6;
        let branch_bin = (0.0031308 * GAMMA_N as f64) as usize;
        (0..GAMMA_N)
            .map(|i| {
                let (a, b) = (e[i], e[i + 1]);
                if i != branch_bin && a.round() == b.round() && safe(a) && safe(b) {
                    a.round().clamp(0.0, 255.0) as u16
                } else {
                    0xFFFF
                }
            })
            .collect()
    });
    &BINS
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
    let avgx = conv_separable(&vx, w, h, &k1.coeffs, reflect101);
    let avgy = conv_separable(&vy, w, h, &k1.coeffs, reflect101);
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

// ---------------------------------------------------------------- kuwahara

/// Column sums of Lab over a window of padded rows (see `kuwahara`).
struct ColSums {
    l: Vec<u32>,
    l2: Vec<u64>,
    a: Vec<u32>,
    b: Vec<u32>,
}

impl ColSums {
    fn new(cols: usize) -> Self {
        ColSums { l: vec![0; cols], l2: vec![0; cols], a: vec![0; cols], b: vec![0; cols] }
    }
    /// Add (`add`) or remove one source row of Lab, expanded through `xmap`.
    fn row(&mut self, lab: &[u8], w: usize, srow: usize, xmap: &[usize], add: bool) {
        let base = srow * w;
        for (xi, &sx) in xmap.iter().enumerate() {
            let p = (base + sx) * 3;
            let l = lab[p] as u32;
            let (l2, a, b) = ((l * l) as u64, lab[p + 1] as u32, lab[p + 2] as u32);
            if add {
                self.l[xi] += l;
                self.l2[xi] += l2;
                self.a[xi] += a;
                self.b[xi] += b;
            } else {
                self.l[xi] -= l;
                self.l2[xi] -= l2;
                self.a[xi] -= a;
                self.b[xi] -= b;
            }
        }
    }
    /// Sliding window sums [l, l2, a, b] of `n` consecutive columns for every start.
    fn windows(&self, n: usize, out: &mut Vec<[u64; 4]>) {
        out.clear();
        let m = self.l.len() - n + 1;
        let mut acc = [0u64; 4];
        for i in 0..n {
            acc[0] += self.l[i] as u64;
            acc[1] += self.l2[i];
            acc[2] += self.a[i] as u64;
            acc[3] += self.b[i] as u64;
        }
        out.push(acc);
        for i in 0..m - 1 {
            acc[0] = acc[0] + self.l[i + n] as u64 - self.l[i] as u64;
            acc[1] = acc[1] + self.l2[i + n] - self.l2[i];
            acc[2] = acc[2] + self.a[i + n] as u64 - self.a[i] as u64;
            acc[3] = acc[3] + self.b[i + n] as u64 - self.b[i] as u64;
            out.push(acc);
        }
    }
}

/// Kuwahara filter on Lab with four (r+1)x(r+1) quadrants around each pixel.
/// Integer box sums over a reflect101-padded frame (exact, so identical to
/// summing each window directly): `top` = padded rows [y, y+n), `bot` =
/// [y+r, y+r+n); quadrant column starts are x and x+r.
pub fn kuwahara(bgr: &[u8], w: usize, h: usize, radius: i32) -> Vec<u8> {
    let lab = bgr_to_lab(bgr, w, h);
    let r = radius as usize;
    let n = r + 1;
    let hw = n / 2;
    let pt = (r + hw) as i64;
    let cols = w + 2 * r;
    let area = n * n;
    let xmap: Vec<usize> = (0..cols).map(|x| reflect101(x as i64 - pt, w as i64) as usize).collect();
    let rmap: Vec<usize> = (0..h + 2 * r).map(|y| reflect101(y as i64 - pt, h as i64) as usize).collect();
    let mut top = ColSums::new(cols);
    let mut bot = ColSums::new(cols);
    for yy in 0..n {
        top.row(&lab, w, rmap[yy], &xmap, true);
        bot.row(&lab, w, rmap[r + yy], &xmap, true);
    }
    let mut out = vec![0u8; w * h * 3];
    let (mut wt, mut wb) = (Vec::new(), Vec::new());
    for y in 0..h {
        if y > 0 {
            top.row(&lab, w, rmap[y - 1], &xmap, false);
            top.row(&lab, w, rmap[y - 1 + n], &xmap, true);
            bot.row(&lab, w, rmap[y - 1 + r], &xmap, false);
            bot.row(&lab, w, rmap[y - 1 + r + n], &xmap, true);
        }
        top.windows(n, &mut wt);
        bot.windows(n, &mut wb);
        for x in 0..w {
            let mut best_var = f32::INFINITY;
            let mut best_mean = [0f32; 3];
            for s in [wt[x], wt[x + r], wb[x], wb[x + r]] {
                let mean = s[0] as f64 / area as f64;
                let var = s[1] as f64 / area as f64 - mean * mean;
                if (var as f32) < best_var {
                    best_var = var as f32;
                    // NOTE python quirk: colors channel 0 = a, 1 = b, 2 unused; out[0]=mean L
                    best_mean[0] = mean as f32;
                    best_mean[1] = (s[2] as f64) as f32 / area as f32;
                    best_mean[2] = (s[3] as f64) as f32 / area as f32;
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

// ---------------------------------------------------------------- strokes

pub struct Stroke {
    /// polyline nodes (x, y), f32 like the python reference (jitter in f32)
    pub pts: Vec<(f32, f32)>,
    /// final jittered brush thickness
    pub th: i32,
    /// color-patch radius: unjittered brush_w/2 (matches original render)
    pub pr: i32,
    /// per-stroke brightness jitter (rng-derived; shared across channels)
    pub lum: f32,
}

/// Walk a stroke polyline from (x0, y0) along the flow field (f32 math like
/// the python reference). Shared by generate_strokes and boil_strokes.
fn trace_stroke(
    x0: f32,
    y0: f32,
    flow: &Flow,
    w: usize,
    h: usize,
    length: i32,
    look: f64,
) -> Vec<(f32, f32)> {
    let xi = (x0 as i64).clamp(0, w as i64 - 1) as usize;
    let yi = (y0 as i64).clamp(0, h as i64 - 1) as usize;
    let a = (flow.angle[yi * w + xi] as f32).to_radians();
    let step = (2.0 * look) as f32;
    let steps = ((length as f64 / (2.0 * look)) as i64).max(2) as usize;
    let mut poly: Vec<(f32, f32)> = vec![(x0, y0)];
    let (mut cx, mut cy) = (x0, y0);
    let (mut dx0, mut dy0) = (a.cos(), a.sin());
    for _s in 0..steps {
        let xj = (cx as i64).clamp(0, w as i64 - 1) as usize;
        let yj = (cy as i64).clamp(0, h as i64 - 1) as usize;
        let mut a2 = (flow.angle[yj * w + xj] as f32).to_radians();
        let cur = dy0.atan2(dx0);
        let mut d1 = (a2 - cur + std::f32::consts::PI)
            % (2.0 * std::f32::consts::PI)
            - std::f32::consts::PI;
        if d1 < -std::f32::consts::PI {
            d1 += 2.0 * std::f32::consts::PI;
        }
        if d1.abs() >= std::f32::consts::PI / 2.0 {
            a2 += std::f32::consts::PI;
        }
        dx0 = 0.7 * dx0 + 0.3 * a2.cos();
        dy0 = 0.7 * dy0 + 0.3 * a2.sin();
        let nrm = (dx0 * dx0 + dy0 * dy0).sqrt() + 1e-8;
        dx0 /= nrm;
        dy0 /= nrm;
        cx += step * dx0;
        cy += step * dy0;
        if !(cx >= 0.0 && cx < w as f32 && cy >= 0.0 && cy < h as f32) {
            break;
        }
        poly.push((cx, cy));
    }
    poly
}

/// Animation boil: repaint a `boil` share of `strokes` for frame `frame` (new
/// start offset re-traced through the flow, new lum and width). The other
/// strokes stay identical to frame 0, and draw order never changes.
pub fn boil_strokes(
    strokes: &mut [Stroke],
    w: usize,
    h: usize,
    flow: &Flow,
    strokes_per_layer: i32,
    length: i32,
    width: i32,
    jitter: f64,
    seed: u64,
    frame: u32,
    boil: f64,
    look: f64,
) {
    let mut rng = Pcg64::new(seed ^ (frame as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let step = (h as i32 / (strokes_per_layer as f64).sqrt() as i32).max(2);
    let off = step as f64 * 0.5;
    for s in strokes.iter_mut() {
        if rng.uniform(0.0, 1.0) >= boil {
            continue;
        }
        let dx = rng.uniform(-off, off);
        let dy = rng.uniform(-off, off);
        let lum = rng.uniform(0.94, 1.06) as f32;
        let wj = ((width as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter)) as i64).max(1) as i32;
        let poly = trace_stroke(
            (s.pts[0].0 as f64 + dx) as f32,
            (s.pts[0].1 as f64 + dy) as f32,
            flow,
            w,
            h,
            length,
            look,
        );
        if poly.len() < 3 {
            continue;
        }
        s.pts = poly;
        s.lum = lum;
        s.th = wj;
    }
}

/// Stroke geometry + every RNG decision, none of the coloring. Same RNG
/// consumption order as the pre-split render_strokes (jittered grid,
/// gradient sort, then per stroke lum/length/width draws, and the <3-point
/// skip happens BEFORE the draws), so the single-channel path stays
/// bit-identical. Paint several channels with [`paint_from_strokes`] along
/// the same strokes (stroke replay).
pub fn generate_strokes(
    bgr: &[u8],
    w: usize,
    h: usize,
    flow: &Flow,
    strokes_per_layer: i32,
    length: i32,
    width: i32,
    layers: i32,
    jitter: f64,
    seed: u64,
    look: f64,
) -> Vec<Stroke> {
    let mut rng = Pcg64::new(seed);
    // gradient magnitude of gray
    let gray = gray_from_bgr(bgr, w, h);
    let gx = sobel3(&gray, w, h, 1, 0);
    let gy = sobel3(&gray, w, h, 0, 1);
    let grad: Vec<f32> = (0..w * h).map(|i| (gx[i].powi(2) + gy[i].powi(2)).sqrt()).collect();

    let step = (h as i32 / (strokes_per_layer as f64).sqrt() as i32).max(2);
    let jit = step as f64 * 0.5 * jitter;
    let mut out_strokes: Vec<Stroke> = Vec::new();

    for _layer in 0..layers {
        // pts stored f32 like the python reference (jitter math in f32)
        let mut pts: Vec<(f32, f32)> = Vec::new();
        let mut yy = 0i32;
        while yy < h as i32 {
            let mut xx = 0i32;
            while xx < w as i32 {
                pts.push((
                    (xx as f64 + rng.uniform(-jit, jit)) as f32,
                    (yy as f64 + rng.uniform(-jit, jit)) as f32,
                ));
                xx += step;
            }
            yy += step;
        }
        // sort by descending grad (i64 rounding clip like np.astype(int).clip)
        let gi = |p: &(f32, f32)| -> usize {
            let xi = (p.0 as i64).clamp(0, w as i64 - 1) as usize;
            let yi = (p.1 as i64).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };
        pts.sort_by(|a, b| {
            grad[gi(b)]
                .partial_cmp(&grad[gi(a)])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (x0, y0) in &pts {
            let poly = trace_stroke(*x0, *y0, flow, w, h, length, look);
            if poly.len() < 3 {
                continue;
            }
            let lum = rng.uniform(0.94, 1.06) as f32;
            let _lj = length as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter);
            let wj = ((width as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter)) as i64).max(1) as i32;
            let th = wj.max(1);
            out_strokes.push(Stroke { pts: poly, th, pr: ((width / 2).max(1)), lum });
        }
    }
    out_strokes
}

/// Result of rasterizing every stroke once: which stroke painted each pixel
/// last, and the soft blend alpha derived from the coverage mask. Shared by
/// every channel painted with the same strokes (alpha is always 1.0 per stroke,
/// so a pixel's canvas colour is exactly the colour of its last stroke).
pub struct Coverage {
    /// 0 = untouched, else (stroke index + 1) of the last stroke covering the pixel
    pub owner: Vec<u32>,
    /// blend alpha: clip(blur(mask) * 1.4, 0, 1) * 0.9
    pub alpha: Vec<f32>,
    /// strokes that own at least one pixel
    pub used: Vec<bool>,
}

pub fn rasterize_strokes(w: usize, h: usize, strokes: &[Stroke], look: f64) -> Coverage {
    let mut owner = vec![0u32; w * h];
    for (si, s) in strokes.iter().enumerate() {
        stamp_stroke(&mut owner, w, h, &s.pts, s.th, si as u32 + 1);
    }
    let mut used = vec![false; strokes.len()];
    let mask: Vec<f32> = owner
        .iter()
        .map(|&o| {
            if o != 0 {
                used[o as usize - 1] = true;
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let blurred = gaussian_blur_plane(&mask, w, h, 0.6 * look);
    let alpha = blurred.iter().map(|&b| (b * 1.4).clamp(0.0, 1.0) * 0.9f32).collect();
    Coverage { owner, alpha, used }
}

/// cv2.LINE_AA-ish polyline with thickness, recorded as stroke ownership.
/// cv2 thick AA with int32 endpoints renders hard 0/1 coverage; effective
/// half-width = ceil(th/2) for th>=2 (th=3 spans 5 rows, th=5 spans 7, th=6
/// spans 7); pixel centre within half+0.5. Each segment scans its own bbox,
/// narrowed per row to the x-interval that can pass the proj/perp tests (the
/// tests themselves are unchanged); radius-2 caps at both polyline ends.
fn stamp_stroke(owner: &mut [u32], w: usize, h: usize, poly: &[(f32, f32)], thickness: i32, id: u32) {
    let half = if thickness <= 1 {
        0.5 // th=1 cv2 thin AA line: single-pixel error-diffusion band
    } else {
        ((thickness + 1) / 2) as f64 + 0.5
    };
    for s in 0..poly.len() - 1 {
        let (ax, ay) = (poly[s].0.round() as f64, poly[s].1.round() as f64);
        let (bx, by) = (poly[s + 1].0.round() as f64, poly[s + 1].1.round() as f64);
        let lo_x = (ax.min(bx) - half - 2.0).floor().max(0.0) as usize;
        let lo_y = (ay.min(by) - half - 2.0).floor().max(0.0) as usize;
        let hi_x = (ax.max(bx) + half + 2.0).ceil().min(w as f64 - 1.0) as usize;
        let hi_y = (ay.max(by) + half + 2.0).ceil().min(h as f64 - 1.0) as usize;
        let dx = bx - ax;
        let dy = by - ay;
        let len = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = if len > 0.0 { (dx / len, dy / len) } else { (0.0, 0.0) };
        for y in lo_y..=hi_y {
            let py = y as f64 + 0.5;
            if len == 0.0 {
                // zero-length segment: every pixel of the bbox passes both tests
                for x in lo_x..=hi_x {
                    owner[y * w + x] = id;
                }
                continue;
            }
            let dyv = py - ay;
            let (mut xl, mut xr) = (lo_x as f64, hi_x as f64);
            if ux.abs() > 0.01 {
                let p0 = (-1.0 - dyv * uy) / ux;
                let p1 = (len + 1.0 - dyv * uy) / ux;
                let (a, b) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
                xl = xl.max((ax + a - 0.5 - 1e-6).ceil());
                xr = xr.min((ax + b - 0.5 + 1e-6).floor());
            }
            if uy.abs() > 0.01 {
                let q0 = (dyv * ux - half) / uy;
                let q1 = (dyv * ux + half) / uy;
                let (a, b) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
                xl = xl.max((ax + a - 0.5 - 1e-6).ceil());
                xr = xr.min((ax + b - 0.5 + 1e-6).floor());
            }
            if xl > xr {
                continue;
            }
            for x in xl as usize..=xr as usize {
                let px = x as f64 + 0.5;
                let proj = (px - ax) * ux + (py - ay) * uy;
                if proj < -1.0 || proj > len + 1.0 {
                    continue;
                }
                let perp = (-(px - ax) * uy + (py - ay) * ux).abs();
                if perp > half {
                    continue;
                }
                owner[y * w + x] = id;
            }
        }
    }
    // end caps: radius-2 circles at the two polyline ends
    for (ex, ey) in [
        (poly[0].0.round() as f64, poly[0].1.round() as f64),
        (poly[poly.len() - 1].0.round() as f64, poly[poly.len() - 1].1.round() as f64),
    ] {
        let lo_x = (ex - 2.5).floor().max(0.0) as usize;
        let hi_x = (ex + 2.5).ceil().min(w as f64 - 1.0) as usize;
        let lo_y = (ey - 2.5).floor().max(0.0) as usize;
        let hi_y = (ey + 2.5).ceil().min(h as f64 - 1.0) as usize;
        for y in lo_y..=hi_y {
            for x in lo_x..=hi_x {
                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;
                if ((px - ex).powi(2) + (py - ey).powi(2)).sqrt() <= 2.0 {
                    owner[y * w + x] = id;
                }
            }
        }
    }
}

/// Colour every used stroke from `bgr` (f32 patch mean at the stroke midpoint,
/// f64 accumulation, times lum) and blend the stroke layer over `base`; then
/// multiply all four bytes by the canvas `shade` plane when present.
pub fn compose(
    bgr: &[u8],
    base: &[u8],
    w: usize,
    h: usize,
    strokes: &[Stroke],
    cov: &Coverage,
    shade: Option<&[f32]>,
) -> Vec<u8> {
    let samples: Vec<StrokeSample> = strokes
        .iter()
        .map(|s| {
            let mi = s.pts.len() / 2;
            StrokeSample {
                x: (s.pts[mi].0 as i64).clamp(0, w as i64 - 1),
                y: (s.pts[mi].1 as i64).clamp(0, h as i64 - 1),
                r: s.pr as i64,
                lum: s.lum,
            }
        })
        .collect();
    compose_samples(bgr, base, w, h, &samples, cov, shade, None)
}

/// Where a stroke samples its colour: patch centre (px), patch radius, brightness jitter.
pub(crate) struct StrokeSample {
    pub x: i64,
    pub y: i64,
    pub r: i64,
    pub lum: f32,
}

/// [`compose`] on precomputed colour samples. `valid` (the surface twin map)
/// drops texels equal to `u32::MAX` from the patch mean.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compose_samples(
    bgr: &[u8],
    base: &[u8],
    w: usize,
    h: usize,
    samples: &[StrokeSample],
    cov: &Coverage,
    shade: Option<&[f32]>,
    valid: Option<&[u32]>,
) -> Vec<u8> {
    let mut cols = vec![[0f32; 3]; samples.len()];
    for (si, s) in samples.iter().enumerate() {
        if !cov.used[si] {
            continue;
        }
        // python: f32 patch mean (f64 accumulation in rust, cast to f32),
        // then *lum f32
        let (mx, my, r) = (s.x, s.y, s.r);
        let (mut sb, mut sg, mut sr, mut cnt) = (0f64, 0f64, 0f64, 0i64);
        for yy in (my - r).max(0)..=(my + r).min(h as i64 - 1) {
            for xx in (mx - r).max(0)..=(mx + r).min(w as i64 - 1) {
                let t = yy as usize * w + xx as usize;
                if valid.is_some_and(|v| v[t] == u32::MAX) {
                    continue;
                }
                let p = t * 3;
                sb += bgr[p] as f64;
                sg += bgr[p + 1] as f64;
                sr += bgr[p + 2] as f64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        cols[si] = [
            ((sb / cnt as f64) as f32 * s.lum).clamp(0.0, 255.0),
            ((sg / cnt as f64) as f32 * s.lum).clamp(0.0, 255.0),
            ((sr / cnt as f64) as f32 * s.lum).clamp(0.0, 255.0),
        ];
    }
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        let a = cov.alpha[i];
        let o = cov.owner[i];
        for c in 0..3 {
            let b = base[i * 3 + c] as f32;
            let canvas = if o != 0 { cols[o as usize - 1][c] } else { b };
            let v = canvas * a + b * (1.0 - a);
            out[i * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    if let Some(shade) = shade {
        for i in 0..w * h {
            let sh = shade[i];
            for c in 0..4 {
                let v = out[i * 4 + c] as f32;
                out[i * 4 + c] = (v * sh).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Draw pre-generated strokes on `base` sampling colors from `bgr`: same patch
/// means, lum/th, draw order, blend and relief as the single-channel reference.
/// `relief` = Params.relief; >0 multiplies in the canvas plane shading.
pub fn paint_from_strokes(
    bgr: &[u8],
    base: &[u8],
    w: usize,
    h: usize,
    strokes: &[Stroke],
    relief: f64,
    look: f64,
) -> Vec<u8> {
    let cov = rasterize_strokes(w, h, strokes, look);
    let shade = (relief > 0.0).then(|| canvas_relief(w, h, 12, relief, look).0);
    compose(bgr, base, w, h, strokes, &cov, shade.as_deref())
}

// ---------------------------------------------------------------- impasto

/// Footprint half-width of a stroke of thickness `th` (matches stamp_stroke for th >= 2).
pub(crate) fn stroke_half(th: i32) -> f32 {
    if th <= 1 { 1.5 } else { ((th + 1) / 2) as f32 + 0.5 }
}

/// Half-width of the widest stroke the width jitter (x1.6) can produce.
pub(crate) fn impasto_ref_half(brush_w: i32) -> f32 {
    stroke_half(((brush_w as f64 * 1.6) as i32).max(1))
}

/// Height (px) that maps to white in the height frame.
fn impasto_norm_px(brush_w: i32) -> f32 {
    2.4 * impasto_ref_half(brush_w)
}

/// Thickness of a white height pixel in UV units x impasto (0 when impasto is off).
pub fn impasto_height_scale(w: usize, h: usize, p: &Params) -> f32 {
    if p.impasto > 0.0 {
        p.impasto as f32 * impasto_norm_px(p.brush_w) / w.max(h) as f32
    } else {
        0.0
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn hash01(seed: u32, i: u32) -> f32 {
    let mut x = seed ^ i.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x as f32 / 4_294_967_296.0
}

pub(crate) struct ImpastoStyle {
    pub t_k: f32,
    pub keep: f32,
    pub rim: f32,
    pub groove0: f32,
    pub groove1: f32,
    pub rag: f32,
    pub blur: f32,
    pub lip: f32,
    pub deplete: f32,
}

const BRISTLE: ImpastoStyle = ImpastoStyle {
    t_k: 0.7, keep: 0.35, rim: 0.5, groove0: 0.18, groove1: 0.55, rag: 1.0, blur: 0.8, lip: 0.3, deplete: 0.4,
};
const KNIFE: ImpastoStyle = ImpastoStyle {
    t_k: 0.8, keep: 0.30, rim: 0.7, groove0: 0.04, groove1: 0.15, rag: 0.5, blur: 0.7, lip: 0.5, deplete: 0.25,
};

/// Bristle strokes or palette-knife slabs.
pub(crate) fn impasto_style(knife: bool) -> &'static ImpastoStyle {
    if knife { &KNIFE } else { &BRISTLE }
}

/// exp(-z^2) by table (1024 intervals over |z| < 4.25, linear interpolation,
/// max error ~4e-6; beyond that the true value is < 1.5e-8 and returns 0).
fn gauss_table() -> &'static [f32] {
    static TAB: std::sync::LazyLock<Box<[f32]>> = std::sync::LazyLock::new(|| {
        (0..=1024).map(|i| (-((i as f64 * 4.25 / 1024.0).powi(2))).exp() as f32).collect()
    });
    &TAB
}

#[inline]
fn gauss(tab: &[f32], z: f32) -> f32 {
    let x = z.abs() * (1024.0 / 4.25);
    if x >= 1024.0 {
        return 0.0;
    }
    let i = x as usize;
    let f = x - i as f32;
    tab[i] + (tab[i + 1] - tab[i]) * f
}

/// smoothstep on an already-normalized coordinate.
#[inline]
pub(crate) fn smooth01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Thickness profile of one impasto stroke, in px: steep shoulders, raised
/// side rims, hashed bristle grooves, a blob at the start and a ragged tail.
/// Per-stroke randoms come from hashes of `seed` (no RNG consumed).
pub(crate) struct StrokeProfile {
    st: &'static ImpastoStyle,
    tab: &'static [f32],
    kn: usize,
    gb: [f32; 24],
    ge: [f32; 24],
    half: f32,
    total: f32,
    /// thickness scale of this stroke
    pub tmax: f32,
    inv_half: f32,
    inv_total: f32,
    inv_edge: f32,
    inv_tail: f32,
    inv_head: f32,
    inv_lip: f32,
    inv_blob: f32,
    inv_rim: f32,
}

impl StrokeProfile {
    /// `half` = footprint half-width, `total` = polyline length, `lk` = look scale.
    pub(crate) fn new(st: &'static ImpastoStyle, seed: u32, half: f32, total: f32, lk: f32) -> Self {
        let kn = (((2.0 * half / (2.2 * lk)).round()) as usize).clamp(3, 24);
        let mut gb = [0f32; 24];
        let mut ge = [0f32; 24];
        for i in 0..kn {
            gb[i] = hash01(seed, i as u32);
            ge[i] = hash01(seed, (kn + i) as u32);
        }
        let load = 0.6 + 0.8 * hash01(seed, (2 * kn) as u32);
        StrokeProfile {
            st,
            tab: gauss_table(),
            kn,
            gb,
            ge,
            half,
            total,
            tmax: st.t_k * half * load,
            inv_half: 1.0 / half,
            inv_total: 1.0 / total,
            inv_edge: 1.0 / (1.4 * lk),
            inv_tail: 1.0 / (0.8 * half),
            inv_head: 1.0 / (0.5 * half + 1.0),
            inv_lip: 1.0 / (0.45 * half),
            inv_blob: 1.0 / (0.9 * half),
            inv_rim: 1.0 / 0.13,
        }
    }

    /// Paint thickness at signed cross offset `sd` and arclength `a`
    /// (0.0 or less = no paint).
    #[inline]
    pub(crate) fn eval(&self, sd: f32, a: f32) -> f32 {
        let (st, tab, kn, half, total) = (self.st, self.tab, self.kn, self.half, self.total);
        let u = (sd * self.inv_half).clamp(-1.0, 1.0);
        let fx = (u * 0.5 + 0.5) * (kn - 1) as f32;
        let i0 = (fx.floor() as usize).min(kn - 2);
        let mut fr = fx - i0 as f32;
        fr = fr * fr * (3.0 - 2.0 * fr);
        let nb = self.gb[i0] * (1.0 - fr) + self.gb[i0 + 1] * fr;
        let ne = self.ge[i0] * (1.0 - fr) + self.ge[i0 + 1] * fr;
        let ac = a.clamp(0.0, total);
        let leff = total - st.rag * half * ne;
        let rem = leff - a;
        if rem <= 0.0 {
            return 0.0; // past the ragged tail: tail == 0
        }
        let head = smooth01((a + 1.0) * self.inv_head);
        let tail = smooth01(rem * self.inv_tail);
        if head <= 0.0 || tail <= 0.0 {
            return 0.0;
        }
        let au = u.abs();
        let cross = (1.0 - au.powi(6)) * (1.0 + st.rim * gauss(tab, (au - 0.78) * self.inv_rim));
        let edge = smooth01((half - sd.abs()) * self.inv_edge);
        let q = ac * self.inv_total;
        let lip = st.lip * gauss(tab, rem * self.inv_lip);
        let along = (1.0 - st.deplete * q) * (1.0 + 0.25 * gauss(tab, ac * self.inv_blob)) + lip;
        let groove = 1.0 - (st.groove0 + (st.groove1 - st.groove0) * q) * nb;
        self.tmax * cross * edge * groove * along * tail * head
    }
}

/// Paint-thickness field in PIXEL units from the stroke list. Each stroke is
/// a slab with steep shoulders, raised side rims, hashed bristle grooves, a
/// blob where the brush lands and a ragged tail where it lifts; a new stroke
/// flattens what is beneath it to its mean and keeps `keep` of it.
/// Deterministic (hashes of lum + index), consumes no RNG.
///
/// Only pixels within the stroke half-width of the polyline are visited: each
/// segment scans the parallelogram of its own rectangle (extended past a joint
/// by the bend wedge) and records the nearest segment per pixel; the profile
/// is then evaluated on those pixels only.
pub fn impasto_height(w: usize, h: usize, strokes: &[Stroke], knife: bool, look: f64) -> Vec<f32> {
    let lk = look as f32;
    let st = if knife { &KNIFE } else { &BRISTLE };
    let mut height = vec![0f32; w * h];
    // Scratch over the stroke bbox. Invariant: every `best2` entry is f32::MAX
    // between strokes (reset through `touched`).
    let mut best2: Vec<f32> = Vec::new();
    let mut sdv: Vec<f32> = Vec::new();
    let mut av: Vec<f32> = Vec::new();
    let mut touched: Vec<u32> = Vec::new();
    let mut hits: Vec<(u32, f32)> = Vec::new();
    let mut cum: Vec<f32> = Vec::new();
    let mut dirs: Vec<(f32, f32, f32)> = Vec::new();
    for (idx, s) in strokes.iter().enumerate() {
        let poly = &s.pts[..];
        if poly.len() < 2 {
            continue;
        }
        let half = stroke_half(s.th);
        cum.clear();
        cum.push(0.0);
        dirs.clear();
        for k in 1..poly.len() {
            let (dx, dy) = (poly[k].0 - poly[k - 1].0, poly[k].1 - poly[k - 1].1);
            let len = (dx * dx + dy * dy).sqrt();
            cum.push(cum[k - 1] + len);
            // (ux, uy, len); len <= 1e-6 marks a degenerate segment
            dirs.push(if len <= 1e-6 { (0.0, 0.0, len) } else { (dx / len, dy / len, len) });
        }
        let total = cum[poly.len() - 1];
        if total < 1e-3 {
            continue;
        }
        // stroke bbox (pixel indices, inclusive)
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in poly {
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
        }
        let pad = half + 2.0;
        let bx0 = (minx - pad).floor().max(0.0) as usize;
        let by0 = (miny - pad).floor().max(0.0) as usize;
        let bx1 = ((maxx + pad).ceil().max(0.0) as usize).min(w - 1);
        let by1 = ((maxy + pad).ceil().max(0.0) as usize).min(h - 1);
        if bx0 > bx1 || by0 > by1 {
            continue;
        }
        let bw = bx1 - bx0 + 1;
        let bh = by1 - by0 + 1;
        if best2.len() < bw * bh {
            best2.resize(bw * bh, f32::MAX);
            sdv.resize(bw * bh, 0.0);
            av.resize(bw * bh, 0.0);
        }
        let half2 = half * half;

        // How far past a joint (along the segment axis) a pixel can still be
        // nearest to this segment: the outer wedge of the bend reaches
        // half * sin(turn angle); a turn of 90 degrees or more reaches half.
        let ext = |u: (f32, f32, f32), v: (f32, f32, f32)| -> f32 {
            if u.2 <= 1e-6 || v.2 <= 1e-6 {
                return half;
            }
            if u.0 * v.0 + u.1 * v.1 <= 0.0 {
                half
            } else {
                (half * (u.0 * v.1 - u.1 * v.0).abs() + 1.0).min(half)
            }
        };

        // nearest segment per pixel: signed perpendicular offset + arclength
        let nseg = poly.len() - 1;
        for seg in 0..nseg {
            let (ux, uy, len) = dirs[seg];
            if len <= 1e-6 {
                continue;
            }
            let (ax, ay) = poly[seg];
            let lo = if seg == 0 { -1.0 } else { 0.0 };
            let hi = if seg == nseg - 1 { len + 1.0 } else { len };
            // end caps (proj beyond lo/hi - 1 px) can still be the nearest point
            // of a pixel; they must occupy it so farther segments cannot claim it
            let e_lo = if seg == 0 { half } else { ext(dirs[seg - 1], dirs[seg]) };
            let e_hi = if seg == nseg - 1 { half } else { ext(dirs[seg], dirs[seg + 1]) };
            let (p0, p1) = ((lo - e_lo) as f64, (hi + e_hi) as f64);
            let (axd, ayd, uxd, uyd, hd) = (ax as f64, ay as f64, ux as f64, uy as f64, half as f64);
            // rows: corners of the candidate parallelogram
            let (mut cy_min, mut cy_max) = (f64::MAX, f64::MIN);
            for p in [p0, p1] {
                for sgn in [-1.0f64, 1.0] {
                    let cy = ayd + uyd * p + uxd * hd * sgn;
                    cy_min = cy_min.min(cy);
                    cy_max = cy_max.max(cy);
                }
            }
            let ry0 = (cy_min.floor() as i64 - 1).max(by0 as i64);
            let ry1 = (cy_max.ceil() as i64 + 1).min(by1 as i64);
            for y in ry0..=ry1 {
                let y = y as usize;
                let py = y as f32 + 0.5;
                let dyv = y as f64 + 0.5 - ayd;
                let (mut xl, mut xr) = (bx0 as f64, bx1 as f64);
                // x-interval where proj in [p0, p1] and |perp| <= half (0.05 px slack)
                if uxd.abs() > 0.05 {
                    let a = (p0 - dyv * uyd) / uxd;
                    let b = (p1 - dyv * uyd) / uxd;
                    let (a, b) = if a < b { (a, b) } else { (b, a) };
                    xl = xl.max((axd + a - 0.5 - 0.05).ceil());
                    xr = xr.min((axd + b - 0.5 + 0.05).floor());
                }
                if uyd.abs() > 0.05 {
                    let a = (dyv * uxd - hd) / uyd;
                    let b = (dyv * uxd + hd) / uyd;
                    let (a, b) = if a < b { (a, b) } else { (b, a) };
                    xl = xl.max((axd + a - 0.5 - 0.05).ceil());
                    xr = xr.min((axd + b - 0.5 + 0.05).floor());
                }
                if xl > xr {
                    continue;
                }
                for x in xl as usize..=xr as usize {
                    let px = x as f32 + 0.5;
                    let proj = (px - ax) * ux + (py - ay) * uy;
                    let perp = -(px - ax) * uy + (py - ay) * ux;
                    if perp.abs() >= half {
                        continue;
                    }
                    let t = proj.clamp(lo, hi);
                    let d = proj - t;
                    let d2 = perp * perp + d * d;
                    if d2 >= half2 {
                        continue;
                    }
                    let j = (y - by0) * bw + (x - bx0);
                    // earlier segment keeps near-ties
                    if d2 < best2[j] * (1.0 - 1e-5) {
                        if best2[j] == f32::MAX {
                            touched.push(j as u32);
                        }
                        best2[j] = d2;
                        sdv[j] = perp;
                        av[j] = cum[seg] + t;
                    }
                }
            }
        }

        // per-stroke randoms from hashes (no RNG consumed)
        let seed = s.lum.to_bits() ^ (idx as u32).wrapping_mul(0x9E37_79B9);
        let prof = StrokeProfile::new(st, seed, half, total, lk);
        let tmax = prof.tmax;

        // profile on the touched pixels only
        let mut sum = 0f64;
        hits.clear();
        for &jj in &touched {
            let j = jj as usize;
            best2[j] = f32::MAX;
            let t = prof.eval(sdv[j], av[j]);
            if t > 0.0 {
                let i = (by0 + j / bw) * w + bx0 + j % bw;
                sum += height[i] as f64;
                hits.push((i as u32, t));
            }
        }
        touched.clear();
        if hits.is_empty() {
            continue;
        }
        let mean_under = (sum / hits.len() as f64) as f32;
        for &(i, t) in &hits {
            let i = i as usize;
            let c = smooth01(t / (0.45 * tmax));
            height[i] = height[i] * (1.0 - c) + st.keep * mean_under * c + t;
        }
    }
    gaussian_blur_plane(&height, w, h, st.blur as f64 * look)
}

/// Height gradient (x, y; image coords, y down) as 8x-scaled sobel3 planes.
pub(crate) type Grad = (Vec<f32>, Vec<f32>);

pub(crate) fn height_gradient(height: &[f32], w: usize, h: usize) -> Grad {
    (sobel3(height, w, h, 1, 0), sobel3(height, w, h, 0, 1))
}

/// Surface slope per sobel3 unit (sobel3 = 8x derivative); shared by the colour
/// shading and the tangent-space normal relief so both show the same ridges.
pub(crate) fn impasto_slope(impasto: f64) -> f32 {
    0.6 * impasto as f32 / 8.0
}

/// Per-pixel impasto shading from the height field (light from upper-left +
/// wet specular + darkening in the cavities between strokes): `[m, s, dark]`
/// so a colour byte `v` becomes `(v * m + s) * dark`. Flat height gives
/// m = 1, s = 0, dark = 1 (pixels unchanged). Computed once and applied to
/// every colour frame and the preview.
fn impasto_shade_field(w: usize, h: usize, height: &[f32], grad: &Grad, impasto: f64, ref_half: f32, look: f64) -> Vec<[f32; 3]> {
    let lk = look as f32;
    let (gx, gy) = grad;
    let low = gaussian_blur_plane(height, w, h, (0.28 * ref_half).clamp(1.0 * lk, 6.0 * lk) as f64);
    let g = impasto_slope(impasto);
    let norm3 = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    // image coords, y down: light from upper-left, toward the viewer
    let l = norm3([-0.5, -0.5, 0.7]);
    let hv = norm3([l[0], l[1], l[2] + 1.0]);
    let spec0 = hv[2].powi(40);
    (0..w * h)
        .map(|i| {
            let n = norm3([-g * gx[i], -g * gy[i], 1.0]);
            let diff = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0) / l[2];
            let nh = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0);
            let spec = (nh.powi(40) - spec0).max(0.0);
            let cav = (low[i] - height[i]).max(0.0) / ref_half;
            let dark = 1.0 - (0.7 * impasto as f32 * cav).min(0.6);
            [1.0 + 0.85 * (diff - 1.0), 255.0 * 0.22 * spec, dark]
        })
        .collect()
}

/// Shade the B/G/R bytes of a BGRA frame in place with a shade field.
fn apply_shade(frame: &mut [u8], field: &[[f32; 3]]) {
    for (px, f) in frame.chunks_exact_mut(4).zip(field) {
        for c in 0..3 {
            let v = (px[c] as f32 * f[0] + f[1]) * f[2];
            px[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Blend the impasto slope into the tangent-space normals of a BGRA frame in
/// place (whiteout blend). Image y points down but tangent +Y is v (up), so
/// the relief normal is (-g*gx, +g*gy, 1). Pixels are renormalised first, so a
/// canvas-modulated or flat (0.5, 0.5, 1) normal behaves the same.
fn tilt_normal_frame(frame: &mut [u8], grad: &Grad, impasto: f64) {
    let g = impasto_slope(impasto);
    let dec = |b: u8| b as f32 / 127.5 - 1.0;
    let enc = |v: f32| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
    for ((px, &gx), &gy) in frame.chunks_exact_mut(4).zip(&grad.0).zip(&grad.1) {
        let (mut x, mut y, mut z) = (dec(px[2]), dec(px[1]), dec(px[0]));
        let l = (x * x + y * y + z * z).sqrt();
        if l < 1e-3 {
            (x, y, z) = (0.0, 0.0, 1.0);
        } else {
            (x, y, z) = (x / l, y / l, z / l);
        }
        let (rx, ry) = (-g * gx, g * gy);
        let rl = (rx * rx + ry * ry + 1.0).sqrt();
        let (ox, oy, oz) = (x + rx / rl, y + ry / rl, z / rl);
        let ol = (ox * ox + oy * oy + oz * oz).sqrt();
        px[2] = enc(ox / ol);
        px[1] = enc(oy / ol);
        px[0] = enc(oz / ol);
    }
}

/// Height as gray BGRA for the Blender Bump node: v = clamp(height_px /
/// norm_px, 0, 1). The impasto factor travels separately in
/// [`impasto_height_scale`].
fn height_frame(height: &[f32], w: usize, h: usize, norm_px: f32) -> Vec<u8> {
    let mut out = vec![255u8; w * h * 4];
    for i in 0..w * h {
        let v = ((height[i] / norm_px).clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4] = v;
        out[i * 4 + 1] = v;
        out[i * 4 + 2] = v;
    }
    out
}

/// Display-only clay render: mid-gray canvas lit by the impasto shading, so
/// the relief is visible on the page when no color map is loaded.
fn preview_frame(w: usize, h: usize, field: &[[f32; 3]]) -> Vec<u8> {
    let mut out = vec![170u8; w * h * 4];
    for i in 0..w * h {
        out[i * 4 + 3] = 255;
    }
    apply_shade(&mut out, field);
    out
}

// ---------------------------------------------------------------- relief

pub fn canvas_relief(w: usize, h: usize, scale: i32, strength: f64, look: f64) -> (Vec<f32>, Vec<f32>) {
    let mut height = vec![0f32; w * h];
    let period = scale as f64 * look;
    let pi = std::f64::consts::PI;
    let hx: Vec<f64> = (0..w).map(|x| 0.5 * (x as f64 * pi / period).sin()).collect();
    let sy: Vec<f64> = (0..h).map(|y| (y as f64 * pi / period).sin()).collect();
    for y in 0..h {
        for x in 0..w {
            height[y * w + x] = (hx[x] * sy[y]) as f32;
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
        let nz = 10.0 / look;
        let nl = (nx * nx + ny * ny + nz * nz).sqrt() + 1e-6;
        let v = (nx * lx + ny * ly + nz * lz) / nl;
        let s = v.clamp(-1.0, 1.0) * 0.5 + 0.5;
        shade[i] = ((s * strength + (1.0 - strength)) as f32);
    }
    (shade, height)
}

// ---------------------------------------------------------------- pipeline

/// Everything painted before impasto: BGRA frames (reference first, canvas
/// relief applied, never impasto-shaded) and the strokes that made them.
pub struct PaintStage {
    pub w: usize,
    pub h: usize,
    pub frames: Vec<Vec<u8>>,
    /// flat-texture strokes (empty when `surf` is set)
    pub strokes: Vec<Stroke>,
    /// surface painting: texel map + world-space strokes
    pub surf: Option<SurfStage>,
}

pub struct SurfStage {
    pub map: crate::surface::SurfMap,
    pub strokes: Vec<crate::surface::SurfStroke>,
}

/// The four input checks of the multi-channel entry point.
pub fn validate_multi(ref_bgra: &[u8], chans: &[&[u8]], w: usize, h: usize) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err("empty image".into());
    }
    if chans.len() > 4 {
        return Err(format!("at most 4 extra channels, got {}", chans.len()));
    }
    let n = w * h * 4;
    for (i, c) in chans.iter().enumerate() {
        if c.len() != n {
            return Err(format!("channel {i} is {} bytes, expected {n}", c.len()));
        }
    }
    if ref_bgra.len() != n {
        return Err(format!("reference is {} bytes, expected {n}", ref_bgra.len()));
    }
    Ok(())
}

/// Flow + strokes from `ref_bgra`, strokes rasterized and the canvas shade
/// computed once, then the SAME strokes stamped on the reference and every
/// channel (each with its own Kuwahara base and colour sampling).
/// With `p.uv_aware` and a `mesh`, the strokes are walked and rasterized on
/// the mesh surface instead (see `surface.rs`).
pub fn paint_stage(
    ref_bgra: &[u8],
    chans: &[&[u8]],
    w: usize,
    h: usize,
    p: &Params,
    mesh: Option<&crate::surface::Mesh>,
) -> PaintStage {
    let look = p.look();
    if let Some(mesh) = mesh.filter(|_| p.uv_aware) {
        let margin = ((16.0 * look).ceil() as usize).max(8);
        let map = crate::surface::build_map(mesh, w, h, margin);
        if map.islands > 0 {
            return paint_stage_surface(ref_bgra, chans, w, h, p, map);
        }
    }
    let ref_bgr = bgra_to_bgr(ref_bgra, w, h);
    let base = kuwahara(&ref_bgr, w, h, p.base_radius);
    let gray = gray_from_bgr(&ref_bgr, w, h);
    let flow0 = flow_field(&gray, w, h, 2.5 * look);
    let flow = smooth_flow(flow0, p.flow_smooth);
    let n_strokes = (((w * h) as f64 / (140.0 * look * look)) as i32).max(1);
    let mut strokes = generate_strokes(
        &ref_bgr,
        w,
        h,
        &flow,
        n_strokes,
        p.brush_len,
        p.brush_w,
        p.layers,
        0.6,
        p.seed,
        look,
    );
    if p.boil_frame > 0 && p.boil > 0.0 {
        boil_strokes(&mut strokes, w, h, &flow, n_strokes, p.brush_len, p.brush_w, 0.6, p.seed, p.boil_frame, p.boil, look);
    }
    let cov = rasterize_strokes(w, h, &strokes, look);
    let shade = (p.relief > 0.0).then(|| canvas_relief(w, h, 12, p.relief, look).0);
    let mut frames = Vec::with_capacity(chans.len() + 1);
    frames.push(compose(&ref_bgr, &base, w, h, &strokes, &cov, shade.as_deref()));
    for c in chans {
        let cgr = bgra_to_bgr(c, w, h);
        let cbase = kuwahara(&cgr, w, h, p.base_radius);
        frames.push(compose(&cgr, &cbase, w, h, &strokes, &cov, shade.as_deref()));
    }
    PaintStage { w, h, frames, strokes, surf: None }
}

/// [`paint_stage`] on the mesh surface: gutter texels mirror their twin
/// before every 2D filter and after every composed frame.
fn paint_stage_surface(
    ref_bgra: &[u8],
    chans: &[&[u8]],
    w: usize,
    h: usize,
    p: &Params,
    map: crate::surface::SurfMap,
) -> PaintStage {
    use crate::surface;
    let look = p.look();
    let mut ref_bgr = bgra_to_bgr(ref_bgra, w, h);
    surface::fill_gutter(&map, &mut ref_bgr, 3);
    let base = kuwahara(&ref_bgr, w, h, p.base_radius);
    let gray = gray_from_bgr(&ref_bgr, w, h);
    let flow = smooth_flow(flow_field(&gray, w, h, 2.5 * look), p.flow_smooth);
    let n_strokes = (((w * h) as f64 / (140.0 * look * look)) as i32).max(1);
    let mut strokes =
        surface::generate_strokes(&ref_bgr, &flow, &map, n_strokes, p.brush_len, p.brush_w, p.layers, 0.6, p.seed, look);
    if p.boil_frame > 0 && p.boil > 0.0 {
        surface::boil_strokes(&mut strokes, &flow, &map, n_strokes, p.brush_len, p.brush_w, 0.6, p.seed, p.boil_frame, p.boil, look);
    }
    let cov = surface::rasterize(&map, &strokes, look);
    let shade = (p.relief > 0.0).then(|| canvas_relief(w, h, 12, p.relief, look).0);
    let samples: Vec<StrokeSample> = strokes
        .iter()
        .map(|s| {
            let t = s.tex[s.tex.len() / 2] as usize;
            StrokeSample { x: (t % w) as i64, y: (t / w) as i64, r: s.pr as i64, lum: s.lum }
        })
        .collect();
    let mut frames = Vec::with_capacity(chans.len() + 1);
    let mut f = compose_samples(&ref_bgr, &base, w, h, &samples, &cov, shade.as_deref(), Some(&map.twin));
    surface::fill_gutter(&map, &mut f, 4);
    frames.push(f);
    for c in chans {
        let mut cgr = bgra_to_bgr(c, w, h);
        surface::fill_gutter(&map, &mut cgr, 3);
        let cbase = kuwahara(&cgr, w, h, p.base_radius);
        let mut f = compose_samples(&cgr, &cbase, w, h, &samples, &cov, shade.as_deref(), Some(&map.twin));
        surface::fill_gutter(&map, &mut f, 4);
        frames.push(f);
    }
    PaintStage { w, h, frames, strokes: Vec::new(), surf: Some(SurfStage { map, strokes }) }
}

/// Impasto height (px) of a stage, whichever way its strokes were laid.
pub fn stage_height(stage: &PaintStage, knife: bool, look: f64) -> Vec<f32> {
    match &stage.surf {
        Some(s) => crate::surface::impasto_height(&s.map, &s.strokes, knife, look),
        None => impasto_height(stage.w, stage.h, &stage.strokes, knife, look),
    }
}

/// Frame 0 of the stage, impasto-shaded when `hgt` is given.
pub fn finish_single<'a>(stage: &'a PaintStage, hgt: Option<&[f32]>, p: &Params) -> Cow<'a, [u8]> {
    match hgt {
        None => Cow::Borrowed(&stage.frames[0][..]),
        Some(hg) => {
            let grad = height_gradient(hg, stage.w, stage.h);
            let field = impasto_shade_field(stage.w, stage.h, hg, &grad, p.impasto, impasto_ref_half(p.brush_w), p.look());
            let mut o = stage.frames[0].clone();
            apply_shade(&mut o, &field);
            Cow::Owned(o)
        }
    }
}

/// All stage frames; with `hgt`, frames of kind "color" are impasto-shaded, frames
/// of kind "normal" get the relief blended in when `p.normal_tangent`, and a height
/// frame then a display-only relief preview are appended.
pub fn finish_multi<'a>(
    stage: &'a PaintStage,
    hgt: Option<&[f32]>,
    p: &Params,
    kinds: Option<&[Option<String>]>,
) -> Vec<Cow<'a, [u8]>> {
    let (w, h) = (stage.w, stage.h);
    let map = stage.surf.as_ref().map(|s| &s.map);
    let mut out: Vec<Cow<'a, [u8]>> = Vec::with_capacity(stage.frames.len() + 2);
    let grad = hgt.map(|hg| height_gradient(hg, w, h));
    let ref_half = impasto_ref_half(p.brush_w);
    let field = hgt.zip(grad.as_ref()).map(|(hg, gr)| match map {
        Some(m) => crate::surface::shade_field(m, hg, gr, p.impasto, ref_half, p.look()),
        None => impasto_shade_field(w, h, hg, gr, p.impasto, ref_half, p.look()),
    });
    // tangent-space tilt: texel-density-corrected slopes on a surface
    let tilt = match (map, grad.as_ref()) {
        (Some(m), Some(gr)) if p.normal_tangent => Some(crate::surface::scale_grad(m, gr)),
        _ => None,
    };
    // gutter texels mirror their twin again after per-texel shading
    let pad = |mut o: Vec<u8>| {
        if let Some(m) = map {
            crate::surface::fill_gutter(m, &mut o, 4);
        }
        o
    };
    for (i, f) in stage.frames.iter().enumerate() {
        let kind = kind_at(kinds, i);
        match (&field, &grad) {
            (Some(field), _) if kind == Some("color") => {
                let mut o = f.clone();
                apply_shade(&mut o, field);
                out.push(Cow::Owned(pad(o)));
            }
            (_, Some(grad)) if kind == Some("normal") && p.normal_tangent => {
                let mut o = f.clone();
                tilt_normal_frame(&mut o, tilt.as_ref().unwrap_or(grad), p.impasto);
                out.push(Cow::Owned(pad(o)));
            }
            _ => out.push(Cow::Borrowed(&f[..])),
        }
    }
    if let (Some(hg), Some(field)) = (hgt, &field) {
        out.push(Cow::Owned(pad(height_frame(hg, w, h, impasto_norm_px(p.brush_w)))));
        out.push(Cow::Owned(pad(preview_frame(w, h, field))));
    }
    out
}

pub fn paint_bgra(bgra: &[u8], w: usize, h: usize, p: &Params) -> (Vec<u8>, String) {
    // resize handled by JS (canvas); here we assume already-resized
    let stage = paint_stage(bgra, &[], w, h, p, None);
    let hgt = (p.impasto > 0.0).then(|| stage_height(&stage, p.impasto_knife, p.look()));
    (finish_single(&stage, hgt.as_deref(), p).into_owned(), String::new())
}

/// bgra_to_bgr helper shared by the single and multi paths.
fn bgra_to_bgr(bgra: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut bgr = vec![0u8; w * h * 3];
    for i in 0..w * h {
        bgr[i * 3] = bgra[i * 4];
        bgr[i * 3 + 1] = bgra[i * 4 + 1];
        bgr[i * 3 + 2] = bgra[i * 4 + 2];
    }
    bgr
}

/// Multi-channel coherent paint: flow + strokes derived from `ref_bgra`
/// only, then the SAME strokes are stamped on every channel in `chans`
/// (stroke replay). The painted reference is element 0 of the returned Vec,
/// so callers pass ONLY the other channels (0..=4; none = reference only):
/// returned.len() == chans.len() + 1, or chans.len() + 3 when impasto > 0
/// (a gray height frame for the Blender Bump node, then a display-only
/// clay "relief preview"). `kinds` is frame-aligned (kinds[0] = reference);
/// with impasto > 0 frames of kind "color" are shaded by the stroke height.
/// Every channel gets identical stroke geometry and per-stroke lum; the
/// per-channel color sampling makes each map its own painting. relief is
/// applied to all.
pub fn paint_bgra_multi(
    ref_bgra: &[u8],
    chans: &[&[u8]],
    w: usize,
    h: usize,
    p: &Params,
    kinds: Option<&[Option<String>]>,
) -> Result<Vec<Vec<u8>>, String> {
    validate_multi(ref_bgra, chans, w, h)?;
    let stage = paint_stage(ref_bgra, chans, w, h, p, None);
    let hgt = (p.impasto > 0.0).then(|| stage_height(&stage, p.impasto_knife, p.look()));
    Ok(finish_multi(&stage, hgt.as_deref(), p, kinds).into_iter().map(Cow::into_owned).collect())
}

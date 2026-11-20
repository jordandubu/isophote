//! `impasto_height` (pruned scan + table Gaussians) vs the original
//! bbox-scan implementation kept below verbatim as `impasto_height_ref`.
use isophote_core::pipeline::*;

fn stroke_half(th: i32) -> f32 {
    if th <= 1 { 1.5 } else { ((th + 1) / 2) as f32 + 0.5 }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash01(seed: u32, i: u32) -> f32 {
    let mut x = seed ^ i.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x as f32 / 4_294_967_296.0
}

struct ImpastoStyle {
    t_k: f32,
    keep: f32,
    rim: f32,
    groove0: f32,
    groove1: f32,
    rag: f32,
    blur: f32,
    lip: f32,
    deplete: f32,
}

const BRISTLE: ImpastoStyle = ImpastoStyle {
    t_k: 0.7, keep: 0.35, rim: 0.5, groove0: 0.18, groove1: 0.55, rag: 1.0, blur: 0.8, lip: 0.3, deplete: 0.4,
};
const KNIFE: ImpastoStyle = ImpastoStyle {
    t_k: 0.8, keep: 0.30, rim: 0.7, groove0: 0.04, groove1: 0.15, rag: 0.5, blur: 0.7, lip: 0.5, deplete: 0.25,
};

fn impasto_height_ref(w: usize, h: usize, strokes: &[Stroke], knife: bool, look: f64) -> Vec<f32> {
    let lk = look as f32;
    let st = if knife { &KNIFE } else { &BRISTLE };
    let mut height = vec![0f32; w * h];
    let mut best: Vec<f32> = Vec::new();
    let mut sdv: Vec<f32> = Vec::new();
    let mut av: Vec<f32> = Vec::new();
    let mut tv: Vec<f32> = Vec::new();
    let mut cum: Vec<f32> = Vec::new();
    for (idx, s) in strokes.iter().enumerate() {
        let poly = &s.pts[..];
        if poly.len() < 2 {
            continue;
        }
        let half = stroke_half(s.th);
        cum.clear();
        cum.push(0.0);
        for k in 1..poly.len() {
            let (dx, dy) = (poly[k].0 - poly[k - 1].0, poly[k].1 - poly[k - 1].1);
            cum.push(cum[k - 1] + (dx * dx + dy * dy).sqrt());
        }
        let total = cum[poly.len() - 1];
        if total < 1e-3 {
            continue;
        }
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
        best.clear();
        best.resize(bw * bh, f32::MAX);
        sdv.clear();
        sdv.resize(bw * bh, 0.0);
        av.clear();
        av.resize(bw * bh, 0.0);
        tv.clear();
        tv.resize(bw * bh, 0.0);

        for seg in 0..poly.len() - 1 {
            let (ax, ay) = poly[seg];
            let (bx, by) = poly[seg + 1];
            let (dx, dy) = (bx - ax, by - ay);
            let len = (dx * dx + dy * dy).sqrt();
            if len <= 1e-6 {
                continue;
            }
            let (ux, uy) = (dx / len, dy / len);
            let lo = if seg == 0 { -1.0 } else { 0.0 };
            let hi = if seg == poly.len() - 2 { len + 1.0 } else { len };
            let sx0 = ((ax.min(bx) - pad).floor().max(bx0 as f32)) as usize;
            let sy0 = ((ay.min(by) - pad).floor().max(by0 as f32)) as usize;
            let sx1 = ((ax.max(bx) + pad).ceil().max(0.0) as usize).min(bx1);
            let sy1 = ((ay.max(by) + pad).ceil().max(0.0) as usize).min(by1);
            for y in sy0..=sy1 {
                let py = y as f32 + 0.5;
                for x in sx0..=sx1 {
                    let px = x as f32 + 0.5;
                    let proj = (px - ax) * ux + (py - ay) * uy;
                    let perp = -(px - ax) * uy + (py - ay) * ux;
                    let t = proj.clamp(lo, hi);
                    let dist = perp.hypot(proj - t);
                    let j = (y - by0) * bw + (x - bx0);
                    if dist < best[j] {
                        best[j] = dist;
                        sdv[j] = perp;
                        av[j] = cum[seg] + t;
                    }
                }
            }
        }

        let seed = s.lum.to_bits() ^ (idx as u32).wrapping_mul(0x9E37_79B9);
        let kn = (((2.0 * half / (2.2 * lk)).round()) as usize).clamp(3, 24);
        let mut gb = [0f32; 24];
        let mut ge = [0f32; 24];
        for i in 0..kn {
            gb[i] = hash01(seed, i as u32);
            ge[i] = hash01(seed, (kn + i) as u32);
        }
        let load = 0.6 + 0.8 * hash01(seed, (2 * kn) as u32);
        let tmax = st.t_k * half * load;

        let mut sum = 0f64;
        let mut cnt = 0usize;
        for j in 0..bw * bh {
            if best[j] >= half {
                continue;
            }
            let sd = sdv[j];
            let a = av[j];
            let u = (sd / half).clamp(-1.0, 1.0);
            let fx = (u * 0.5 + 0.5) * (kn - 1) as f32;
            let i0 = (fx.floor() as usize).min(kn - 2);
            let mut fr = fx - i0 as f32;
            fr = fr * fr * (3.0 - 2.0 * fr);
            let nb = gb[i0] * (1.0 - fr) + gb[i0 + 1] * fr;
            let ne = ge[i0] * (1.0 - fr) + ge[i0 + 1] * fr;
            let au = u.abs();
            let cross = (1.0 - au.powi(6)) * (1.0 + st.rim * (-((au - 0.78) / 0.13).powi(2)).exp());
            let edge = smoothstep(0.0, 1.4 * lk, half - sd.abs());
            let ac = a.clamp(0.0, total);
            let q = ac / total;
            let leff = total - st.rag * half * ne;
            let tail = smoothstep(0.0, 0.8 * half, leff - a);
            let head = smoothstep(-1.0, 0.5 * half, a);
            let lip = if leff - a > 0.0 {
                st.lip * (-((leff - a) / (0.45 * half)).powi(2)).exp()
            } else {
                0.0
            };
            let along = (1.0 - st.deplete * q) * (1.0 + 0.25 * (-(ac / (0.9 * half)).powi(2)).exp()) + lip;
            let groove = 1.0 - (st.groove0 + (st.groove1 - st.groove0) * q) * nb;
            let t = tmax * cross * edge * groove * along * tail * head;
            tv[j] = t;
            if t > 0.0 {
                let i = (by0 + j / bw) * w + bx0 + j % bw;
                sum += height[i] as f64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        let mean_under = (sum / cnt as f64) as f32;
        for j in 0..bw * bh {
            let t = tv[j];
            if t > 0.0 {
                let i = (by0 + j / bw) * w + bx0 + j % bw;
                let c = smoothstep(0.0, 0.45 * tmax, t);
                height[i] = height[i] * (1.0 - c) + st.keep * mean_under * c + t;
            }
        }
    }
    gaussian_blur_plane(&height, w, h, st.blur as f64 * look)
}

fn synth_bgr(w: usize, h: usize) -> Vec<u8> {
    let mut v = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            v[i] = ((x * 5 + y * y / 7) % 256) as u8;
            v[i + 1] = (((x ^ y) * 3) % 256) as u8;
            v[i + 2] = ((x * y / 16 + 90 * ((x / 23 + y / 17) % 2)) % 256) as u8;
        }
    }
    v
}

fn natural(w: usize, h: usize, len: i32, width: i32, look: f64) -> Vec<Stroke> {
    let bgr = synth_bgr(w, h);
    let flow = smooth_flow(flow_field(&gray_from_bgr(&bgr, w, h), w, h, 2.5 * look), 4);
    let n = (((w * h) as f64 / (140.0 * look * look)) as i32).max(1);
    generate_strokes(&bgr, w, h, &flow, n, len, width, 2, 0.6, 7, look)
}

fn check(name: &str, w: usize, h: usize, strokes: &[Stroke], knife: bool, look: f64) {
    assert!(!strokes.is_empty(), "{name}: no strokes");
    let new = impasto_height(w, h, strokes, knife, look);
    let old = impasto_height_ref(w, h, strokes, knife, look);
    let mean_old = old.iter().map(|v| *v as f64).sum::<f64>() / old.len() as f64;
    let mean_diff = new.iter().zip(&old).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / old.len() as f64;
    let max_diff = new.iter().zip(&old).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let big = new.iter().zip(&old).filter(|(a, b)| (**a - **b).abs() > 1.0).count();
    println!("{name}: mean(ref) {mean_old:.4} mean|diff| {mean_diff:.6} max|diff| {max_diff:.4} px>1.0: {big}/{}", old.len());
    assert!(mean_old > 0.0, "{name}: reference height is empty");
    assert!(mean_diff <= 0.005 * mean_old, "{name}: mean |diff| {mean_diff} vs mean {mean_old}");
    assert!(big * 1000 <= old.len(), "{name}: {big} pixels differ by more than 1.0");
}

#[test]
fn impasto_height_matches_reference() {
    check("small", 160, 120, &natural(160, 120, 30, 8, 1.0), false, 1.0);
    let big = natural(200, 150, 63, 37, 1.0);
    check("bold-bristle", 200, 150, &big, false, 1.0);
    check("bold-knife", 200, 150, &big, true, 1.0);
    check("look04", 160, 120, &natural(160, 120, 12, 3, 0.4), false, 0.4);
}

#[test]
fn impasto_height_wedge_bend() {
    // 240-degree arc: bends sharper than the stroke is wide, so pixels on the
    // outer side of a joint must still find their nearest segment
    let pts: Vec<(f32, f32)> = (0..9)
        .map(|i| {
            let a = (i as f32 * 30.0).to_radians();
            (80.0 + 40.0 * a.cos(), 60.0 + 40.0 * a.sin())
        })
        .collect();
    let strokes = vec![Stroke { pts, th: 20, pr: 10, lum: 1.0 }];
    check("wedge-bristle", 160, 120, &strokes, false, 1.0);
    check("wedge-knife", 160, 120, &strokes, true, 1.0);
}

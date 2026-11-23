//! Surface painting (`uv_aware`): strokes, impasto and gutters must continue
//! across UV seams. Fixture: a unit plane cut along x = 0.5 into two UV
//! islands, the right one rotated 90 degrees in UV space.
use isophote_core::pipeline::{finish_multi, paint_bgra_multi, paint_stage, stage_height, Flow, Params};
use isophote_core::surface::*;
use isophote_core::{wasm_paint_multi, wasm_paint_multi_frames, wasm_set_mesh};

const W: usize = 256;
/// world units per texel of both islands
const UPP: f32 = 1.0 / 128.0;

fn uv_a(p: [f32; 3]) -> [f32; 2] {
    [0.05 + 0.5 * p[0], 0.05 + 0.5 * p[1]]
}
fn uv_b(p: [f32; 3]) -> [f32; 2] {
    [0.90 - 0.5 * p[1], 0.65 + 0.5 * (p[0] - 0.5)]
}

fn plane() -> Mesh {
    let vpos = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]];
    // (triangle, on island A)
    let tris: [([u32; 3], bool); 4] = [([0, 1, 4], true), ([0, 4, 3], true), ([1, 2, 5], false), ([1, 5, 4], false)];
    let mut m = Mesh { vpos, tri_vert: Vec::new(), tri_uv: Vec::new(), tri_nrm: Vec::new() };
    for (t, a) in tris {
        let uv = |v: u32| if a { uv_a(m.vpos[v as usize]) } else { uv_b(m.vpos[v as usize]) };
        m.tri_uv.push([uv(t[0]), uv(t[1]), uv(t[2])]);
        m.tri_vert.push(t);
        m.tri_nrm.push([[0.0, 0.0, 1.0]; 3]);
    }
    m
}

/// The add-on's wire format (see `Mesh::parse`).
fn blob(m: &Mesh) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"ISOM");
    for v in [1u32, m.tri_vert.len() as u32, m.vpos.len() as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [1f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for p in &m.vpos {
        p.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    }
    for t in &m.tri_vert {
        t.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    }
    for t in &m.tri_uv {
        t.iter().flatten().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    }
    for t in &m.tri_nrm {
        t.iter().flatten().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    }
    b
}

/// Texel index of a UV coordinate.
fn texel(uv: [f32; 2]) -> usize {
    let (x, y) = ((uv[0] * W as f32) as usize, ((1.0 - uv[1]) * W as f32) as usize);
    y * W + x
}

/// BGRA image whose colour depends on the world position only, so it is
/// continuous across the seam.
fn world_image(map: &SurfMap) -> Vec<u8> {
    let mut img = vec![0u8; W * W * 4];
    for i in 0..W * W {
        img[i * 4 + 3] = 255;
        if map.tri[i] == NONE {
            continue;
        }
        let p = map.pos(i, (i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
        img[i * 4] = (128.0 + 100.0 * (12.0 * (p[0] + 0.6 * p[1])).sin()) as u8;
        img[i * 4 + 1] = (128.0 + 100.0 * (9.0 * (p[1] - 0.3 * p[0])).cos()) as u8;
        img[i * 4 + 2] = 128;
    }
    fill_gutter(map, &mut img, 4);
    img
}

fn params(uv_aware: bool) -> Params {
    Params {
        brush_len: 30, brush_w: 8, layers: 2, relief: 0.0, base_radius: 3, flow_smooth: 4, seed: 7,
        max_size: None, impasto: 1.0, impasto_knife: false, boil: 0.0, boil_frame: 0,
        look_scale: 0.0, out_size: 0, normal_tangent: false, uv_aware, shape_follow: 0.0,
    }
}

#[test]
fn gutter_continues_across_the_seam() {
    let map = build_map(&plane(), W, W, 16);
    assert!((map.upp - UPP).abs() < 1e-6, "upp {}", map.upp);
    let mut seam = 0;
    for i in 0..W * W {
        if map.tri[i] != NONE || map.gedge[i] == NONE {
            continue;
        }
        seam += 1;
        let (x, y) = ((i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
        let (qx, qy) = map.across(map.gedge[i], x, y);
        let twin = map.twin[i] as usize;
        assert_eq!(map.texel(qx, qy), Some(twin));
        let got = map.pos(twin, qx, qy);
        // where this gutter texel would be if its own island's chart went on
        let (u, v) = (x / W as f32, 1.0 - y / W as f32);
        let want = if got[0] > 0.5 {
            [(u - 0.05) / 0.5, (v - 0.05) / 0.5] // beside island A
        } else {
            [(v - 0.65) / 0.5 + 0.5, (0.90 - u) / 0.5] // beside island B
        };
        assert!((got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3, "texel {i}: {got:?} vs {want:?}");
    }
    assert!(seam > 1000, "only {seam} seam gutter texels");
}

/// A stroke walked in +x from island A, with its footprint.
fn crossing_stroke(map: &SurfMap) -> SurfStroke {
    // flow = world +x on both islands (B is rotated, so that is "up" there)
    let angle: Vec<f32> = (0..W * W).map(|i| if i / W < 100 { 90.0 } else { 0.0 }).collect();
    let flow = Flow { angle, coherence: vec![1.0; W * W], w: W, h: W };
    let uv = uv_a([0.40, 0.5, 0.0]);
    let start = (uv[0] * W as f32, (1.0 - uv[1]) * W as f32);
    let (pts, nrm, tex) = trace(start.0, start.1, &flow, map, 60, 1.0).expect("seed is on island A");
    SurfStroke { pts, nrm, tex, start, th: 9, pr: 4, lum: 1.0 }
}

#[test]
fn stroke_walks_across_the_seam() {
    let map = build_map(&plane(), W, W, 16);
    let s = crossing_stroke(&map);
    assert_eq!(s.pts.len(), 31);
    for k in 1..s.pts.len() {
        let d = ((s.pts[k][0] - s.pts[k - 1][0]).powi(2) + (s.pts[k][1] - s.pts[k - 1][1]).powi(2)).sqrt();
        assert!((d - 2.0 * UPP).abs() < 0.02 * UPP, "step {k} is {d}");
        assert!((s.pts[k][1] - 0.5).abs() < 1e-3, "node {k} left the line: {:?}", s.pts[k]);
    }
    assert!(s.pts[0][0] < 0.45 && s.pts[30][0] > 0.8, "{:?} .. {:?}", s.pts[0], s.pts[30]);
}

#[test]
fn footprint_matches_brute_force_on_both_islands() {
    let map = build_map(&plane(), W, W, 16);
    let s = crossing_stroke(&map);
    let mut hits = Vec::new();
    gather(&map, &s, &mut Scratch::new(W * W), &mut hits);
    // the stroke is the straight line y = 0.5 from x0 - UPP to x1 + UPP
    let (x0, x1) = (s.pts[0][0] - UPP, s.pts[30][0] + UPP);
    let reach = 5.5 * UPP; // stroke_half(9) texels
    let dist = |p: [f32; 3]| {
        let dx = (x0 - p[0]).max(p[0] - x1).max(0.0);
        (dx * dx + (p[1] - 0.5).powi(2)).sqrt()
    };
    let mut hit = vec![false; W * W];
    let (mut on_a, mut on_b) = (0, 0);
    for h in &hits {
        let i = h.i as usize;
        hit[i] = true;
        let p = map.pos(i, (i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
        assert!(dist(p) < reach * 1.001, "texel {i} is outside the stroke");
        assert!((h.d2.sqrt() * UPP - dist(p)).abs() < 0.02 * UPP, "texel {i}: wrong distance");
        if p[0] < 0.5 { on_a += 1 } else { on_b += 1 }
    }
    assert!(on_a > 100 && on_b > 100, "hits: {on_a} on A, {on_b} on B");
    for i in 0..W * W {
        if map.tri[i] == NONE {
            continue;
        }
        let p = map.pos(i, (i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
        assert!(hit[i] || dist(p) > reach * 0.999, "texel {i} at {p:?} was missed");
    }
}

#[test]
fn impasto_is_continuous_across_the_seam() {
    let mesh = plane();
    let map = build_map(&mesh, W, W, 16);
    let img = world_image(&map);
    let seam_jump = |uv_aware: bool| -> (f64, f64) {
        let p = params(uv_aware);
        let stage = paint_stage(&img, &[], W, W, &p, uv_aware.then_some(&mesh));
        assert_eq!(stage.surf.is_some(), uv_aware);
        let hgt = stage_height(&stage, false, 1.0);
        if uv_aware {
            // painted gutter texels repeat their twin
            for i in 0..W * W {
                let t = map.twin[i] as usize;
                if map.tri[i] == NONE && map.twin[i] != NONE {
                    assert_eq!(stage.frames[0][i * 4..i * 4 + 4], stage.frames[0][t * 4..t * 4 + 4], "gutter texel {i}");
                    assert_eq!(hgt[i], hgt[t], "gutter height {i}");
                }
            }
        }
        // texel pairs 1.5 texels apart in the world, one on each island
        let (mut jump, mut mean) = (0f64, 0f64);
        for k in 0..80 {
            let y = 0.05 + 0.9 * k as f32 / 79.0;
            let a = hgt[texel(uv_a([0.5 - 0.75 * UPP, y, 0.0]))];
            let b = hgt[texel(uv_b([0.5 + 0.75 * UPP, y, 0.0]))];
            jump += (a - b).abs() as f64 / 80.0;
            mean += (a + b) as f64 / 160.0;
        }
        (jump, mean)
    };
    let (flat, flat_mean) = seam_jump(false);
    let (surf, surf_mean) = seam_jump(true);
    println!("seam jump: flat {flat:.4} (mean height {flat_mean:.4}), surface {surf:.4} (mean height {surf_mean:.4})");
    assert!(surf_mean > 0.5, "no paint at the seam");
    assert!(surf < 0.5 * flat, "height still breaks at the seam: {surf} vs {flat}");
}

#[test]
fn wasm_entry_paints_on_the_mesh() {
    let mesh = plane();
    let map = build_map(&mesh, W, W, 16);
    let img = world_image(&map);
    let kinds = vec![Some("color".to_string())];
    let json = "{\"brush_len\":30,\"brush_w\":8,\"layers\":2,\"relief\":0.0,\"base_radius\":3,\"flow_smooth\":4,\
                \"seed\":7,\"impasto\":1.0,\"uv_aware\":true,\"kinds\":[\"color\"]}";
    let run = || {
        let mut out = vec![0u8; 3 * W * W * 4];
        let ret = wasm_paint_multi(W as i32, W as i32, 1, img.as_ptr(), out.as_mut_ptr(), json.as_ptr(), json.len() as i32);
        assert_eq!(ret, 0);
        assert_eq!(wasm_paint_multi_frames(), 3);
        out
    };

    let b = blob(&mesh);
    assert_eq!(wasm_set_mesh(b.as_ptr(), b.len() as i32), 0);
    let p = params(true);
    let stage = paint_stage(&img, &[], W, W, &p, Some(&mesh));
    let hgt = stage_height(&stage, false, 1.0);
    let want: Vec<u8> = finish_multi(&stage, Some(&hgt), &p, Some(&kinds)).iter().flat_map(|f| f.iter().copied()).collect();
    assert!(run() == want, "uv_aware paint differs from the surface pipeline");

    // a malformed blob is refused and clears the mesh: classic paint
    assert_eq!(wasm_set_mesh(b.as_ptr(), b.len() as i32 - 1), 1);
    let flat: Vec<u8> = paint_bgra_multi(&img, &[], W, W, &p, Some(&kinds)).unwrap().concat();
    assert!(run() == flat, "without a mesh the paint must be the classic one");
    assert!(flat != want);
}

/// Closed-ish tube (axis z, radius 1, length 2) or uv-sphere (`sphere`), one
/// UV island, radial corner normals.
fn round_mesh(sphere: bool) -> Mesh {
    let (nu, nv) = (48usize, 24usize);
    let at = |i: usize, j: usize| -> ([f32; 3], [f32; 3], [f32; 2]) {
        let a = std::f32::consts::TAU * (i % nu) as f32 / nu as f32;
        let (p, n) = if sphere {
            let b = std::f32::consts::PI * (0.04 + 0.92 * j as f32 / nv as f32);
            let n = [b.sin() * a.cos(), b.sin() * a.sin(), b.cos()];
            (n, n)
        } else {
            ([a.cos(), a.sin(), 2.0 * j as f32 / nv as f32 - 1.0], [a.cos(), a.sin(), 0.0])
        };
        (p, n, [0.05 + 0.9 * i as f32 / nu as f32, 0.05 + 0.9 * j as f32 / nv as f32])
    };
    let mut m = Mesh { vpos: Vec::new(), tri_vert: Vec::new(), tri_uv: Vec::new(), tri_nrm: Vec::new() };
    for j in 0..nv {
        for i in 0..nu {
            let quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let base = m.vpos.len() as u32;
            for &(a, b) in &quad {
                m.vpos.push(at(a, b).0);
            }
            for tri in [[0usize, 1, 2], [0, 2, 3]] {
                m.tri_vert.push([base + tri[0] as u32, base + tri[1] as u32, base + tri[2] as u32]);
                m.tri_uv.push([at(quad[tri[0]].0, quad[tri[0]].1).2, at(quad[tri[1]].0, quad[tri[1]].1).2, at(quad[tri[2]].0, quad[tri[2]].1).2]);
                m.tri_nrm.push([at(quad[tri[0]].0, quad[tri[0]].1).1, at(quad[tri[1]].0, quad[tri[1]].1).1, at(quad[tri[2]].0, quad[tri[2]].1).1]);
            }
        }
    }
    m
}

#[test]
fn curvature_direction_wraps_a_tube_and_is_undecided_on_a_sphere() {
    let (dir, w) = shape_flow(&round_mesh(false));
    for t in 0..dir.len() {
        assert!(dir[t][2].abs() < 0.1, "tube triangle {t}: direction {:?} runs along the axis", dir[t]);
        assert!(w[t] > 0.7, "tube triangle {t}: weight {}", w[t]);
    }
    let (_, w) = shape_flow(&round_mesh(true));
    let mean = w.iter().sum::<f32>() / w.len() as f32;
    assert!(mean < 0.15, "sphere weight {mean}");
}

#[test]
fn shape_follow_turns_strokes_around_the_tube() {
    let mesh = round_mesh(false);
    let map = build_map(&mesh, W, W, 16);
    // flow along the tube axis (pixel-space +y = v = world z)
    let make = || Flow { angle: vec![90.0; W * W], coherence: vec![1.0; W * W], w: W, h: W };
    let along = |flow: &Flow| -> f32 {
        let (mut s, mut n) = (0f32, 0f32);
        for i in 0..W * W {
            if map.tri[i] == NONE {
                continue;
            }
            let t = &map.tris[map.tri[i] as usize];
            let a = flow.angle[i].to_radians();
            let d = [t.px[0] * a.cos() + t.py[0] * a.sin(), t.px[1] * a.cos() + t.py[1] * a.sin(), t.px[2] * a.cos() + t.py[2] * a.sin()];
            s += d[2].abs() / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            n += 1.0;
        }
        s / n
    };
    let mut off = make();
    steer_flow(&map, &mut off, 0.0);
    let mut on = make();
    steer_flow(&map, &mut on, 1.0);
    let (a_off, a_on) = (along(&off), along(&on));
    assert!(a_off > 0.99, "unsteered flow should stay along the axis: {a_off}");
    assert!(a_on < 0.15, "full follow should wrap around the tube: {a_on}");
}

//! surface.rs — UV-seam-aware painting. Strokes walk and are rasterized on the
//! mesh surface (world space) instead of the flat texture, so they continue
//! across UV seams. A `SurfMap` gives every texel its triangle; texels just
//! outside an island (the gutter) point at the texel that lies across the
//! seam, which is how walks and flood fills cross from one island to another.

use crate::pcg64::Pcg64;
use crate::pipeline::{self, Coverage, Flow, Grad, StrokeProfile};

/// "no triangle / no texel / no seam".
pub const NONE: u32 = u32::MAX;

type V3 = [f32; 3];

#[inline]
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn mul(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
/// Unit vector; a zero vector is returned unchanged.
#[inline]
fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        mul(a, 1.0 / l)
    } else {
        a
    }
}

// ---------------------------------------------------------------- mesh

/// Triangle soup sent by the Blender add-on (world space, Blender UVs: v up).
pub struct Mesh {
    pub vpos: Vec<V3>,
    pub tri_vert: Vec<[u32; 3]>,
    pub tri_uv: Vec<[[f32; 2]; 3]>,
    pub tri_nrm: Vec<[V3; 3]>,
}

impl Mesh {
    /// Parse the add-on blob (little-endian): "ISOM", u32 version = 1, u32
    /// ntri, u32 nvert, 9 f32 object matrix (unused here), nvert*3 f32
    /// positions, ntri*3 u32 vertex indices, ntri*6 f32 corner UVs, ntri*9 f32
    /// corner normals.
    pub fn parse(b: &[u8]) -> Result<Mesh, String> {
        if b.len() < 52 || &b[0..4] != b"ISOM" {
            return Err("not a mesh blob".into());
        }
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if u(4) != 1 {
            return Err(format!("unsupported mesh version {}", u(4)));
        }
        let (nt, nv) = (u(8) as usize, u(12) as usize);
        if nt == 0 {
            return Err("mesh has no triangles".into());
        }
        let need = nv
            .checked_mul(12)
            .and_then(|a| nt.checked_mul(72).and_then(|c| a.checked_add(c)))
            .and_then(|a| a.checked_add(52));
        if need != Some(b.len()) {
            return Err(format!("mesh blob is {} bytes, header says {:?}", b.len(), need));
        }
        let mut o = 52;
        let mut vpos = Vec::with_capacity(nv);
        for _ in 0..nv {
            vpos.push([f(o), f(o + 4), f(o + 8)]);
            o += 12;
        }
        let mut tri_vert = Vec::with_capacity(nt);
        for _ in 0..nt {
            let t = [u(o), u(o + 4), u(o + 8)];
            if t.iter().any(|&v| v as usize >= nv) {
                return Err("mesh vertex index out of range".into());
            }
            tri_vert.push(t);
            o += 12;
        }
        let mut tri_uv = Vec::with_capacity(nt);
        for _ in 0..nt {
            tri_uv.push([[f(o), f(o + 4)], [f(o + 8), f(o + 12)], [f(o + 16), f(o + 20)]]);
            o += 24;
        }
        let mut tri_nrm = Vec::with_capacity(nt);
        for _ in 0..nt {
            let mut n = [[0f32; 3]; 3];
            for c in n.iter_mut() {
                *c = [f(o), f(o + 4), f(o + 8)];
                o += 12;
            }
            tri_nrm.push(n);
        }
        Ok(Mesh { vpos, tri_vert, tri_uv, tri_nrm })
    }
}

/// Per triangle: the unit tangent along which the surface curves most (the
/// way a brush wraps around a form: around a cylinder, not along it) and how
/// clearly one direction dominates, 0..1 (0 on planes, spheres and saddles).
/// The shape operator of every triangle is fitted to its corner normals
/// (`dn = S e` over the three edges, least squares), then smoothed twice over
/// the vertex rings, area weighted, with vertices welded by position.
pub fn shape_flow(mesh: &Mesh) -> (Vec<V3>, Vec<f32>) {
    use std::collections::HashMap;
    type Sym = [f32; 6]; // xx yy zz xy xz yz
    let nt = mesh.tri_vert.len();
    let corners = |t: usize| -> [V3; 3] {
        let v = mesh.tri_vert[t];
        [mesh.vpos[v[0] as usize], mesh.vpos[v[1] as usize], mesh.vpos[v[2] as usize]]
    };
    // tangent frame of a triangle: u along the first edge, v = n x u, area
    let frame = |p: &[V3; 3]| -> (V3, V3, f32) {
        let c = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let area = 0.5 * dot(c, c).sqrt();
        let u = unit(sub(p[1], p[0]));
        (u, cross(unit(c), u), area)
    };
    let outer = |a: V3, b: V3, s: f32, t: &mut Sym| {
        t[0] += s * a[0] * b[0];
        t[1] += s * a[1] * b[1];
        t[2] += s * a[2] * b[2];
        t[3] += s * 0.5 * (a[0] * b[1] + a[1] * b[0]);
        t[4] += s * 0.5 * (a[0] * b[2] + a[2] * b[0]);
        t[5] += s * 0.5 * (a[1] * b[2] + a[2] * b[1]);
    };
    let apply = |t: &Sym, v: V3| -> V3 {
        [
            t[0] * v[0] + t[3] * v[1] + t[4] * v[2],
            t[3] * v[0] + t[1] * v[1] + t[5] * v[2],
            t[4] * v[0] + t[5] * v[1] + t[2] * v[2],
        ]
    };
    let mut tens: Vec<Sym> = vec![[0.0; 6]; nt];
    let mut area = vec![0f32; nt];
    for t in 0..nt {
        let p = corners(t);
        let (u, v, a) = frame(&p);
        area[t] = a;
        if a <= 0.0 {
            continue;
        }
        // normal equations of E a + F b = c, F a + G b = d over the 3 edges
        let (mut m, mut r) = ([[0f32; 3]; 3], [0f32; 3]);
        for i in 0..3 {
            let j = (i + 1) % 3;
            let e = sub(p[j], p[i]);
            let dn = sub(mesh.tri_nrm[t][j], mesh.tri_nrm[t][i]);
            let (a, b, c, d) = (dot(e, u), dot(e, v), dot(dn, u), dot(dn, v));
            m[0][0] += a * a;
            m[0][1] += a * b;
            m[1][0] += a * b;
            m[1][1] += a * a + b * b;
            m[1][2] += a * b;
            m[2][1] += a * b;
            m[2][2] += b * b;
            r[0] += a * c;
            r[1] += b * c + a * d;
            r[2] += b * d;
        }
        let det = |m: &[[f32; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d0 = det(&m);
        if d0.abs() < 1e-30 {
            continue;
        }
        let mut x = [0f32; 3];
        for (k, xk) in x.iter_mut().enumerate() {
            let mut mk = m;
            for row in 0..3 {
                mk[row][k] = r[row];
            }
            *xk = det(&mk) / d0;
        }
        let mut s: Sym = [0.0; 6];
        outer(u, u, x[0], &mut s);
        outer(u, v, x[1], &mut s);
        outer(v, u, x[1], &mut s);
        outer(v, v, x[2], &mut s);
        tens[t] = s;
    }
    // weld by position, then smooth: vertex = area-weighted mean of its triangles,
    // triangle = mean of its vertices
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.vpos {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let radius = 0.5 * dot(sub(hi, lo), sub(hi, lo)).sqrt().max(1e-12);
    let q = 1.0 / (radius * 1e-5);
    let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
    let vid: Vec<[usize; 3]> = (0..nt)
        .map(|t| {
            let p = corners(t);
            let mut o = [0usize; 3];
            for c in 0..3 {
                let k = [(p[c][0] * q).round() as i64, (p[c][1] * q).round() as i64, (p[c][2] * q).round() as i64];
                let n = ids.len();
                o[c] = *ids.entry(k).or_insert(n);
            }
            o
        })
        .collect();
    for _ in 0..2 {
        let mut vt: Vec<Sym> = vec![[0.0; 6]; ids.len()];
        let mut va = vec![0f32; ids.len()];
        for t in 0..nt {
            for &v in &vid[t] {
                for k in 0..6 {
                    vt[v][k] += area[t] * tens[t][k];
                }
                va[v] += area[t];
            }
        }
        for t in 0..nt {
            let mut s: Sym = [0.0; 6];
            for &v in &vid[t] {
                for k in 0..6 {
                    s[k] += vt[v][k] / va[v].max(1e-20) / 3.0;
                }
            }
            tens[t] = s;
        }
    }
    // principal curvatures in the triangle's own tangent plane
    let eps = 0.1 / radius;
    let mut dir = vec![[0f32; 3]; nt];
    let mut wgt = vec![0f32; nt];
    for t in 0..nt {
        let (u, v, a) = frame(&corners(t));
        if a <= 0.0 {
            dir[t] = u;
            continue;
        }
        let (su, sv) = (apply(&tens[t], u), apply(&tens[t], v));
        let (e, f, g) = (dot(u, su), dot(u, sv), dot(v, sv));
        let (mean, rad) = (0.5 * (e + g), (0.25 * (e - g) * (e - g) + f * f).sqrt());
        let (kmax, kmin) = (mean + rad, mean - rad);
        let mut th = 0.5 * (2.0 * f).atan2(e - g); // direction of kmax
        if kmin.abs() > kmax.abs() {
            th += std::f32::consts::FRAC_PI_2; // wrap around the form: the stronger |k|
        }
        dir[t] = unit(add(mul(u, th.cos()), mul(v, th.sin())));
        let (big, small) = (kmax.abs().max(kmin.abs()), kmax.abs().min(kmin.abs()));
        wgt[t] = ((big - small) / (big + small + eps)).clamp(0.0, 1.0);
    }
    (dir, wgt)
}

// ---------------------------------------------------------------- texel map

/// One triangle as affine functions of the pixel coordinates (x right, y down).
#[derive(Clone, Copy, Default)]
pub struct TriFrame {
    /// world position = o + px * x + py * y
    pub o: V3,
    pub px: V3,
    pub py: V3,
    /// unit face normal (mesh winding, so it points outward)
    pub nf: V3,
    /// a world vector d moves (r0 . d, r1 . d) pixels
    pub r0: V3,
    pub r1: V3,
    /// shading normal (not normalized) = n0 + nx * x + ny * y
    pub n0: V3,
    pub nx: V3,
    pub ny: V3,
}

pub struct SurfMap {
    pub w: usize,
    pub h: usize,
    /// triangle under the texel centre; NONE = the texel is not on an island
    pub tri: Vec<u32>,
    /// island texel this texel shows: itself on an island, the texel across
    /// the seam (or just inside the border) in the gutter; NONE = no surface
    pub twin: Vec<u32>,
    /// gutter texels reached through a seam: index into `edges`; NONE otherwise
    pub gedge: Vec<u32>,
    pub tris: Vec<TriFrame>,
    /// seam transforms, pixel -> pixel on the other side of the seam:
    /// (m[0]*x + m[1]*y + m[2], m[3]*x + m[4]*y + m[5])
    pub edges: Vec<[f32; 6]>,
    /// world units per texel (area average)
    pub upp: f32,
    /// number of island texels (0 = nothing to paint on)
    pub islands: usize,
    /// per triangle: unit tangent of the strongest curvature (see `shape_flow`)
    pub geo_dir: Vec<V3>,
    /// per triangle: how clearly that direction dominates, 0..1
    pub geo_w: Vec<f32>,
}

impl SurfMap {
    /// World position of pixel coordinates (x, y), using the triangle of texel `i`.
    #[inline]
    pub fn pos(&self, i: usize, x: f32, y: f32) -> V3 {
        let t = &self.tris[self.tri[i] as usize];
        add(add(t.o, mul(t.px, x)), mul(t.py, y))
    }

    /// Texel index of pixel coordinates, None outside the image.
    #[inline]
    pub fn texel(&self, x: f32, y: f32) -> Option<usize> {
        if x >= 0.0 && y >= 0.0 && x < self.w as f32 && y < self.h as f32 {
            Some(y as usize * self.w + x as usize)
        } else {
            None
        }
    }

    /// Apply seam transform `e` to pixel coordinates.
    #[inline]
    pub fn across(&self, e: u32, x: f32, y: f32) -> (f32, f32) {
        let m = &self.edges[e as usize];
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }
}

/// A boundary half-edge of an island, in pixel coordinates.
struct Border {
    a: [f32; 2],
    /// unit direction a -> b
    xh: [f32; 2],
    /// unit normal pointing out of the island
    yh: [f32; 2],
    len: f32,
    /// index into `SurfMap::edges`, NONE for an open (unpaired) edge
    seam: u32,
}

/// Rasterize the mesh's UV triangles at `w` x `h` and link a `margin`-pixel
/// gutter around every island to the texels across its seams.
pub fn build_map(mesh: &Mesh, w: usize, h: usize, margin: usize) -> SurfMap {
    let n = w * h;
    let nt = mesh.tri_vert.len();
    let mut tri = vec![NONE; n];
    let mut twin = vec![NONE; n];
    let mut gedge = vec![NONE; n];
    let mut tris = vec![TriFrame::default(); nt];
    let mut ok = vec![false; nt];
    // pixel corners of every triangle (image rows are top-down, v is up)
    let quv = |t: usize, c: usize| -> [f32; 2] {
        let uv = mesh.tri_uv[t][c];
        [uv[0] * w as f32, (1.0 - uv[1]) * h as f32]
    };
    let (mut area3, mut area2s) = (0f64, 0f64);
    let mut islands = 0usize;
    for t in 0..nt {
        let q = [quv(t, 0), quv(t, 1), quv(t, 2)];
        let v = mesh.tri_vert[t];
        let p = [mesh.vpos[v[0] as usize], mesh.vpos[v[1] as usize], mesh.vpos[v[2] as usize]];
        let (e1x, e1y) = (q[1][0] - q[0][0], q[1][1] - q[0][1]);
        let (e2x, e2y) = (q[2][0] - q[0][0], q[2][1] - q[0][1]);
        let area2 = e1x * e2y - e2x * e1y;
        let (d1, d2) = (sub(p[1], p[0]), sub(p[2], p[0]));
        let cr = cross(d1, d2);
        let crl = dot(cr, cr).sqrt();
        if !area2.is_finite() || area2.abs() < 1e-6 || !crl.is_finite() || crl <= 0.0 {
            continue;
        }
        let inv = 1.0 / area2;
        let px = mul(sub(mul(d1, e2y), mul(d2, e1y)), inv);
        let py = mul(sub(mul(d2, e1x), mul(d1, e2x)), inv);
        let (g00, g01, g11) = (dot(px, px), dot(px, py), dot(py, py));
        let det = g00 * g11 - g01 * g01;
        if !(det > 0.0) {
            continue;
        }
        let nr = mesh.tri_nrm[t];
        let (m1, m2) = (sub(nr[1], nr[0]), sub(nr[2], nr[0]));
        let nx = mul(sub(mul(m1, e2y), mul(m2, e1y)), inv);
        let ny = mul(sub(mul(m2, e1x), mul(m1, e2x)), inv);
        tris[t] = TriFrame {
            o: sub(sub(p[0], mul(px, q[0][0])), mul(py, q[0][1])),
            px,
            py,
            nf: mul(cr, 1.0 / crl),
            r0: mul(sub(mul(px, g11), mul(py, g01)), 1.0 / det),
            r1: mul(sub(mul(py, g00), mul(px, g01)), 1.0 / det),
            n0: sub(sub(nr[0], mul(nx, q[0][0])), mul(ny, q[0][1])),
            nx,
            ny,
        };
        ok[t] = true;
        area3 += crl as f64;
        area2s += area2.abs() as f64;
        // texel centres inside the triangle
        let s = if area2 > 0.0 { 1.0f32 } else { -1.0 };
        let minx = q[0][0].min(q[1][0]).min(q[2][0]);
        let maxx = q[0][0].max(q[1][0]).max(q[2][0]);
        let miny = q[0][1].min(q[1][1]).min(q[2][1]);
        let maxy = q[0][1].max(q[1][1]).max(q[2][1]);
        let x0 = (minx - 0.5).floor().max(0.0) as i64;
        let x1 = (maxx.ceil() as i64).min(w as i64 - 1);
        let y0 = (miny - 0.5).floor().max(0.0) as i64;
        let y1 = (maxy.ceil() as i64).min(h as i64 - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((q[1][0] - q[0][0]) * (cy - q[0][1]) - (q[1][1] - q[0][1]) * (cx - q[0][0])) * s;
                let w1 = ((q[2][0] - q[1][0]) * (cy - q[1][1]) - (q[2][1] - q[1][1]) * (cx - q[1][0])) * s;
                let w2 = ((q[0][0] - q[2][0]) * (cy - q[2][1]) - (q[0][1] - q[2][1]) * (cx - q[2][0])) * s;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let i = y as usize * w + x as usize;
                    if tri[i] == NONE {
                        islands += 1;
                    }
                    tri[i] = t as u32;
                    twin[i] = i as u32;
                }
            }
        }
    }
    let upp = if area2s > 0.0 { (area3 / area2s).sqrt() as f32 } else { 0.0 };
    let (geo_dir, geo_w) = shape_flow(mesh);
    let mut map = SurfMap { w, h, tri, twin, gedge: Vec::new(), tris, edges: Vec::new(), upp, islands, geo_dir, geo_w };
    if islands == 0 {
        map.gedge = gedge;
        return map;
    }

    // weld: vertices with identical positions share one id (meshes split
    // along sharp edges still pair their seams)
    let nv = mesh.vpos.len();
    let key = |i: u32| {
        let p = mesh.vpos[i as usize];
        ((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits(), (p[2] + 0.0).to_bits())
    };
    let mut order: Vec<u32> = (0..nv as u32).collect();
    order.sort_unstable_by_key(|&i| (key(i), i));
    let mut canon = vec![0u32; nv];
    let mut k = 0;
    while k < nv {
        let first = order[k];
        let kk = key(first);
        let mut j = k;
        while j < nv && key(order[j]) == kk {
            canon[order[j] as usize] = first;
            j += 1;
        }
        k = j;
    }

    // half-edges grouped by their (welded) vertex pair
    let mut hes: Vec<(u32, u32, u32, u8)> = Vec::new();
    for t in 0..nt {
        if !ok[t] {
            continue;
        }
        for e in 0..3usize {
            let ca = canon[mesh.tri_vert[t][e] as usize];
            let cb = canon[mesh.tri_vert[t][(e + 1) % 3] as usize];
            if ca != cb {
                hes.push((ca.min(cb), ca.max(cb), t as u32, e as u8));
            }
        }
    }
    hes.sort_unstable();
    // geometry of half-edge (t, e): a, b, third corner c, welded id at a
    let geo = |t: u32, e: u8| {
        let (t, e) = (t as usize, e as usize);
        (quv(t, e), quv(t, (e + 1) % 3), quv(t, (e + 2) % 3), canon[mesh.tri_vert[t][e] as usize])
    };
    // (a, xh, yh pointing away from c, len) or None for a zero-length edge
    let frame = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            return None;
        }
        let xh = [dx / len, dy / len];
        let mut yh = [-xh[1], xh[0]];
        if (c[0] - a[0]) * yh[0] + (c[1] - a[1]) * yh[1] > 0.0 {
            yh = [-yh[0], -yh[1]];
        }
        Some((xh, yh, len))
    };
    let mut borders: Vec<Border> = Vec::new();
    let mut edges: Vec<[f32; 6]> = Vec::new();
    let mut g = 0;
    while g < hes.len() {
        let mut g2 = g;
        while g2 < hes.len() && (hes[g2].0, hes[g2].1) == (hes[g].0, hes[g].1) {
            g2 += 1;
        }
        let group = &hes[g..g2];
        g = g2;
        // pair = exactly two half-edges on this vertex pair
        let pair = if group.len() == 2 {
            let (a1, b1, c1, ca1) = geo(group[0].2, group[0].3);
            let (a2, b2, c2, ca2) = geo(group[1].2, group[1].3);
            // endpoints of edge 2 matched to a1 / b1 by welded id
            let (ma, mb) = if ca2 == ca1 { (a2, b2) } else { (b2, a2) };
            let close = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-3 && (p[1] - q[1]).abs() < 1e-3;
            if close(a1, ma) && close(b1, mb) {
                continue; // same place in UV space: an interior edge
            }
            Some(((a1, b1, c1), (a2, b2, c2), (ma, mb)))
        } else {
            None
        };
        match pair {
            Some(((a1, b1, c1), (a2, b2, c2), (ma, mb))) => {
                // each side maps its outside onto the other side's inside
                let sides = [((a1, b1, c1), (ma, mb, c2)), ((a2, b2, c2), if ma == a2 { (a1, b1, c1) } else { (b1, a1, c1) })];
                for ((a, b, c), (ta, tb, tc)) in sides {
                    let (Some((xh, yh, len)), Some((txh, tyh, tlen))) = (frame(a, b, c), frame(ta, tb, tc)) else {
                        continue;
                    };
                    // on the far side "inward" is the opposite of its outward normal
                    let tin = [-tyh[0], -tyh[1]];
                    let k = tlen / len;
                    let m0 = k * (txh[0] * xh[0] + tin[0] * yh[0]);
                    let m1 = k * (txh[0] * xh[1] + tin[0] * yh[1]);
                    let m3 = k * (txh[1] * xh[0] + tin[1] * yh[0]);
                    let m4 = k * (txh[1] * xh[1] + tin[1] * yh[1]);
                    edges.push([m0, m1, ta[0] - m0 * a[0] - m1 * a[1], m3, m4, ta[1] - m3 * a[0] - m4 * a[1]]);
                    borders.push(Border { a, xh, yh, len, seam: edges.len() as u32 - 1 });
                }
            }
            None => {
                for &(_, _, t, e) in group {
                    let (a, b, c, _) = geo(t, e);
                    if let Some((xh, yh, len)) = frame(a, b, c) {
                        borders.push(Border { a, xh, yh, len, seam: NONE });
                    }
                }
            }
        }
    }

    // gutter: every off-island texel within `margin` px of a border takes the
    // nearest border
    let m = margin as f32;
    let mut dist = vec![f32::INFINITY; n];
    let mut cand = vec![NONE; n];
    for (bi, bd) in borders.iter().enumerate() {
        let b = [bd.a[0] + bd.xh[0] * bd.len, bd.a[1] + bd.xh[1] * bd.len];
        let x0 = (bd.a[0].min(b[0]) - m).floor().max(0.0) as i64;
        let x1 = ((bd.a[0].max(b[0]) + m).ceil() as i64).min(w as i64 - 1);
        let y0 = (bd.a[1].min(b[1]) - m).floor().max(0.0) as i64;
        let y1 = ((bd.a[1].max(b[1]) + m).ceil() as i64).min(h as i64 - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y as usize * w + x as usize;
                if map.tri[i] != NONE {
                    continue;
                }
                let (rx, ry) = (x as f32 + 0.5 - bd.a[0], y as f32 + 0.5 - bd.a[1]);
                let s = (rx * bd.xh[0] + ry * bd.xh[1]).clamp(0.0, bd.len);
                let (qx, qy) = (rx - bd.xh[0] * s, ry - bd.xh[1] * s);
                let dd = (qx * qx + qy * qy).sqrt();
                if dd <= m && dd < dist[i] {
                    dist[i] = dd;
                    cand[i] = bi as u32;
                }
            }
        }
    }
    map.edges = edges;
    for i in 0..n {
        if cand[i] == NONE {
            continue;
        }
        let bd = &borders[cand[i] as usize];
        let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
        if bd.seam != NONE {
            let (qx, qy) = map.across(bd.seam, x, y);
            if let Some(j) = map.texel(qx, qy).filter(|&j| map.tri[j] != NONE) {
                map.twin[i] = j as u32;
                gedge[i] = bd.seam;
                continue;
            }
        }
        // open edge (or the seam lookup missed the island): repeat the texel
        // just inside the border
        let (rx, ry) = (x - bd.a[0], y - bd.a[1]);
        let s = (rx * bd.xh[0] + ry * bd.xh[1]).clamp(0.0, bd.len);
        for inset in [0.75f32, 1.5] {
            let qx = bd.a[0] + bd.xh[0] * s - bd.yh[0] * inset;
            let qy = bd.a[1] + bd.xh[1] * s - bd.yh[1] * inset;
            if let Some(j) = map.texel(qx, qy).filter(|&j| map.tri[j] != NONE) {
                map.twin[i] = j as u32;
                break;
            }
        }
    }
    map.gedge = gedge;
    map
}

/// Bend the flow toward the mesh's curvature direction: per island texel the
/// flow (as a world vector) is blended with the triangle's `geo_dir` by
/// `follow * geo_w`, so strokes wrap around the form where it is clearly
/// curved one way and keep following the image elsewhere. Flow angles are
/// pixel-space, as `trace` reads them.
pub fn steer_flow(map: &SurfMap, flow: &mut Flow, follow: f32) {
    if follow <= 0.0 {
        return;
    }
    for i in 0..map.w * map.h {
        let ti = map.tri[i];
        if ti == NONE {
            continue;
        }
        let ti = ti as usize;
        let m = (follow * map.geo_w[ti]).clamp(0.0, 1.0);
        if m <= 0.0 {
            continue;
        }
        let t = &map.tris[ti];
        let a = flow.angle[i].to_radians();
        let f = unit(add(mul(t.px, a.cos()), mul(t.py, a.sin())));
        let mut g = map.geo_dir[ti];
        if dot(f, g) < 0.0 {
            g = mul(g, -1.0);
        }
        let b = unit(add(mul(f, 1.0 - m), mul(g, m)));
        flow.angle[i] = dot(t.r1, b).atan2(dot(t.r0, b)).to_degrees().rem_euclid(180.0);
    }
}

/// Copy every gutter texel's `c` values from its twin.
pub fn fill_gutter<T: Copy>(map: &SurfMap, buf: &mut [T], c: usize) {
    for i in 0..map.w * map.h {
        let t = map.twin[i];
        if map.tri[i] == NONE && t != NONE {
            let t = t as usize;
            for k in 0..c {
                buf[i * c + k] = buf[t * c + k];
            }
        }
    }
}

// ---------------------------------------------------------------- strokes

/// A stroke as a polyline on the surface.
pub struct SurfStroke {
    /// world-space nodes
    pub pts: Vec<V3>,
    /// face normal at each node
    pub nrm: Vec<V3>,
    /// island texel under each node
    pub tex: Vec<u32>,
    /// seed position in pixels (boil re-traces from here)
    pub start: (f32, f32),
    pub th: i32,
    pub pr: i32,
    pub lum: f32,
}

/// Walk a stroke from pixel (x0, y0) along the flow field, on the surface.
/// Steps have a constant world length; stepping into the gutter continues on
/// the island across the seam. None when the seed is not on an island.
pub fn trace(x0: f32, y0: f32, flow: &Flow, map: &SurfMap, length: i32, look: f64) -> Option<(Vec<V3>, Vec<V3>, Vec<u32>)> {
    let (w, h) = (map.w, map.h);
    let mut cx = x0.clamp(0.0, w as f32 - 0.001);
    let mut cy = y0.clamp(0.0, h as f32 - 0.001);
    let mut i = cy as usize * w + cx as usize;
    if map.tri[i] == NONE {
        return None;
    }
    let step3 = (2.0 * look) as f32 * map.upp;
    let steps = ((length as f64 / (2.0 * look)) as i64).max(2) as usize;
    // flow direction of texel i as a world-space unit vector
    let dir = |i: usize| -> V3 {
        let t = &map.tris[map.tri[i] as usize];
        let a = flow.angle[i].to_radians();
        unit(add(mul(t.px, a.cos()), mul(t.py, a.sin())))
    };
    let mut d = dir(i);
    let mut pts = vec![map.pos(i, cx, cy)];
    let mut nrm = vec![map.tris[map.tri[i] as usize].nf];
    let mut tex = vec![i as u32];
    for _ in 0..steps {
        let t = &map.tris[map.tri[i] as usize];
        let mut f = dir(i);
        if dot(f, d) < 0.0 {
            f = mul(f, -1.0);
        }
        d = add(mul(d, 0.7), mul(f, 0.3));
        d = sub(d, mul(t.nf, dot(d, t.nf))); // stay in the surface
        d = mul(d, 1.0 / (dot(d, d).sqrt() + 1e-8));
        cx += step3 * dot(t.r0, d);
        cy += step3 * dot(t.r1, d);
        let Some(mut j) = map.texel(cx, cy) else { break };
        if map.tri[j] == NONE {
            let e = map.gedge[j];
            if e == NONE {
                break; // open border or no surface: the stroke ends here
            }
            let (qx, qy) = map.across(e, cx, cy);
            match map.texel(qx, qy).filter(|&k| map.tri[k] != NONE) {
                Some(k) => {
                    cx = qx;
                    cy = qy;
                    j = k;
                }
                None => {
                    j = map.twin[j] as usize;
                    cx = (j % w) as f32 + 0.5;
                    cy = (j / w) as f32 + 0.5;
                }
            }
        }
        i = j;
        pts.push(map.pos(i, cx, cy));
        nrm.push(map.tris[map.tri[i] as usize].nf);
        tex.push(i as u32);
    }
    Some((pts, nrm, tex))
}

/// `pipeline::generate_strokes` on the surface: same grid, jitter, gradient
/// order and RNG draws; seeds off the islands are skipped.
#[allow(clippy::too_many_arguments)]
pub fn generate_strokes(
    bgr: &[u8],
    flow: &Flow,
    map: &SurfMap,
    strokes_per_layer: i32,
    length: i32,
    width: i32,
    layers: i32,
    jitter: f64,
    seed: u64,
    look: f64,
) -> Vec<SurfStroke> {
    let (w, h) = (map.w, map.h);
    let mut rng = Pcg64::new(seed);
    let gray = pipeline::gray_from_bgr(bgr, w, h);
    let gx = pipeline::sobel3(&gray, w, h, 1, 0);
    let gy = pipeline::sobel3(&gray, w, h, 0, 1);
    let grad: Vec<f32> = (0..w * h).map(|i| (gx[i].powi(2) + gy[i].powi(2)).sqrt()).collect();
    let step = (h as i32 / (strokes_per_layer as f64).sqrt() as i32).max(2);
    let jit = step as f64 * 0.5 * jitter;
    let mut out: Vec<SurfStroke> = Vec::new();
    for _layer in 0..layers {
        let mut pts: Vec<(f32, f32)> = Vec::new();
        let mut yy = 0i32;
        while yy < h as i32 {
            let mut xx = 0i32;
            while xx < w as i32 {
                pts.push(((xx as f64 + rng.uniform(-jit, jit)) as f32, (yy as f64 + rng.uniform(-jit, jit)) as f32));
                xx += step;
            }
            yy += step;
        }
        let gi = |p: &(f32, f32)| -> usize {
            let xi = (p.0 as i64).clamp(0, w as i64 - 1) as usize;
            let yi = (p.1 as i64).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };
        pts.sort_by(|a, b| grad[gi(b)].partial_cmp(&grad[gi(a)]).unwrap_or(std::cmp::Ordering::Equal));
        for &(x0, y0) in &pts {
            let Some((poly, nrm, tex)) = trace(x0, y0, flow, map, length, look) else { continue };
            if poly.len() < 3 {
                continue;
            }
            let lum = rng.uniform(0.94, 1.06) as f32;
            let _lj = length as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter);
            let wj = ((width as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter)) as i64).max(1) as i32;
            out.push(SurfStroke { pts: poly, nrm, tex, start: (x0, y0), th: wj.max(1), pr: (width / 2).max(1), lum });
        }
    }
    out
}

/// `pipeline::boil_strokes` on the surface.
#[allow(clippy::too_many_arguments)]
pub fn boil_strokes(
    strokes: &mut [SurfStroke],
    flow: &Flow,
    map: &SurfMap,
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
    let step = (map.h as i32 / (strokes_per_layer as f64).sqrt() as i32).max(2);
    let off = step as f64 * 0.5;
    for s in strokes.iter_mut() {
        if rng.uniform(0.0, 1.0) >= boil {
            continue;
        }
        let dx = rng.uniform(-off, off);
        let dy = rng.uniform(-off, off);
        let lum = rng.uniform(0.94, 1.06) as f32;
        let wj = ((width as f64 * rng.uniform(1.0 - jitter, 1.0 + jitter)) as i64).max(1) as i32;
        let (mut x, mut y) = ((s.start.0 as f64 + dx) as f32, (s.start.1 as f64 + dy) as f32);
        // a start that slid into the gutter continues across the seam
        if let Some(j) = map.texel(x, y) {
            if map.tri[j] == NONE && map.gedge[j] != NONE {
                (x, y) = map.across(map.gedge[j], x, y);
            }
        }
        let Some((poly, nrm, tex)) = trace(x, y, flow, map, length, look) else { continue };
        if poly.len() < 3 {
            continue;
        }
        s.pts = poly;
        s.nrm = nrm;
        s.tex = tex;
        s.start = (x, y);
        s.lum = lum;
        s.th = wj;
    }
}

// ---------------------------------------------------------------- footprint

/// One texel under a stroke, in pixel units.
pub struct Hit {
    /// island texel
    pub i: u32,
    /// signed distance to the stroke axis
    pub sd: f32,
    /// arclength along the stroke of the nearest axis point
    pub a: f32,
    /// distance past the first / last node along the axis (0 elsewhere)
    pub cap: f32,
    /// squared distance to the polyline
    pub d2: f32,
}

/// Reusable buffers for [`gather`].
pub struct Scratch {
    mark: Vec<u32>,
    stamp: u32,
    queue: Vec<(u32, u32)>,
    segs: Vec<(V3, V3, f32)>,
    cum: Vec<f32>,
}

impl Scratch {
    pub fn new(texels: usize) -> Self {
        Scratch { mark: vec![0; texels], stamp: 0, queue: Vec::new(), segs: Vec::new(), cum: Vec::new() }
    }
}

/// Collect the island texels within reach of stroke `s` into `hits` by flood
/// fill from the stroke's own texels; the fill crosses seams through gutter
/// twins. Distances are measured in world space. Returns the stroke length in
/// pixels.
pub fn gather(map: &SurfMap, s: &SurfStroke, scr: &mut Scratch, hits: &mut Vec<Hit>) -> f32 {
    hits.clear();
    let (w, h) = (map.w, map.h);
    let upp = map.upp;
    let inv = 1.0 / upp;
    let n = s.pts.len();
    if n < 2 {
        return 0.0;
    }
    let nseg = n - 1;
    scr.segs.clear();
    scr.cum.clear();
    scr.cum.push(0.0);
    for k in 1..n {
        let d = sub(s.pts[k], s.pts[k - 1]);
        let l = dot(d, d).sqrt();
        // (start, unit direction, length); length 0 marks a degenerate segment
        scr.segs.push(if l > 1e-12 { (s.pts[k - 1], mul(d, 1.0 / l), l) } else { (s.pts[k - 1], [0.0; 3], 0.0) });
        scr.cum.push(scr.cum[k - 1] + l * inv);
    }
    let total = scr.cum[nseg];
    let hw = pipeline::stroke_half(s.th).max(2.0) * upp;
    let hw2 = hw * hw;
    scr.stamp = scr.stamp.wrapping_add(1);
    if scr.stamp == 0 {
        scr.mark.iter_mut().for_each(|m| *m = 0);
        scr.stamp = 1;
    }
    let stamp = scr.stamp;
    scr.queue.clear();
    for k in 0..n {
        let i = s.tex[k];
        if scr.mark[i as usize] != stamp {
            scr.mark[i as usize] = stamp;
            scr.queue.push((i, k.min(nseg - 1) as u32));
        }
    }
    let mut head = 0;
    while head < scr.queue.len() {
        let (i, hint) = scr.queue[head];
        head += 1;
        let iu = i as usize;
        let (x, y) = (iu % w, iu / w);
        let p = map.pos(iu, x as f32 + 0.5, y as f32 + 0.5);
        // nearest segment near the one the neighbour was closest to
        let lo_s = (hint as usize).saturating_sub(2);
        let hi_s = (hint as usize + 2).min(nseg - 1);
        let (mut bd2, mut bseg, mut bproj, mut bt) = (f32::MAX, usize::MAX, 0f32, 0f32);
        for sg in lo_s..=hi_s {
            let (a, u, len) = scr.segs[sg];
            if len <= 0.0 {
                continue;
            }
            let r = sub(p, a);
            let proj = dot(r, u);
            let lo = if sg == 0 { -upp } else { 0.0 };
            let hi = if sg == nseg - 1 { len + upp } else { len };
            let t = proj.clamp(lo, hi);
            let q = sub(r, mul(u, t));
            let d2 = dot(q, q);
            if d2 < bd2 {
                (bd2, bseg, bproj, bt) = (d2, sg, proj, t);
            }
        }
        if bseg == usize::MAX || bd2 >= hw2 {
            continue;
        }
        // not onto the back of a thin part (still wraps 90 degree edges)
        if dot(map.tris[map.tri[iu] as usize].nf, s.nrm[bseg]) <= -0.5 {
            continue;
        }
        let (a, u, len) = scr.segs[bseg];
        let perp = sub(sub(p, a), mul(u, bproj));
        let pl = dot(perp, perp).sqrt();
        let side = dot(cross(s.nrm[bseg], u), perp);
        let past_end = (bseg == 0 && bproj < -upp) || (bseg == nseg - 1 && bproj > len + upp);
        hits.push(Hit {
            i,
            sd: if side < 0.0 { -pl * inv } else { pl * inv },
            a: scr.cum[bseg] + bt * inv,
            cap: if past_end { (bproj - bt).abs() * inv } else { 0.0 },
            d2: bd2 * inv * inv,
        });
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            let j2 = if map.tri[j] != NONE { j as u32 } else { map.twin[j] };
            if j2 != NONE && scr.mark[j2 as usize] != stamp {
                scr.mark[j2 as usize] = stamp;
                scr.queue.push((j2, bseg as u32));
            }
        }
    }
    total
}

/// `pipeline::rasterize_strokes` on the surface: flat stroke ends with a
/// radius-2 cap, round joints, later strokes on top.
pub fn rasterize(map: &SurfMap, strokes: &[SurfStroke], look: f64) -> Coverage {
    let (w, h) = (map.w, map.h);
    let mut owner = vec![0u32; w * h];
    let mut scr = Scratch::new(w * h);
    let mut hits: Vec<Hit> = Vec::new();
    for (si, s) in strokes.iter().enumerate() {
        gather(map, s, &mut scr, &mut hits);
        let half = if s.th <= 1 { 0.5 } else { ((s.th + 1) / 2) as f32 + 0.5 };
        for hit in &hits {
            let inside = if hit.cap > 0.0 { hit.sd * hit.sd + hit.cap * hit.cap <= 4.0 } else { hit.d2 <= half * half };
            if inside {
                owner[hit.i as usize] = si as u32 + 1;
            }
        }
    }
    fill_gutter(map, &mut owner, 1);
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
    let blurred = pipeline::gaussian_blur_plane(&mask, w, h, 0.6 * look);
    let alpha = blurred.iter().map(|&b| (b * 1.4).clamp(0.0, 1.0) * 0.9f32).collect();
    Coverage { owner, alpha, used }
}

/// `pipeline::impasto_height` on the surface (same stroke profile and
/// layering; pixel units).
pub fn impasto_height(map: &SurfMap, strokes: &[SurfStroke], brush: u32, look: f64) -> Vec<f32> {
    let (w, h) = (map.w, map.h);
    let lk = look as f32;
    let st = pipeline::impasto_style(brush);
    let mut height = vec![0f32; w * h];
    let mut scr = Scratch::new(w * h);
    let mut hits: Vec<Hit> = Vec::new();
    let mut paint: Vec<(u32, f32)> = Vec::new();
    for (idx, s) in strokes.iter().enumerate() {
        let total = gather(map, s, &mut scr, &mut hits);
        if total < 1e-3 {
            continue;
        }
        let half = pipeline::stroke_half(s.th);
        let seed = s.lum.to_bits() ^ (idx as u32).wrapping_mul(0x9E37_79B9);
        let prof = StrokeProfile::new(st, seed, half, total, lk);
        let mut sum = 0f64;
        paint.clear();
        for hit in &hits {
            if hit.sd.abs() >= half || hit.d2 >= half * half {
                continue;
            }
            let t = prof.eval(hit.sd, hit.a);
            if t > 0.0 {
                sum += height[hit.i as usize] as f64;
                paint.push((hit.i, t));
            }
        }
        if paint.is_empty() {
            continue;
        }
        let mean_under = (sum / paint.len() as f64) as f32;
        for &(i, t) in &paint {
            let i = i as usize;
            let c = pipeline::smooth01(t / (0.45 * prof.tmax));
            height[i] = height[i] * (1.0 - c) + st.keep * mean_under * c + t;
        }
    }
    fill_gutter(map, &mut height, 1);
    let mut out = pipeline::gaussian_blur_plane(&height, w, h, st.blur as f64 * look);
    fill_gutter(map, &mut out, 1);
    out
}

// ---------------------------------------------------------------- shading

/// `pipeline::impasto_shade_field` lit in world space, so the baked light
/// comes from the same side on both sides of a seam (the flat-texture version
/// lights every island from the upper-left of the image).
pub fn shade_field(map: &SurfMap, height: &[f32], grad: &Grad, impasto: f64, ref_half: f32, look: f64) -> Vec<[f32; 3]> {
    let (w, h) = (map.w, map.h);
    let lk = look as f32;
    let (gx, gy) = grad;
    let low = pipeline::gaussian_blur_plane(height, w, h, (0.28 * ref_half).clamp(1.0 * lk, 6.0 * lk) as f64);
    let g = pipeline::impasto_slope(impasto);
    // same elevation as the flat light (-0.5, -0.5, 0.7), normalized
    let (l_tan, l_up) = (0.710_67f32, 0.703_53f32);
    let toward = unit([-1.0, -1.0, 1.0]);
    (0..w * h)
        .map(|i| {
            if map.tri[i] == NONE {
                return [1.0, 0.0, 1.0];
            }
            let t = &map.tris[map.tri[i] as usize];
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let mut n = add(add(t.n0, mul(t.nx, x)), mul(t.ny, y));
            if !(dot(n, n) > 1e-12) {
                n = t.nf;
            }
            let n = unit(n);
            // light azimuth: a fixed world direction laid into the surface
            let mut p = sub(toward, mul(n, dot(toward, n)));
            for alt in [[0.0f32, 0.0, 1.0], [1.0, 0.0, 0.0]] {
                if dot(p, p) >= 1e-6 {
                    break;
                }
                p = sub(alt, mul(n, dot(alt, n)));
            }
            let l = add(mul(unit(p), l_tan), mul(n, l_up));
            let hv = unit(add(l, n));
            // height gradient as a world vector, then the tilted normal
            let gh = mul(add(mul(t.r0, gx[i]), mul(t.r1, gy[i])), map.upp);
            let n2 = unit(sub(n, mul(gh, g)));
            let diff = dot(n2, l).max(0.0) / l_up;
            let spec = (dot(n2, hv).max(0.0).powi(40) - dot(n, hv).max(0.0).powi(40)).max(0.0);
            let cav = (low[i] - height[i]).max(0.0) / ref_half;
            let dark = 1.0 - (0.7 * impasto as f32 * cav).min(0.6);
            [1.0 + 0.85 * (diff - 1.0), 255.0 * 0.22 * spec, dark]
        })
        .collect()
}

/// Height gradient rescaled per texel so a slope reads the same on islands
/// with different texel densities (tangent-space normal tilt).
pub fn scale_grad(map: &SurfMap, grad: &Grad) -> Grad {
    let mut gx = grad.0.clone();
    let mut gy = grad.1.clone();
    for i in 0..map.w * map.h {
        if map.tri[i] == NONE {
            continue;
        }
        let t = &map.tris[map.tri[i] as usize];
        gx[i] *= map.upp / dot(t.px, t.px).sqrt();
        gy[i] *= map.upp / dot(t.py, t.py).sqrt();
    }
    (gx, gy)
}

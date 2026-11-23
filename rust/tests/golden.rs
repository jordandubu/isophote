//! Golden hashes: byte identity of every impasto-0 path plus Kuwahara/flow/relief stages.
//! `GOLDEN_UPDATE=1 cargo test --test golden` rewrites tests/golden_hashes.txt.
use isophote_core::pipeline::*;

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv_f32(v: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    fnv(&bytes)
}

fn synth(w: usize, h: usize, salt: usize) -> Vec<u8> {
    let mut v = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            v[i] = ((x * 5 + y * y / 7 + salt * 31) % 256) as u8;
            v[i + 1] = (((x ^ y) * 3 + salt * 17) % 256) as u8;
            v[i + 2] = ((x * y / 16 + 90 * ((x / 23 + y / 17 + salt) % 2)) % 256) as u8;
            v[i + 3] = 255;
        }
    }
    v
}

fn to_bgr(bgra: &[u8]) -> Vec<u8> {
    bgra.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}

#[allow(clippy::too_many_arguments)]
fn params(len: i32, bw: i32, layers: i32, relief: f64, radius: i32, smooth: i32, seed: u64) -> Params {
    Params { brush_len: len, brush_w: bw, layers, relief, base_radius: radius, flow_smooth: smooth, seed,
    max_size: None, impasto: 0.0, impasto_knife: false, boil: 0.0, boil_frame: 0,
    look_scale: 0.0, out_size: 0, normal_tangent: false, uv_aware: false, shape_follow: 0.0 }
}

fn cases() -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = Vec::new();
    let mut single = |name: &str, w: usize, h: usize, p: &Params| {
        let img = synth(w, h, 0);
        out.push((name.to_string(), fnv(&paint_bgra(&img, w, h, p).0)));
    };
    single("single_art", 160, 120, &params(30, 8, 2, 0.1, 3, 4, 7));
    single("single_norelief", 160, 120, &params(30, 8, 2, 0.0, 3, 4, 7));
    single("single_look15", 160, 120, &Params { look_scale: 1.5, ..params(45, 12, 2, 0.2, 5, 6, 11) });
    single("single_look04", 160, 120, &Params { look_scale: 0.4, ..params(12, 3, 2, 0.1, 3, 4, 7) });
    single("single_w1", 96, 64, &params(10, 1, 3, 0.1, 2, 2, 5));
    single("single_bold", 200, 150, &params(63, 37, 2, 0.1, 11, 11, 42));
    single("single_tiny", 16, 12, &params(12, 3, 2, 0.1, 6, 8, 3));
    single("single_boil", 128, 96, &Params { boil: 0.3, boil_frame: 2, ..params(20, 3, 2, 0.1, 3, 4, 7) });

    let (w, h) = (96usize, 64usize);
    let (r0, c1, c2) = (synth(w, h, 0), synth(w, h, 1), synth(w, h, 2));
    let multi = paint_bgra_multi(&r0, &[&c1, &c2], w, h, &params(30, 8, 2, 0.1, 3, 4, 7), None).unwrap();
    for (i, f) in multi.iter().enumerate() {
        out.push((format!("multi3/{i}"), fnv(f)));
    }

    let bgr = |w: usize, h: usize| to_bgr(&synth(w, h, 0));
    out.push(("kuwahara_r3".into(), fnv(&kuwahara(&bgr(160, 120), 160, 120, 3))));
    out.push(("kuwahara_r11".into(), fnv(&kuwahara(&bgr(200, 150), 200, 150, 11))));
    out.push(("kuwahara_tiny".into(), fnv(&kuwahara(&bgr(16, 12), 16, 12, 6))));
    let b = bgr(160, 120);
    let flow = smooth_flow(flow_field(&gray_from_bgr(&b, 160, 120), 160, 120, 2.5), 8);
    out.push(("flow_angle".into(), fnv_f32(&flow.angle)));
    out.push(("flow_coh".into(), fnv_f32(&flow.coherence)));
    out.push(("relief_shade".into(), fnv_f32(&canvas_relief(135, 180, 12, 0.1, 1.5).0)));
    out
}

#[test]
fn golden_hashes() {
    let got = cases();
    let path = "tests/golden_hashes.txt";
    if std::env::var("GOLDEN_UPDATE").is_ok() {
        let s: String = got.iter().map(|(n, h)| format!("{n}\t{h:016x}\n")).collect();
        std::fs::write(path, s).unwrap();
        return;
    }
    let want = std::fs::read_to_string(path).unwrap();
    let want: std::collections::HashMap<&str, &str> = want.lines().filter_map(|l| l.split_once('\t')).collect();
    let mut bad = Vec::new();
    for (n, h) in &got {
        match want.get(n.as_str()) {
            Some(w) if *w == format!("{h:016x}") => {}
            Some(w) => bad.push(format!("{n}: got {h:016x} want {w}")),
            None => bad.push(format!("{n}: missing from golden file")),
        }
    }
    assert!(bad.is_empty(), "golden mismatches:\n{}", bad.join("\n"));
}

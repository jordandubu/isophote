//! Stroke loop verification vs python reference fixture.
use isophote_core::pipeline::*;

fn read_npy(path: &str) -> (Vec<usize>, Vec<u8>) {
    let buf = std::fs::read(path).unwrap();
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = String::from_utf8_lossy(&buf[10..10 + hlen]).to_string();
    let dims: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    (dims, buf[10 + hlen..].to_vec())
}

#[test]
#[allow(unused)]
fn strokes_match_python_reference() {
    let (dims, bgr) = read_npy("tests/in_bgr.npy");
    let (h, w) = (dims[0], dims[1]);
    let base = kuwahara(&bgr, w, h, 6);
    let gray = gray_from_bgr(&bgr, w, h);
    let flow0 = flow_field(&gray, w, h, 2.5);
    let flow = smooth_flow(flow0, 8);
    let n_strokes = (w * h) as i32 / 140;
    let strokes = generate_strokes(&bgr, w, h, &flow, n_strokes, 45, 3, 4, 0.6, 42);
    let out_bgra = paint_from_strokes(&bgr, &base, w, h, &strokes, 0.0);
    struct OutWrap { bgra: Vec<u8> }
    let out = OutWrap { bgra: out_bgra };
    let (_, ref_painted) = read_npy("tests/painted_ref.npy");
    let mut big = 0usize;
    let mut sum = 0i64;
    for (a, b) in out.bgra.chunks(4).zip(ref_painted.chunks(3)) {
        let d = ((a[0] as i32 - b[0] as i32).abs()
            + (a[1] as i32 - b[1] as i32).abs()
            + (a[2] as i32 - b[2] as i32).abs()) as i64;
        if d > 30 {
            big += 1;
        }
        sum += d as i64;
    }
    let total = (w * h) as i64;
    let mean_d3 = sum as f64 / total as f64;
    println!("strokes: mean |diff| (3ch sum) = {mean_d3:.1}, pixels >30: {big}/{}", total);
    // Per-pixel AA boundary jitter vs cv2's error-diffusion rasterizer keeps
    // isolated diffs around 15% (fitted residual), while the mean stays low.
    assert!(mean_d3 < 25.0, "rendered result too far from reference");
    assert!(big as i64 <= total / 5, "too many strongly-differing pixels");
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

#[test]
fn multi_replay_same_strokes() {
    use isophote_core::pipeline::*;
    // synthetic 48x32: reference = smooth gradient + shape, chan = same
    // geometry different palette + noise (like normal map vs base color)
    let (w, h) = (48usize, 32usize);
    let mut ref_img = vec![0u8; w * h * 4];
    let mut chan = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            ref_img[i] = (x * 5) as u8;      // B: gradient along x
            ref_img[i + 1] = (y * 7) as u8;  // G: gradient along y
            ref_img[i + 2] = 128;
            ref_img[i + 3] = 255;
            chan[i] = 200u8.saturating_sub((x * 3 % 201) as u8);
            chan[i + 1] = ((x * x / 4 + y * 3) % 256) as u8;
            chan[i + 2] = ((x + y * 11) % 256) as u8;
            chan[i + 3] = 255;
        }
    }
    let p = Params { brush_len: 12, brush_w: 3, layers: 2, relief: 0.1,
                     base_radius: 3, flow_smooth: 4, seed: 7 };
    let outs = paint_bgra_multi(&ref_img, &[&chan], w, h, &p).unwrap();
    assert_eq!(outs.len(), 2);
    // determinism: same inputs -> identical bytes
    let outs2 = paint_bgra_multi(&ref_img, &[&chan], w, h, &p).unwrap();
    assert_eq!(outs, outs2, "multi repaint is nondeterministic");

    // coherence: strokes stamped identically -> the painted channel's
    // deviation from its Kuwahara base should be at the SAME pixel set as
    // the reference's deviation (updated mask must coincide, not the colors)
    // cheap proxy: count of changed pixels (vs kuwahara base) differs <5%
    let n = w * h;
    let changed = |a: &Vec<u8>, base_bgr: &Vec<u8>| -> usize {
        let bgra = |base_bgr: &Vec<u8>, i: usize| -> [u8; 3] {
            [base_bgr[i * 3], base_bgr[i * 3 + 1], base_bgr[i * 3 + 2]]
        };
        (0..n).filter(|&i| {
            let q = [a[i * 4], a[i * 4 + 1], a[i * 4 + 2]];
            let b = bgra(base_bgr, i);
            (q[0].abs_diff(b[0]) as u32 + q[1].abs_diff(b[1]) as u32 + q[2].abs_diff(b[2]) as u32) > 12
        }).count()
    };
    let ref_bgr = {
        let mut b = vec![0u8; n * 3];
        for i in 0..n { b[i*3]=ref_img[i*4]; b[i*3+1]=ref_img[i*4+1]; b[i*3+2]=ref_img[i*4+2]; }
        b
    };
    let ch_bgr = {
        let mut b = vec![0u8; n * 3];
        for i in 0..n { b[i*3]=chan[i*4]; b[i*3+1]=chan[i*4+1]; b[i*3+2]=chan[i*4+2]; }
        b
    };
    let ref_base = kuwahara(&ref_bgr, w, h, p.base_radius);
    let ch_base = kuwahara(&ch_bgr, w, h, p.base_radius);
    let c_ref = changed(&outs[0], &ref_base);
    let c_ch = changed(&outs[1], &ch_base);
    println!("changed ref {} chan {} (of {})", c_ref, c_ch, n);
    let diff_ratio = (c_ref as f64 - c_ch as f64).abs() / (c_ref.max(c_ch) as f64).max(1.0);
    assert!(diff_ratio < 0.05, "stroke stamping not aligned: {} vs {}", c_ref, c_ch);

    // stroke geometry replay: identical geometry shared between the two
    // paint_from_strokes calls (generate_strokes is deterministic per seed)
    let strokes = {
        let ref_bgr2 = ref_bgr.clone();
        let gray = gray_from_bgr(&ref_bgr2, w, h);
        let flow = smooth_flow(flow_field(&gray, w, h, 2.5), p.flow_smooth);
        generate_strokes(&ref_bgr2, w, h, &flow, (w * h) as i32 / 140,
                         p.brush_len, p.brush_w, p.layers, 0.6, p.seed)
    };
    assert!(!strokes.is_empty());
    let painted_ref2 = paint_from_strokes(&ref_bgr, &ref_base, w, h, &strokes, p.relief);
    assert_eq!(painted_ref2, outs[0], "generate+paint path != paint_bgra_multi ref");

    // validation errors
    assert!(paint_bgra_multi(&ref_img, &[&chan], 0, 32, &p).is_err());
    assert!(paint_bgra_multi(&ref_img, &[&chan[..chan.len() - 4]], w, h, &p).is_err());
}

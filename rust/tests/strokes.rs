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
                     base_radius: 3, flow_smooth: 4, seed: 7,
                     max_size: None, impasto: 0.0, impasto_knife: false,
                     boil: 0.0, boil_frame: 0 };
    let outs = paint_bgra_multi(&ref_img, &[&chan], w, h, &p, None).unwrap();
    assert_eq!(outs.len(), 2);
    // determinism: same inputs -> identical bytes
    let outs2 = paint_bgra_multi(&ref_img, &[&chan], w, h, &p, None).unwrap();
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
    assert!(paint_bgra_multi(&ref_img, &[&chan], 0, 32, &p, None).is_err());
    assert!(paint_bgra_multi(&ref_img, &[&chan[..chan.len() - 4]], w, h, &p, None).is_err());
}

/// Synthetic 48x32 reference + second channel (same images as the replay test).
fn synth_pair() -> (usize, usize, Vec<u8>, Vec<u8>) {
    let (w, h) = (48usize, 32usize);
    let mut ref_img = vec![0u8; w * h * 4];
    let mut chan = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            ref_img[i] = (x * 5) as u8;
            ref_img[i + 1] = (y * 7) as u8;
            ref_img[i + 2] = 128;
            ref_img[i + 3] = 255;
            chan[i] = 200u8.saturating_sub((x * 3 % 201) as u8);
            chan[i + 1] = ((x * x / 4 + y * 3) % 256) as u8;
            chan[i + 2] = ((x + y * 11) % 256) as u8;
            chan[i + 3] = 255;
        }
    }
    (w, h, ref_img, chan)
}

#[test]
fn impasto_multi_frames() {
    let (w, h, ref_img, chan) = synth_pair();
    let params = |impasto: f64| Params { brush_len: 12, brush_w: 3, layers: 2, relief: 0.1,
                                         base_radius: 3, flow_smooth: 4, seed: 7,
                                         max_size: None, impasto, impasto_knife: false,
                                         boil: 0.0, boil_frame: 0 };
    let kinds = |a: &str, b: &str| vec![Some(a.to_string()), Some(b.to_string())];
    let cn = kinds("color", "normal");
    let off = paint_bgra_multi(&ref_img, &[&chan], w, h, &params(0.0), Some(&cn)).unwrap();
    assert_eq!(off.len(), 2, "impasto 0 must not add a height frame");

    let on = paint_bgra_multi(&ref_img, &[&chan], w, h, &params(1.0), Some(&cn)).unwrap();
    assert_eq!(on.len(), 4, "impasto > 0 appends the height + preview frames");
    assert_ne!(on[0], off[0], "color reference not shaded by impasto");
    assert_eq!(on[1], off[1], "normal channel must not be touched by impasto");
    let hf = &on[2];
    assert!(hf.chunks(4).all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 255), "height not gray");
    let (lo, hi) = hf.chunks(4).fold((255u8, 0u8), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    assert!(hi > lo, "height frame is flat");
    let pv = &on[3];
    assert!(pv.chunks(4).all(|p| p[3] == 255));
    let (plo, phi) = pv.chunks(4).fold((255u8, 0u8), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    assert!(phi > plo, "preview frame is flat");

    // frame-aligned kinds: [normal, color] shades frame 1, never the reference
    let nc = kinds("normal", "color");
    let on2 = paint_bgra_multi(&ref_img, &[&chan], w, h, &params(1.0), Some(&nc)).unwrap();
    assert_eq!(on2[0], off[0], "normal-kind reference was shaded");
    assert_ne!(on2[1], off[1], "color-kind channel not shaded");

    // single-image path honours impasto
    let (single_off, _) = paint_bgra(&ref_img, w, h, &params(0.0));
    let (single_on, _) = paint_bgra(&ref_img, w, h, &params(1.0));
    assert_eq!(single_off, off[0], "single path diverged from multi reference at impasto 0");
    assert_ne!(single_on, single_off, "single path ignores impasto");
}

#[test]
fn impasto_reference_only() {
    let (w, h, ref_img, _) = synth_pair();
    let params = |impasto: f64| Params { brush_len: 12, brush_w: 3, layers: 2, relief: 0.1,
                                         base_radius: 3, flow_smooth: 4, seed: 7,
                                         max_size: None, impasto, impasto_knife: false,
                                         boil: 0.0, boil_frame: 0 };
    let kinds = vec![Some("normal".to_string())];
    let off = paint_bgra_multi(&ref_img, &[], w, h, &params(0.0), Some(&kinds)).unwrap();
    assert_eq!(off.len(), 1);
    let on = paint_bgra_multi(&ref_img, &[], w, h, &params(1.0), Some(&kinds)).unwrap();
    assert_eq!(on.len(), 3, "reference + height + preview");
    assert_eq!(on[0], off[0], "normal-only reference must stay unshaded");
    assert!(on[2].chunks(4).all(|p| p[3] == 255));
    let (lo, hi) = on[2].chunks(4).fold((255u8, 0u8), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    assert!(hi > lo, "preview frame is flat");
}

#[test]
fn impasto_styles_and_range() {
    let (w, h) = (160usize, 120usize);
    let mut ref_img = vec![255u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            ref_img[i] = (x * 255 / w) as u8;
            ref_img[i + 1] = (y * 255 / h) as u8;
            ref_img[i + 2] = ((x + y) * 3 % 256) as u8;
        }
    }
    let params = |knife: bool| Params { brush_len: 30, brush_w: 8, layers: 2, relief: 0.1,
                                        base_radius: 3, flow_smooth: 4, seed: 7,
                                        max_size: None, impasto: 2.0, impasto_knife: knife,
                                        boil: 0.0, boil_frame: 0 };
    let kinds = vec![Some("normal".to_string())];
    let bristle = paint_bgra_multi(&ref_img, &[], w, h, &params(false), Some(&kinds)).unwrap();
    let knife = paint_bgra_multi(&ref_img, &[], w, h, &params(true), Some(&kinds)).unwrap();
    assert_ne!(bristle[1], knife[1], "knife and bristle heights are identical");
    for hf in [&bristle[1], &knife[1]] {
        let white = hf.chunks(4).filter(|p| p[0] == 255).count();
        assert!(white * 100 < w * h, "height frame clips: {white} white px of {}", w * h);
        assert!(hf.chunks(4).any(|p| p[0] > 100), "height frame is barely used");
    }
}

#[test]
fn boil_keeps_most_strokes() {
    let (w, h) = (128usize, 96usize);
    let mut bgr = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            bgr[i] = (x * 2) as u8;
            bgr[i + 1] = (y * 2) as u8;
            bgr[i + 2] = ((x * y) % 256) as u8;
        }
    }
    let gray = gray_from_bgr(&bgr, w, h);
    let flow = smooth_flow(flow_field(&gray, w, h, 2.5), 4);
    let n_strokes = (w * h) as i32 / 140;
    let base = generate_strokes(&bgr, w, h, &flow, n_strokes, 20, 3, 2, 0.6, 7);
    assert!(base.len() > 20);
    let boiled = |boil: f64, frame: u32| {
        let mut s = generate_strokes(&bgr, w, h, &flow, n_strokes, 20, 3, 2, 0.6, 7);
        boil_strokes(&mut s, w, h, &flow, n_strokes, 20, 3, 0.6, 7, frame, boil);
        s
    };
    let changed = |s: &[Stroke]| s.iter().zip(&base).filter(|(a, b)| a.pts != b.pts || a.lum != b.lum).count();
    assert_eq!(changed(&boiled(0.0, 1)), 0);
    let c30 = changed(&boiled(0.3, 1));
    assert!(c30 > 0 && c30 < base.len(), "boil 0.3 changed {c30} of {}", base.len());
    let c100 = changed(&boiled(1.0, 1));
    assert!(c100 * 10 >= base.len() * 8, "boil 1.0 changed only {c100} of {}", base.len());
    let (a, b) = (boiled(0.3, 2), boiled(0.3, 2));
    assert!(a.iter().zip(&b).all(|(x, y)| x.pts == y.pts && x.lum == y.lum && x.th == y.th));

    // paint path: frame 0 ignores boil, frame 1 differs but is deterministic
    let mut bgra = vec![255u8; w * h * 4];
    for i in 0..w * h {
        bgra[i * 4..i * 4 + 3].copy_from_slice(&bgr[i * 3..i * 3 + 3]);
    }
    let p = |boil: f64, boil_frame: u32| Params { brush_len: 20, brush_w: 3, layers: 2, relief: 0.1,
                                                  base_radius: 3, flow_smooth: 4, seed: 7,
                                                  max_size: None, impasto: 0.0, impasto_knife: false,
                                                  boil, boil_frame };
    let plain = paint_bgra(&bgra, w, h, &p(0.0, 0)).0;
    assert_eq!(paint_bgra(&bgra, w, h, &p(0.3, 0)).0, plain, "frame 0 must ignore boil");
    let f1 = paint_bgra(&bgra, w, h, &p(0.3, 1)).0;
    assert_ne!(f1, plain);
    assert_eq!(f1, paint_bgra(&bgra, w, h, &p(0.3, 1)).0);
    let same = f1.chunks(4).zip(plain.chunks(4)).filter(|(a, b)| a == b).count();
    assert!(same * 2 > w * h, "boil 0.3 repainted most of the frame ({same} same px)");
}

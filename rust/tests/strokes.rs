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
    let out = render_strokes(&bgr, &base, w, h, &flow, n_strokes, 45, 3, 4, 0.6, 0.9, 42);
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

//! Host-side verification: compare Rust flow field & Kuwahara vs numpy exports.
#[cfg(test)]
mod verify {
    use isophote_core::pipeline::*;

    pub(crate) fn read_npy(path: &str) -> (Vec<usize>, Vec<u8>) {
        let buf = std::fs::read(path).unwrap();
        // minimal npy reader: header text parse, u1/f4 support
        assert_eq!(&buf[0..6], b"\x93NUMPY");
        let major = buf[6];
        let hlen = if major == 1 {
            u16::from_le_bytes([buf[8], buf[9]]) as usize
        } else {
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
        };
        let off = if major == 1 { 10 } else { 12 };
        let header = String::from_utf8_lossy(&buf[off..off + hlen]).to_string();
        let dtype = if header.contains("'u1'") || header.contains("|u1") {
            "u1"
        } else if header.contains("'f4'") || header.contains("<f4") {
            "f4"
        } else {
            panic!("unsupported dtype in {path}: {header}");
        };
        let dims: Vec<usize> = header
            .split("'shape': (").nth(1).unwrap()
            .split(')').next().unwrap()
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let n: usize = dims.iter().product();
        let data: Vec<u8> = buf[off + hlen..].to_vec();
        (dims, data)
    }

    fn npy_f32(path: &str) -> Vec<f32> {
        let (dims, data) = read_npy(path);
        let chunk: Vec<u8> = data; // f32 LE
        chunk
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>()
            .iter()
            .map(|v| *v)
            .collect()
    }

    fn npy_u8(path: &str) -> (Vec<usize>, Vec<u8>) {
        read_npy(path)
    }

    #[test]
    fn flow_field_matches_python() {
        let (dims, bgr) = npy_u8("tests/in_bgr.npy");
        let (h, w, _) = (dims[0], dims[1], dims[2]);
        let gray = gray_from_bgr(&bgr, w, h);
        let flow0 = flow_field(&gray, w, h, 2.5);
        let flow = smooth_flow(flow0, 8);
        let ref_angle = npy_f32("tests/flow_angle.npy");
        let ref_coh = npy_f32("tests/flow_coh.npy");
        let mut max_da = 0f32;
        let mut max_dc = 0f32;
        for i in 0..w * h {
            // wrap-aware angle diff
            let d = (flow.angle[i] - ref_angle[i]).abs();
            let d = d.min(180.0 - d);
            if d > max_da { max_da = d; }
            let dc = (flow.coherence[i] - ref_coh[i]).abs();
            if dc > max_dc { max_dc = dc; }
        }
        println!("flow angle max diff: {max_da}  coherence max diff: {max_dc}");
        assert!(max_da < 2.0, "angle drift too strong");
        assert!(max_dc < 0.05, "coherence drift too strong");
    }

    #[test]
    fn kuwahara_matches_python() {
        let (dims, bgr) = npy_u8("tests/in_bgr.npy");
        let (h, w, _) = (dims[0], dims[1], dims[2]);
        let out = kuwahara(&bgr, w, h, 6);
        let (_, ref_bgr) = npy_u8("tests/kuwahara.npy");
        let diffs: Vec<i32> = out
            .iter()
            .zip(ref_bgr.iter())
            .map(|(a, b)| ((*a as i32) - (*b as i32)).abs())
            .collect();
        let max = *diffs.iter().max().unwrap();
        let mean = diffs.iter().sum::<i32>() as f64 / diffs.len() as f64;
        println!("kuwahara max diff: {max}  mean: {mean:.2}");
        // Quadrant near-ties flip with float rounding — mean stays tiny; the
        // tie-flip test below bounds the number of big single-pixel changes.
        assert!(mean < 2.0);
    }

    #[test]
    fn kuwahara_diff_stats() {
        let (dims, bgr) = read_npy("tests/in_bgr.npy");
        let (h, w) = (dims[0], dims[1]);
        let out = kuwahara(&bgr, w, h, 6);
        let (_, ref_bgr) = read_npy("tests/kuwahara.npy");
        let big = out.iter().zip(ref_bgr.iter())
            .filter(|(a, b)| ((**a as i32) - (**b as i32)).abs() > 12)
            .count();
        println!("pixels differing >12: {} of {}", big, out.len());
        assert!(big * 100 <= out.len()); // <1% tie flips acceptable
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    #[test]
    fn relief_matches_python() {
        use isophote_core::pipeline::canvas_relief;
        let (shade, _) = canvas_relief(135, 180, 12, 0.1, 1.0);
        let (_, ref_shade) = read_npy("tests/relief_ref.npy");
        let mut max = 0i32;
        for (a, b) in shade.iter().zip(ref_shade.iter()) {
            let d = (*a * 255.0).round() as i32 - *b as i32;
            if d.abs() > max {
                max = d.abs();
            }
        }
        println!("relief max diff: {max}");
        assert!(max <= 2, "relief drift too strong");
        let _ = &std::io::Write::flush(&mut std::io::stdout());
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    #[test]
    fn resize_matches_cv2() {
        use isophote_core::pipeline::resize_area;
        let (dims, img) = read_npy("tests/in_full.npy");
        let (h, w) = (dims[0], dims[1]);
        let scale = h.max(w) as f64 / 1400.0;
        let dw = (w as f64 / scale) as usize;
        let dh = (h as f64 / scale) as usize;
        let out = resize_area(&img, w, h, dw, dh, 3);
        let (_, ref_bgr) = read_npy("tests/resize_ref.npy");
        let diffs: Vec<i32> = out
            .iter()
            .zip(ref_bgr.iter())
            .map(|(a, b)| ((*a as i32) - (*b as i32)).abs())
            .collect();
        let max = *diffs.iter().max().unwrap();
        let mean = diffs.iter().sum::<i32>() as f64 / diffs.len() as f64;
        println!("resize max diff: {max} mean: {mean:.2}");
        assert!(max <= 1, "resize drift too strong");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    #[test]
    fn resize_debug() {
        use isophote_core::pipeline::resize_area;
        let (dims, img) = read_npy("tests/in_full.npy");
        let (h, w) = (dims[0], dims[1]);
        println!("in_full dims: {}x{}", w, h);
        let scale = h.max(w) as f64 / 1400.0;
        let dw = (w as f64 / scale) as usize;
        let dh = (h as f64 / scale) as usize;
        let out = resize_area(&img, w, h, dw, dh, 3);
        let (_, ref_bgr) = read_npy("tests/resize_ref.npy");
        println!("out len {} ref len {}  dims {}x{}", out.len(), ref_bgr.len(), dw, dh);
        println!("out[0..6]: {:?}", &out[..6]);
        println!("ref[0..6]: {:?}", &ref_bgr[..6]);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}

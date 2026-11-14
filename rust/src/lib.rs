pub mod pcg64;
pub mod pipeline;

pub use pcg64::Pcg64;
pub use pipeline::paint_bgra;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn hex_to_f64(h: &str) -> f64 {
        // numpy .hex() format: "0x1.461fd79fb3850p-1"
        let (mant, exp) = h.split_once('p').unwrap();
        // mantissa after "0x1.": digits starting at first fraction digit
        let mant_body = mant.trim_start_matches("0x1.");
        let is_neg = h.starts_with('-');
        let mut v = 1.0f64;
        let mut scale = 1.0f64 / 16.0;
        for c in mant_body.chars() {
            let d = c.to_digit(16).unwrap() as f64;
            v += d * scale;
            scale /= 16.0;
        }
        let e: i32 = exp.parse().unwrap();
        v * 2f64.powi(e) * if is_neg { -1.0 } else { 1.0 }
    }

    /// numpy_vectors.json: ground truth from numpy. Tolerance: exact bit match
    /// except last-dabble rounding — allow 1 ulp.
    #[test]
    fn pcg64_matches_numpy() {
        let vecs: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
            &fs::read_to_string("tests/numpy_vectors.json").unwrap(),
        )
        .unwrap();
        let mut checked = 0;
        for (seed, vec) in vecs {
            let hexes: Vec<String> = serde_json::from_value(vec).unwrap();
            let mut p = Pcg64::new(seed.parse::<u64>().unwrap());
            for h in hexes {
                let want = hex_to_f64(&h);
                let got = p.next_f64();
                let ulp_diff = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                assert!(
                    ulp_diff == 0,
                    "seed {seed}: got {} (0x{:x}), want {} (0x{:x})",
                    got,
                    got.to_bits(),
                    want,
                    want.to_bits()
                );
                checked += 1;
            }
        }
        assert!(checked >= 24);
    }
}

// ------------------------------------------------------- wasm C ABI exports

/// Allocate `len` bytes in wasm linear memory. JS fills the returned buffer
/// via `new Uint8Array(memory.buffer, ptr, len)`.
#[no_mangle]
pub extern "C" fn wasm_alloc(len: i32) -> *mut u8 {
    if len <= 0 {
        return std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::from_size_align(len as usize, 16).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

/// Free a pointer from `wasm_alloc`.
#[no_mangle]
pub extern "C" fn wasm_free(ptr: *mut u8, len: i32) {
    if ptr.is_null() || len <= 0 {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(len as usize, 16).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) };
}

/// Paint a BGRA image in-place semantics: reads BGRA from `in_ptr`
/// (w*h*4 bytes), writes painted BGRA to `out_ptr` (w*h*4 bytes).
/// `params_ptr`/`params_len` point at UTF-8 JSON:
///   {"brush_len":26,"brush_w":7,"layers":2,"relief":0.35,
///    "base_radius":6,"flow_smooth":8,"seed":42,"max_size":1400}
/// Returns elapsed milliseconds as i32.
///
/// The input is resized here with cv2-INTER_AREA-exact weights when
/// max_size > max_dim (matching isophote.paint_bytes).
#[no_mangle]
pub extern "C" fn wasm_paint(
    w: i32,
    h: i32,
    in_ptr: *const u8,
    out_ptr: *mut u8,
    params_ptr: *const u8,
    params_len: i32,
) -> i32 {
    if w <= 0 || h <= 0 || in_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let (w, h) = (w as usize, h as usize);
    let in_bytes = unsafe { std::slice::from_raw_parts(in_ptr, w * h * 4) };
    let json = if params_ptr.is_null() || params_len <= 0 {
        String::new()
    } else {
        let pslice = unsafe { std::slice::from_raw_parts(params_ptr, params_len as usize) };
        String::from_utf8_lossy(pslice).into_owned()
    };
    let p: pipeline::Params = if json.is_empty() {
        pipeline::Params {
            brush_len: 26,
            brush_w: 7,
            layers: 2,
            relief: 0.35,
            base_radius: 6,
            flow_smooth: 8,
            seed: 42,
        }
    } else {
        match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(_) => return -2,
        }
    };

    // cv2-compatible downscale to max_size before painting (INTER_AREA path)
    let max_size = 1400i32;
    let src_buf;
    let (sw, sh);
    if w.max(h) as i32 > max_size {
        let scale = w.max(h) as f64 / max_size as f64;
        let dw = ((w as f64 / scale) as usize).max(1);
        let dh = ((h as f64 / scale) as usize).max(1);
        src_buf = pipeline::resize_area(in_bytes, w, h, dw, dh, 4);
        (sw, sh) = (dw, dh);
    } else {
        src_buf = in_bytes.to_vec();
        (sw, sh) = (w, h);
    }

    let (painted, _secs) = pipeline::paint_bgra(&src_buf, sw, sh, &p);
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, sw * sh * 4) };
    out.copy_from_slice(&painted);
    0 // timing measured on the JS side (performance.now)
}

/// Painted dimensions (after possible internal downscale) for the last call.
/// Returns packed (sw << 16) | sh.
#[no_mangle]
pub extern "C" fn wasm_paint_out_dims(w: i32, h: i32) -> i64 {
    let max_size = 1400i32;
    if w.max(h) > max_size {
        let scale = w.max(h) as f64 / max_size as f64;
        let dw = ((w as f64 / scale) as usize).max(1);
        let dh = ((h as f64 / scale) as usize).max(1);
        return (((dh as i64) << 32) | (dw as i64 & 0xffff_ffff)) as i64;
    }
    ((h as i64) << 32) | (w as i64 & 0xffff_ffff)
}

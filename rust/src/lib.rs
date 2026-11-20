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

/// Painted dims of the last wasm_paint call (packed (h << 16) | w as i64).
static LAST_OUT_DIMS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Frame count written by the last successful wasm_paint_multi call.
static LAST_MULTI_FRAMES: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// f32 bits of the last wasm_paint_multi height scale (see wasm_paint_multi_height_scale).
static LAST_HEIGHT_SCALE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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
            max_size: None,
            impasto: 0.0,
            impasto_knife: false,
            boil: 0.0,
            boil_frame: 0,
        }
    } else {
        match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(_) => return -2,
        }
    };

    // cv2-compatible downscale before painting (INTER_AREA path).
    // max_size param overrides the 1400 default (0 disables the cap).
    let max_size = p.max_size.unwrap_or(1400).max(0) as i32;
    let src_buf;
    let (sw, sh);
    match pipeline::downscale_dims(w, h, max_size) {
        Some((dw, dh, _s)) => {
            src_buf = pipeline::resize_area(in_bytes, w, h, dw, dh, 4);
            (sw, sh) = (dw, dh);
        }
        None => {
            src_buf = in_bytes.to_vec();
            (sw, sh) = (w, h);
        }
    }

    let (painted, _secs) = pipeline::paint_bgra(&src_buf, sw, sh, &p);
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, sw * sh * 4) };
    out.copy_from_slice(&painted);
    LAST_OUT_DIMS.store(
        (((sh as i64) << 32) | (sw as i64 & 0xffff_ffff)) as i64,
        std::sync::atomic::Ordering::Relaxed);
    0 // timing measured on the JS side (performance.now)
}

/// Painted dimensions (after possible internal downscale) for the last call.
/// Returns packed (sw << 16) | sh.
#[no_mangle]
pub extern "C" fn wasm_paint_out_dims(_w: i32, _h: i32) -> i64 {
    // painted dims of the last wasm_paint call (accounts for max_size cap)
    LAST_OUT_DIMS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Multi-channel coherent paint ("stroke replay"). Blob at `in_ptr` is
/// `k` sequential BGRA frames (each `w*h*4`); **frame 0 is the reference**
/// (drives flow + strokes), frames 1..k are the other channels painted with
/// the same strokes. Params JSON identical to `wasm_paint`.
///
/// `out_ptr` must hold `(k+2)*w*h*4` bytes (no downscale here: JS sends
/// already-decoded same-size frames). On success the painted BGRA frames
/// are written sequentially at `out_ptr` (reference first, same order as
/// input) and the return value is `0`; read `wasm_paint_multi_frames()`
/// frames: `k`, or `k+2` when impasto > 0 (gray impasto height map, then
/// the display-only relief preview). Params may carry frame-aligned
/// `"kinds"` (kinds[0] = reference) so impasto shades only "color" frames.
///
/// On failure returns `N > 0`: the caller must then read the first `N`
/// bytes at `out_ptr` as the UTF-8 error message. Returns `-1` only for
/// null pointers / non-positive dims (no message written).
///
/// # Safety
/// `in_ptr`/`out_ptr`/`params_ptr` must be valid for the described lengths.
#[no_mangle]
pub extern "C" fn wasm_paint_multi(
    w: i32,
    h: i32,
    k: i32,
    in_ptr: *const u8,
    out_ptr: *mut u8,
    params_ptr: *const u8,
    params_len: i32,
) -> i32 {
    if w <= 0 || h <= 0 || k <= 0 || k > 8 || in_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let (w, h, k) = (w as usize, h as usize, k as usize);
    let frame = w * h * 4;
    let blob = unsafe { std::slice::from_raw_parts(in_ptr, frame * k) };
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
            max_size: None,
            impasto: 0.0,
            impasto_knife: false,
            boil: 0.0,
            boil_frame: 0,
        }
    } else {
        match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => return write_err(out_ptr, frame * k, format!("bad params: {e}")),
        }
    };
    let frames: Vec<&[u8]> = (0..k).map(|i| &blob[i * frame..(i + 1) * frame]).collect();
    // frame-aligned kind names (kinds[0] = reference) from the same params
    // JSON; impasto shades only "color" frames
    let kinds: Option<Vec<Option<String>>> = serde_json::from_str::<pipeline::KindsHint>(&json)
        .ok()
        .map(|h| h.kinds);
    match pipeline::paint_bgra_multi(frames[0], &frames[1..], w, h, &p, kinds.as_deref()) {
        Ok(painted) => {
            // no downscale in the multi path: same dims in/out
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, frame * painted.len()) };
            for (dst, src) in out.chunks_exact_mut(frame).zip(painted.iter()) {
                dst.copy_from_slice(src);
            }
            LAST_MULTI_FRAMES.store(painted.len() as i32, std::sync::atomic::Ordering::Relaxed);
            LAST_HEIGHT_SCALE.store(
                pipeline::impasto_height_scale(w, h, &p).to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            0
        }
        Err(e) => write_err(out_ptr, frame * k, e),
    }
}

/// Painted frame count of the last successful wasm_paint_multi call
/// (k, or k+2 with the impasto height + relief preview frames).
#[no_mangle]
pub extern "C" fn wasm_paint_multi_frames() -> i32 {
    LAST_MULTI_FRAMES.load(std::sync::atomic::Ordering::Relaxed)
}

/// UV-unit thickness of a white height pixel x impasto for the last
/// successful wasm_paint_multi call (0 = no height frame).
#[no_mangle]
pub extern "C" fn wasm_paint_multi_height_scale() -> f32 {
    f32::from_bits(LAST_HEIGHT_SCALE.load(std::sync::atomic::Ordering::Relaxed))
}

/// Write a UTF-8 error message at `out_ptr`, truncated/zero-padded to `cap`
/// bytes; returns the message length as the export's error return.
fn write_err(out_ptr: *mut u8, cap: usize, msg: String) -> i32 {
    let bytes = msg.as_bytes();
    let n = bytes.len().min(cap);
    if n > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, cap) };
        // zero the whole buffer first so the caller can bound the message
        out.iter_mut().for_each(|b| *b = 0);
        out[..n].copy_from_slice(&bytes[..n]);
    }
    n as i32
}

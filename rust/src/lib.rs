pub mod pcg64;
pub mod pipeline;
pub mod surface;

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

/// Largest PaintStage (+ height field) kept between calls; bigger ones are
/// dropped right after use so wasm memory does not stay pinned.
const CACHE_MAX_BYTES: usize = 128 << 20;

/// Last paint stage + impasto height, keyed by everything that shapes the
/// painting except impasto strength/brush/out_size/kinds (those only change
/// the final shading or output size, so tweaking them skips the repaint).
struct StageCache {
    key: u64,
    stage: pipeline::PaintStage,
    /// (brush, height field) for the last impasto brush computed on this stage
    height: Option<(u32, Vec<f32>)>,
}

static CACHE: std::sync::Mutex<Option<StageCache>> = std::sync::Mutex::new(None);

/// Mesh for surface painting (`uv_aware`), set by `wasm_set_mesh`: hash of
/// its bytes (part of the stage cache key) + the parsed mesh.
static MESH: std::sync::Mutex<Option<(u64, std::sync::Arc<surface::Mesh>)>> = std::sync::Mutex::new(None);

/// 64-bit multiply-xorshift hash over 8-byte little-endian words (std's
/// SipHash is ~10x slower on wasm32 and inputs reach hundreds of MB).
fn hash_words(mut h: u64, bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(K);
        h ^= h >> 32;
    }
    for &b in chunks.remainder() {
        h = (h ^ b as u64).wrapping_mul(K);
        h ^= h >> 32;
    }
    h
}

/// Cache key: entry kind, dims, frame count, every painting param, then the raw input bytes.
fn stage_key(tag: u8, w: usize, h: usize, k: usize, p: &pipeline::Params, input: &[u8]) -> u64 {
    let mut key = hash_words(0xcbf2_9ce4_8422_2325, &[tag]);
    for v in [
        w as u64,
        h as u64,
        k as u64,
        p.brush_len as i64 as u64,
        p.brush_w as i64 as u64,
        p.layers as i64 as u64,
        p.relief.to_bits(),
        p.base_radius as i64 as u64,
        p.flow_smooth as i64 as u64,
        p.seed,
        p.max_size.unwrap_or(-1) as i64 as u64,
        p.boil.to_bits(),
        p.boil_frame as u64,
        p.look().to_bits(),
    ] {
        key = hash_words(key, &v.to_le_bytes());
    }
    hash_words(key, input)
}

/// Run `f` on the cached stage for `key` (built with `build` on a miss) and,
/// when `p.impasto > 0`, the impasto height for `p.impasto_brush` (computed
/// once per stage and brush).
fn with_stage<R>(
    key: u64,
    p: &pipeline::Params,
    build: impl FnOnce() -> pipeline::PaintStage,
    f: impl FnOnce(&pipeline::PaintStage, Option<&[f32]>) -> R,
) -> R {
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().map_or(true, |c| c.key != key) {
        *guard = None; // free the old stage before building the new one
        *guard = Some(StageCache { key, stage: build(), height: None });
    }
    let c = guard.as_mut().unwrap();
    if p.impasto > 0.0 && c.height.as_ref().map_or(true, |(brush, _)| *brush != p.impasto_brush) {
        let hf = pipeline::stage_height(&c.stage, p.impasto_brush, p.look());
        c.height = Some((p.impasto_brush, hf));
    }
    let hgt = if p.impasto > 0.0 { c.height.as_ref().map(|(_, v)| &v[..]) } else { None };
    let r = f(&c.stage, hgt);
    let px = c.stage.w * c.stage.h;
    let bytes = (c.stage.frames.len() + 1) * px * 4
        + c.height.as_ref().map_or(0, |_| px * 4)
        + c.stage.strokes.iter().map(|s| s.pts.len() * 8 + 32).sum::<usize>()
        + c.stage.surf.as_ref().map_or(0, |s| {
            px * 12 + s.map.tris.len() * 108 + s.strokes.iter().map(|k| k.pts.len() * 28 + 32).sum::<usize>()
        });
    if bytes > CACHE_MAX_BYTES {
        *guard = None;
    }
    r
}

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

/// Set (or, with `len <= 0`, clear) the mesh used by `uv_aware` paints: the
/// add-on's mesh blob (layout in `surface::Mesh::parse`). Returns 0 on
/// success, 1 when the blob is malformed (the mesh is then cleared).
#[no_mangle]
pub extern "C" fn wasm_set_mesh(ptr: *const u8, len: i32) -> i32 {
    let mut slot = MESH.lock().unwrap_or_else(|e| e.into_inner());
    *slot = None;
    if ptr.is_null() || len <= 0 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match surface::Mesh::parse(bytes) {
        Ok(m) => {
            *slot = Some((hash_words(0xcbf2_9ce4_8422_2325, bytes), std::sync::Arc::new(m)));
            0
        }
        Err(_) => 1,
    }
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
            impasto_brush: 0,
            boil: 0.0,
            boil_frame: 0,
            look_scale: 0.0,
            out_size: 0,
            normal_tangent: false,
            uv_aware: false,
            shape_follow: 0.0,
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
    let key = stage_key(1, w, h, 1, &p, in_bytes);
    let build = || {
        let resized;
        let (src, sw, sh): (&[u8], usize, usize) = match pipeline::downscale_dims(w, h, max_size) {
            Some((dw, dh, _s)) => {
                resized = pipeline::resize_area(in_bytes, w, h, dw, dh, 4);
                (&resized[..], dw, dh)
            }
            None => (in_bytes, w, h),
        };
        pipeline::paint_stage(src, &[], sw, sh, &p, None)
    };
    with_stage(key, &p, build, |stage, hgt| {
        let painted = pipeline::finish_single(stage, hgt, &p);
        let (sw, sh) = (stage.w, stage.h);
        // out_size: area-downscale the painted frame (brushwork was laid at sw x sh)
        let (ow, oh) = out_dims(sw, sh, p.out_size);
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, ow * oh * 4) };
        if (ow, oh) == (sw, sh) {
            out.copy_from_slice(&painted);
        } else {
            out.copy_from_slice(&pipeline::resize_area(&painted, sw, sh, ow, oh, 4));
        }
        LAST_OUT_DIMS.store(
            (((oh as i64) << 32) | (ow as i64 & 0xffff_ffff)) as i64,
            std::sync::atomic::Ordering::Relaxed);
    });
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
/// `out_ptr` must hold `(k+2)*w*h*4` bytes (JS sends already-decoded
/// same-size frames; `out_size` in the params area-downscales the painted
/// frames to that long edge, so the output never exceeds this). On success
/// the painted BGRA frames
/// are written sequentially at `out_ptr`, each `ow*oh*4` bytes (dims from
/// `wasm_paint_out_dims`; reference first, same order as
/// input) and the return value is `0`; read `wasm_paint_multi_frames()` for the
/// frame count: `k`, or `k+2` when impasto > 0 (gray impasto height map, then
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
            impasto_brush: 0,
            boil: 0.0,
            boil_frame: 0,
            look_scale: 0.0,
            out_size: 0,
            normal_tangent: false,
            uv_aware: false,
            shape_follow: 0.0,
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
    if let Err(e) = pipeline::validate_multi(frames[0], &frames[1..], w, h) {
        return write_err(out_ptr, frame * k, e);
    }
    // surface painting needs the mesh from wasm_set_mesh; without one the
    // paint is the classic flat-texture one
    let mesh = if p.uv_aware { MESH.lock().unwrap_or_else(|e| e.into_inner()).clone() } else { None };
    let mut key = stage_key(2, w, h, k, &p, blob);
    if let Some((mesh_hash, _)) = &mesh {
        key = hash_words(key, &mesh_hash.to_le_bytes());
        key = hash_words(key, &p.shape_follow.to_bits().to_le_bytes());
    }
    let mesh = mesh.as_ref().map(|(_, m)| &**m);
    with_stage(key, &p, || pipeline::paint_stage(frames[0], &frames[1..], w, h, &p, mesh), |stage, hgt| {
        let painted = pipeline::finish_multi(stage, hgt, &p, kinds.as_deref());
        // out_size: area-downscale each painted frame to the requested long edge
        let (ow, oh) = out_dims(w, h, p.out_size);
        let fo = ow * oh * 4;
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, fo * painted.len()) };
        for (dst, src) in out.chunks_exact_mut(fo).zip(painted.iter()) {
            if (ow, oh) == (w, h) {
                dst.copy_from_slice(src);
            } else {
                dst.copy_from_slice(&pipeline::resize_area(src, w, h, ow, oh, 4));
            }
        }
        LAST_OUT_DIMS.store(
            (((oh as i64) << 32) | (ow as i64 & 0xffff_ffff)) as i64,
            std::sync::atomic::Ordering::Relaxed);
        LAST_MULTI_FRAMES.store(painted.len() as i32, std::sync::atomic::Ordering::Relaxed);
        LAST_HEIGHT_SCALE.store(
            pipeline::impasto_height_scale(w, h, &p).to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
    });
    0
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

/// Output dims for a painted `w x h` frame under `out_size` (0 / >= long edge = unchanged).
fn out_dims(w: usize, h: usize, out_size: i32) -> (usize, usize) {
    pipeline::downscale_dims(w, h, out_size).map_or((w, h), |(dw, dh, _)| (dw, dh))
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

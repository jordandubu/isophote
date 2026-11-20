//! The wasm exports keep the last paint stage; every call (cache hit or miss)
//! must equal a from-scratch `pipeline::paint_bgra*` with the same params.
use isophote_core::pipeline::{paint_bgra, paint_bgra_multi, Params};
use isophote_core::{wasm_paint, wasm_paint_multi, wasm_paint_multi_frames, wasm_paint_out_dims};

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

fn params(impasto: f64, knife: bool, seed: u64) -> Params {
    Params {
        brush_len: 30, brush_w: 8, layers: 2, relief: 0.1, base_radius: 3, flow_smooth: 4, seed,
        max_size: None, impasto, impasto_knife: knife, boil: 0.0, boil_frame: 0,
        look_scale: 0.0, out_size: 0,
    }
}

fn json(p: &Params, kinds: bool) -> String {
    format!(
        "{{\"brush_len\":{},\"brush_w\":{},\"layers\":{},\"relief\":{},\"base_radius\":{},\"flow_smooth\":{},\
         \"seed\":{},\"impasto\":{},\"impasto_knife\":{}{}}}",
        p.brush_len, p.brush_w, p.layers, p.relief, p.base_radius, p.flow_smooth, p.seed, p.impasto, p.impasto_knife,
        if kinds { ",\"kinds\":[\"color\",\"normal\"]" } else { "" }
    )
}

const SEQ: [(f64, bool, u64); 6] = [
    (0.5, false, 7),
    (1.0, false, 7), // strength change: cache hit
    (1.0, true, 7),  // knife toggle: cache hit, new height
    (0.0, false, 7), // impasto off: cache hit, no shading
    (1.0, false, 8), // new seed: miss
    (1.0, false, 7), // back to the first seed: miss, still correct
];

#[test]
fn cached_calls_equal_fresh_paints() {
    let (w, h) = (96usize, 64usize);
    let (r0, c1) = (synth(w, h, 0), synth(w, h, 1));
    let kinds = vec![Some("color".to_string()), Some("normal".to_string())];

    for (impasto, knife, seed) in SEQ {
        let p = params(impasto, knife, seed);
        let js = json(&p, false);
        let mut out = vec![0u8; w * h * 4];
        let ret = wasm_paint(w as i32, h as i32, r0.as_ptr(), out.as_mut_ptr(), js.as_ptr(), js.len() as i32);
        assert_eq!(ret, 0);
        assert_eq!(out, paint_bgra(&r0, w, h, &p).0, "single impasto {impasto} knife {knife} seed {seed}");
    }

    let mut blob = r0.clone();
    blob.extend_from_slice(&c1);
    for (impasto, knife, seed) in SEQ {
        let p = params(impasto, knife, seed);
        let js = json(&p, true);
        let mut out = vec![0u8; 4 * w * h * 4];
        let ret = wasm_paint_multi(w as i32, h as i32, 2, blob.as_ptr(), out.as_mut_ptr(), js.as_ptr(), js.len() as i32);
        assert_eq!(ret, 0);
        let want = paint_bgra_multi(&r0, &[&c1], w, h, &p, Some(&kinds)).unwrap();
        assert_eq!(wasm_paint_multi_frames() as usize, want.len());
        let dims = wasm_paint_out_dims(w as i32, h as i32);
        assert_eq!((dims & 0xffff_ffff) as usize, w);
        for (i, f) in want.iter().enumerate() {
            assert_eq!(&out[i * w * h * 4..(i + 1) * w * h * 4], &f[..], "multi frame {i} impasto {impasto} knife {knife} seed {seed}");
        }
    }
}

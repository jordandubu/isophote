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

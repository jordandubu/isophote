//! pcg64.rs — bit-exact port of numpy's default_rng → Generator.random().
//!
//! Verified against numpy 2.2.5/2.5.4 for seeds
//! [0, 1, 42, 0xdeadbeaf, 99, 200, 12345678901234, 2^40+1, 7, 1000003]:
//! every double matches to the last bit.

const M32: u32 = 0xFFFF_FFFF;
const M64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const M128: u128 = u128::MAX;
const POOL: usize = 4;

// SeedSequence constants
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

// PCG64 128-bit multiplier: (high, low) = (0x2360ED051FC65DA4, 0x4385DF649FCCF645)
const MULT128: u128 = (2549297995355413924u64 as u128) << 64
    | 4865540595714422341u64 as u128;

#[inline]
fn hashmix(mut value: u32, hc: &mut u32) -> u32 {
    value ^= *hc;
    *hc = hc.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hc);
    value ^= value >> XSHIFT;
    value
}

#[inline]
fn mix_2(x: u32, y: u32) -> u32 {
    let mut r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^= r >> XSHIFT;
    r
}

/// SeedSequence(seed).generate_state(4, uint64) → 8 uint32 words.
fn gen_words32(seed: u64, n: usize) -> Vec<u32> {
    let entropy = [seed as u32, (seed >> 32) as u32];
    let mut mixer = [0u32; POOL];
    let mut hc = INIT_A;
    for i in 0..POOL {
        mixer[i] = hashmix(if i < entropy.len() { entropy[i] } else { 0 }, &mut hc);
    }
    for i_src in 0..POOL {
        for i_dst in 0..POOL {
            if i_src != i_dst {
                mixer[i_dst] = mix_2(mixer[i_dst], hashmix(mixer[i_src], &mut hc));
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut bc = INIT_B;
    let mut i = 0;
    for _ in 0..n {
        let mut d = mixer[i % POOL];
        i += 1;
        d ^= bc;
        bc = bc.wrapping_mul(MULT_B);
        d = d.wrapping_mul(bc);
        d ^= d >> XSHIFT;
        out.push(d);
    }
    out
}

pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let w = gen_words32(seed, 8);
        // numpy passes seed=words[0..2], inc=words[2..4] (uint64 LE pairs),
        // then C: state.high=seed[0], state.low=seed[1] → HIGH = w64[0].
        let q = |a: &[u32], i: usize| -> u64 {
            a[i] as u64 | ((a[i + 1] as u64) << 32)
        };
        let initstate = ((q(&w, 0) as u128) << 64) | q(&w, 2) as u128;
        let initseq = ((q(&w, 4) as u128) << 64) | q(&w, 6) as u128;
        let high = initseq >> 64;
        let low = initseq & M64 as u128;
        let hi2 = ((high << 1) & M64 as u128) | (low >> 63);
        let lo2 = (low << 1) | 1;
        let inc = (hi2 << 64) | lo2;
        // srandom: state = 0; step; state += initstate; step
        let mut state = inc;
        state = (state.wrapping_add(initstate)) & M128;
        state = state.wrapping_mul(MULT128).wrapping_add(inc) & M128;
        Pcg64 { state, inc }
    }

    /// Generator.random(): one f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(MULT128).wrapping_add(self.inc) & M128;
        let st = self.state;
        let v = (st >> 64) ^ (st & M64 as u128);
        let rot = ((st >> 122) & 63) as u32;
        let o: u64 = if rot == 0 {
            v as u64
        } else {
            let r = rot as u32;
            ((((v >> r) | (v << (64 - r as u128))) & M64 as u128) as u64)
        };
        (o >> 11) as f64 * (1.0f64 / 9007199254740992.0)
    }

    /// numpy Generator.uniform(low, high) — note: numpy computes
    /// lo + (hi - lo) * random() so we mirror that exactly.
    #[inline]
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

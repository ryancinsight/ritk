//! Exact port of `itk::Statistics::MersenneTwisterRandomVariateGenerator`
//! (MT19937), the uniform generator behind `SaltAndPepperNoise`.
//!
//! Reproduces SimpleITK bit-for-bit when seeded identically
//! (`SetSeed(Hash(userSeed, regionStart))`). `GetVariate()` returns a closed-range
//! `[0, 1]` double, `int / (2³² − 1)`.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

/// ITK MT19937 uniform generator. Stateful and sequential.
pub(crate) struct MersenneTwister {
    state: [u32; N],
    left: usize,
    next: usize,
}

#[inline]
fn twist(m: u32, s0: u32, s1: u32) -> u32 {
    let mix = (s0 & UPPER) | (s1 & LOWER);
    m ^ (mix >> 1) ^ (if s1 & 1 != 0 { MATRIX_A } else { 0 })
}

impl MersenneTwister {
    /// Seed the generator (ITK `SetSeed`).
    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            state[i] = (1_812_433_253u32)
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let mut g = Self {
            state,
            left: 0,
            next: 0,
        };
        g.reload();
        g
    }

    fn reload(&mut self) {
        let s = &mut self.state;
        for k in 0..(N - M) {
            s[k] = twist(s[k + M], s[k], s[k + 1]);
        }
        for k in (N - M)..(N - 1) {
            s[k] = twist(s[k + M - N], s[k], s[k + 1]);
        }
        s[N - 1] = twist(s[M - 1], s[N - 1], s[0]);
        self.left = N;
        self.next = 0;
    }

    /// Next 32-bit integer variate (tempered MT19937 output).
    pub(crate) fn integer(&mut self) -> u32 {
        if self.left == 0 {
            self.reload();
        }
        self.left -= 1;
        let mut s1 = self.state[self.next];
        self.next += 1;
        s1 ^= s1 >> 11;
        s1 ^= (s1 << 7) & 0x9d2c_5680;
        s1 ^= (s1 << 15) & 0xefc6_0000;
        s1 ^ (s1 >> 18)
    }

    /// Closed-range uniform variate in `[0, 1]` (ITK `GetVariate`).
    #[inline]
    pub(crate) fn variate(&mut self) -> f64 {
        self.integer() as f64 * (1.0 / 4_294_967_295.0)
    }
}

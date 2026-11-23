//! Seeded Gaussian spatial sampling in ITK traversal order.

use std::f64::consts::PI;

/// Bit-exact port of `itk::Statistics::MersenneTwisterRandomVariateGenerator`.
pub(super) struct ItkMt {
    state: [u32; 624],
    left: usize,
    next: usize,
}

impl ItkMt {
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        let mut mt = Self {
            state,
            left: 0,
            next: 0,
        };
        mt.reload();
        mt
    }

    #[inline]
    fn twist(m: u32, s0: u32, s1: u32) -> u32 {
        let mix = (s0 & 0x8000_0000) | (s1 & 0x7fff_ffff);
        m ^ (mix >> 1) ^ ((s1 & 1).wrapping_neg() & 0x9908_b0df)
    }

    fn reload(&mut self) {
        const M: usize = 397;
        let s = &mut self.state;
        for i in 0..(624 - M) {
            s[i] = Self::twist(s[i + M], s[i], s[i + 1]);
        }
        for i in (624 - M)..623 {
            s[i] = Self::twist(s[i + M - 624], s[i], s[i + 1]);
        }
        s[623] = Self::twist(s[M - 1], s[623], s[0]);
        self.left = 624;
        self.next = 0;
    }

    #[inline]
    pub(super) fn next_u32(&mut self) -> u32 {
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

    /// `GetNormalVariate(mean, variance)` via the Box-Muller transform.
    #[inline]
    fn normal(&mut self, mean: f64, variance: f64) -> f64 {
        // GetVariateWithOpenRange(): (u32 + 0.5) / 2^32.
        let u1 = (self.next_u32() as f64 + 0.5) / 4_294_967_296.0;
        let r = (-2.0 * (1.0 - u1).ln() * variance).sqrt();
        // GetVariateWithOpenUpperRange(): u32 / 2^32.
        let u2 = self.next_u32() as f64 / 4_294_967_296.0;
        mean + r * (2.0 * PI * u2).cos()
    }

    /// `GaussianRandomSpatialNeighborSubsampler::GetIntegerVariate`: draw
    /// `⌊N(mean, variance)⌋`, rejecting until within `[lo, hi]`.
    #[inline]
    pub(super) fn gauss_int(&mut self, lo: i64, hi: i64, mean: f64, variance: f64) -> i64 {
        loop {
            let draw = self.normal(mean, variance).floor();
            // ITK casts the result to unsigned int before its bounds test.
            // Apple ARM64 lowers this cast to FCVTMU. Its negative results
            // saturate to zero, which is accepted only at the low image face.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let ri = i64::from(draw as u32);
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let ri = draw as i64;
            if ri >= lo && ri <= hi {
                return ri;
            }
        }
    }
}

#[inline]
pub(super) fn sampling_interval(
    position: i64,
    size: i64,
    patch_radius: i64,
    sample_radius: i64,
) -> (i64, i64) {
    let constrained_lo = position.min(patch_radius);
    let constrained_hi = position.max(size - patch_radius - 1);
    (
        constrained_lo.max(position.saturating_sub(sample_radius).max(0)),
        constrained_hi.min(position.saturating_add(sample_radius)),
    )
}

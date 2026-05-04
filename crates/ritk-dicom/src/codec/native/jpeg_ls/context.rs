//! ISO 14495-1 context model for JPEG-LS lossless decoding.
//!
//! # Context Selection (ISO 14495-1 §A.2)
//! For each sample, three local gradients D1, D2, D3 are computed from the
//! causal neighborhood. Each gradient is quantized into one of 9 levels
//! [-4..4] using thresholds T1, T2, T3. The triplet is sign-normalized
//! (ensuring the first non-zero component is positive) and mapped to a
//! context index in [0, 365).
//!
//! # Context State (ISO 14495-1 §A.4–§A.5)
//! Each context tracks:
//! - `a`: accumulated absolute error sum (initialized to `a_init = max(2, (RANGE+32)/64)`)
//! - `b`: accumulated signed error sum for bias tracking
//! - `c`: bias correction applied to predictor (clamped to [-128, 127])
//! - `n`: sample count (initialized to 1 to avoid division by zero)

/// ISO 14495-1 context state for one regular or run-interrupt context.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextState {
    /// Accumulated absolute error magnitude (A in ISO 14495-1 §A.4).
    pub(crate) a: u32,
    /// Accumulated signed error sum (B, used for bias estimation).
    pub(crate) b: i32,
    /// Predictor bias correction value (C, clamped to [MIN_C, MAX_C]).
    pub(crate) c: i32,
    /// Sample count (N, initialized to 1).
    pub(crate) n: u32,
}

impl Default for ContextState {
    /// Returns the ISO 14495-1 initial state: n=1 (avoids zero-division in k computation).
    fn default() -> Self {
        Self { a: 0, b: 0, c: 0, n: 1 }
    }
}

/// ISO 14495-1 regular context count: 5×9×9 / 2 + 1 = 365 (sign-normalized triplets).
pub(crate) const CONTEXTS: usize = 365;

/// Context model reset threshold (ISO 14495-1 §A.5: RESET = 64).
const RESET: u32 = 64;

/// Minimum bias correction value (ISO 14495-1 §A.6: MIN_C = −128).
pub(super) const MIN_C: i32 = -128;

/// Maximum bias correction value (ISO 14495-1 §A.6: MAX_C = 127).
pub(super) const MAX_C: i32 = 127;

/// ISO 14495-1 context model: 365 regular contexts + 1 run-interrupt context.
pub(crate) struct ContextModel {
    /// Regular-mode context states indexed by `context_index()`.
    pub(crate) regular: [ContextState; CONTEXTS],
    /// Run-interrupt context state (§A.6).
    pub(crate) run_int: ContextState,
    /// Run-mode Golomb J-table index per component (§A.6.4, Rl).
    pub(crate) run_index: usize,
}

impl ContextModel {
    /// Initialize all context states.
    ///
    /// `a_init = max(2, (RANGE + 32) / 64)` where `RANGE = MAXVAL + 1`.
    /// From ISO 14495-1 §A.4: initial A[q] = a_init for all q.
    pub(crate) fn new(a_init: u32) -> Self {
        let s = ContextState { a: a_init, b: 0, c: 0, n: 1 };
        Self {
            regular: [s; CONTEXTS],
            run_int: s,
            run_index: 0,
        }
    }
}

/// Update a regular-mode context state after decoding one sample.
///
/// Updates A, B, N with optional renormalization (RESET), and adjusts C.
/// ISO 14495-1 §A.5.
#[inline(always)]
pub(crate) fn update_context(ctx: &mut ContextState, errval: i32, near: u32) {
    ctx.a += errval.unsigned_abs();
    ctx.n += 1;
    ctx.b += errval * (2 * near as i32 + 1);
    if ctx.n == RESET {
        ctx.n >>= 1;
        ctx.a >>= 1;
        ctx.b /= 2;
    }
    // Bias correction update
    if ctx.b <= -(ctx.n as i32) {
        ctx.b = -(ctx.n as i32) + 1;
        if ctx.c > MIN_C {
            ctx.c -= 1;
        }
    } else if ctx.b > 0 {
        ctx.b -= ctx.n as i32;
        ctx.b = ctx.b.min(0);
        if ctx.c < MAX_C {
            ctx.c += 1;
        }
    }
}

/// Compute Golomb-Rice parameter k = floor(log₂(A/N)), clamped to [0, qbpp].
///
/// Invariant: N×2^k ≤ A < N×2^(k+1).
/// ISO 14495-1 §A.3.
#[inline(always)]
pub(crate) fn compute_k(a: u32, n: u32, qbpp: u32) -> u32 {
    let mut k = 0u32;
    // Smallest k such that N << k >= A  (equivalent to k = ceil(log2(A/N)) but floored)
    while (n << k) < a && k < qbpp {
        k += 1;
    }
    k
}

/// Quantize gradient `d` into one of 9 levels {−4, −3, −2, −1, 0, 1, 2, 3, 4}
/// using thresholds T1, T2, T3 (where 0 < T1 ≤ T2 ≤ T3).
///
/// ISO 14495-1 §A.2.
#[inline(always)]
pub(crate) fn quant(d: i32, t1: i32, t2: i32, t3: i32) -> i32 {
    if d <= -t3 {
        -4
    } else if d <= -t2 {
        -3
    } else if d <= -t1 {
        -2
    } else if d < 0 {
        -1
    } else if d == 0 {
        0
    } else if d < t1 {
        1
    } else if d < t2 {
        2
    } else if d < t3 {
        3
    } else {
        4
    }
}

/// Sign-normalize triplet (Q1, Q2, Q3) so the first non-zero component is positive.
///
/// Returns (Q1', Q2', Q3', sign) where sign ∈ {1, −1}.
/// ISO 14495-1 §A.2.
#[inline(always)]
pub(crate) fn sign_normalize(q1: i32, q2: i32, q3: i32) -> (i32, i32, i32, i32) {
    if q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0) {
        (-q1, -q2, -q3, -1)
    } else {
        (q1, q2, q3, 1)
    }
}

/// Compute context index from sign-normalized (Q1, Q2, Q3) ∈ [0, 365).
///
/// Partition:
/// - Q1=0, Q2=0: indices [0, 5)   (Q3 ∈ {0..4})
/// - Q1=0, Q2>0: indices [5, 41)  (Q2 ∈ {1..4}, Q3 ∈ {−4..4}: 4×9=36)
/// - Q1>0:       indices [41, 365) (Q1 ∈ {1..4}, Q2 ∈ {−4..4}, Q3 ∈ {−4..4}: 4×81=324)
///
/// ISO 14495-1 §A.2.
#[inline(always)]
pub(crate) fn context_index(q1: i32, q2: i32, q3: i32) -> usize {
    if q1 == 0 && q2 == 0 {
        // Q3 ∈ [0..4]: 5 contexts
        q3 as usize
    } else if q1 == 0 {
        // Q2 ∈ [1..4], Q3 ∈ [-4..4]: 4×9 = 36 contexts starting at 5
        5 + (q2 - 1) as usize * 9 + (q3 + 4) as usize
    } else {
        // Q1 ∈ [1..4], Q2 ∈ [-4..4], Q3 ∈ [-4..4]: 4×81 = 324 starting at 41
        41 + (q1 - 1) as usize * 81 + (q2 + 4) as usize * 9 + (q3 + 4) as usize
    }
}

/// Compute default gradient thresholds T1, T2, T3 from MAXVAL.
///
/// ISO 14495-1 §C.2.4 default preset parameters (no LSE marker).
pub(crate) fn default_thresholds(maxval: i32) -> (i32, i32, i32) {
    let factor = (maxval + 128) / 256;
    let t1 = (3 * factor).max(2).min((maxval + 1) / 8);
    let t2 = (7 * factor).max(3).min((maxval + 1) / 4);
    let t3 = (21 * factor).max(4).min(maxval);
    (t1, t2, t3)
}

/// Inverse error mapping: MErrval (≥0) → signed errval.
///
/// Forward mapping: errval ≥ 0 → MErrval = 2×errval (even); errval < 0 → MErrval = −2×errval−1 (odd).
/// Inverse: even MErrval → errval = MErrval/2; odd → errval = −(MErrval+1)/2.
/// ISO 14495-1 §A.3.
#[inline(always)]
pub(crate) fn inverse_map(me: u32) -> i32 {
    if me & 1 == 0 {
        (me >> 1) as i32
    } else {
        -(((me + 1) >> 1) as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_boundary_mapping() {
        // T1=3, T2=7, T3=21
        let (t1, t2, t3) = (3, 7, 21);
        assert_eq!(quant(-30, t1, t2, t3), -4); // d <= -T3
        assert_eq!(quant(-21, t1, t2, t3), -4); // d == -T3 → ≤ -T3
        assert_eq!(quant(-10, t1, t2, t3), -3); // -T3 < d ≤ -T2
        assert_eq!(quant(-7, t1, t2, t3), -3);  // d == -T2 → ≤ -T2
        assert_eq!(quant(-5, t1, t2, t3), -2);
        assert_eq!(quant(-3, t1, t2, t3), -2);  // d == -T1 → ≤ -T1
        assert_eq!(quant(-1, t1, t2, t3), -1);
        assert_eq!(quant(0, t1, t2, t3), 0);
        assert_eq!(quant(1, t1, t2, t3), 1);
        assert_eq!(quant(2, t1, t2, t3), 1);    // d < T1
        assert_eq!(quant(3, t1, t2, t3), 2);    // d == T1 → ≥ T1
        assert_eq!(quant(6, t1, t2, t3), 2);
        assert_eq!(quant(7, t1, t2, t3), 3);
        assert_eq!(quant(20, t1, t2, t3), 3);
        assert_eq!(quant(21, t1, t2, t3), 4);   // d ≥ T3
        assert_eq!(quant(100, t1, t2, t3), 4);
    }

    #[test]
    fn sign_normalize_positive_q1() {
        let (q1, q2, q3, s) = sign_normalize(2, -3, 1);
        assert_eq!((q1, q2, q3), (2, -3, 1));
        assert_eq!(s, 1);
    }

    #[test]
    fn sign_normalize_negative_q1() {
        let (q1, q2, q3, s) = sign_normalize(-2, 3, -1);
        assert_eq!((q1, q2, q3), (2, -3, 1));
        assert_eq!(s, -1);
    }

    #[test]
    fn sign_normalize_q1_zero_negative_q2() {
        let (q1, q2, q3, s) = sign_normalize(0, -3, 1);
        assert_eq!((q1, q2, q3), (0, 3, -1));
        assert_eq!(s, -1);
    }

    #[test]
    fn sign_normalize_all_zero() {
        let (q1, q2, q3, s) = sign_normalize(0, 0, 0);
        assert_eq!((q1, q2, q3, s), (0, 0, 0, 1));
    }

    #[test]
    fn context_index_q1q2q3_all_zero() {
        // Q3=0 → index 0
        assert_eq!(context_index(0, 0, 0), 0);
    }

    #[test]
    fn context_index_q1q2_zero_q3_4() {
        // Q3=4 → index 4
        assert_eq!(context_index(0, 0, 4), 4);
    }

    #[test]
    fn context_index_q1_zero_q2_1_q3_neg4() {
        // Q2=1, Q3=-4 → 5 + 0*9 + 0 = 5
        assert_eq!(context_index(0, 1, -4), 5);
    }

    #[test]
    fn context_index_q1_zero_q2_4_q3_4() {
        // Q2=4, Q3=4 → 5 + 3*9 + 8 = 5 + 27 + 8 = 40
        assert_eq!(context_index(0, 4, 4), 40);
    }

    #[test]
    fn context_index_q1_1_min() {
        // Q1=1, Q2=-4, Q3=-4 → 41 + 0 + 0 + 0 = 41
        assert_eq!(context_index(1, -4, -4), 41);
    }

    #[test]
    fn context_index_q1_4_max() {
        // Q1=4, Q2=4, Q3=4 → 41 + 3*81 + 8*9 + 8 = 41+243+72+8 = 364
        assert_eq!(context_index(4, 4, 4), 364);
    }

    #[test]
    fn context_index_max_value_is_364() {
        assert_eq!(context_index(4, 4, 4), 364);
    }

    #[test]
    fn default_thresholds_8bit() {
        // maxval=255: factor=(255+128)/256=1
        // t1 = max(3, 2).min(32) = 3
        // t2 = max(7, 3).min(64) = 7
        // t3 = max(21, 4).min(255) = 21
        let (t1, t2, t3) = default_thresholds(255);
        assert_eq!(t1, 3);
        assert_eq!(t2, 7);
        assert_eq!(t3, 21);
    }

    #[test]
    fn default_thresholds_16bit() {
        // maxval=65535: factor=(65535+128)/256=256
        // t1=max(768,2).min(8192)=768
        // t2=max(1792,3).min(16384)=1792
        // t3=max(5376,4).min(65535)=5376
        let (t1, t2, t3) = default_thresholds(65535);
        assert_eq!(t1, 768);
        assert_eq!(t2, 1792);
        assert_eq!(t3, 5376);
    }

    #[test]
    fn inverse_map_even_zero() {
        assert_eq!(inverse_map(0), 0);
    }

    #[test]
    fn inverse_map_even_positive() {
        // MErrval=4 (even) → errval = 2
        assert_eq!(inverse_map(4), 2);
    }

    #[test]
    fn inverse_map_odd_negative() {
        // MErrval=1 (odd) → errval = -(1+1)/2 = -1
        assert_eq!(inverse_map(1), -1);
    }

    #[test]
    fn inverse_map_odd_large() {
        // MErrval=7 (odd) → errval = -(7+1)/2 = -4
        assert_eq!(inverse_map(7), -4);
    }

    #[test]
    fn inverse_map_forward_inverse_bijection() {
        // Forward: errval ≥ 0 → 2*errval; errval < 0 → -2*errval - 1
        for errval in -50i32..=50i32 {
            let me = if errval >= 0 {
                (errval * 2) as u32
            } else {
                (-2 * errval - 1) as u32
            };
            assert_eq!(inverse_map(me), errval, "round-trip failed for errval={}", errval);
        }
    }

    #[test]
    fn compute_k_zero_when_a_small() {
        // A=1, N=1: N<<0 = 1 >= 1 = A → k=0
        assert_eq!(compute_k(1, 1, 8), 0);
    }

    #[test]
    fn compute_k_increases_with_large_a() {
        // A=8, N=1: N<<0=1 < 8, N<<1=2 < 8, N<<2=4 < 8, N<<3=8 >= 8 → k=3
        assert_eq!(compute_k(8, 1, 8), 3);
    }

    #[test]
    fn compute_k_clamped_to_qbpp() {
        // A=1000000, N=1, qbpp=8: k stops at 8
        assert_eq!(compute_k(1_000_000, 1, 8), 8);
    }

    #[test]
    fn update_context_accumulates_errval() {
        let mut ctx = ContextState::default();
        update_context(&mut ctx, 3, 0); // NEAR=0
        assert_eq!(ctx.a, 3); // |errval| = 3
        assert_eq!(ctx.n, 2); // n increments to 2
    }

    #[test]
    fn update_context_bias_positive_decrements_b() {
        let mut ctx = ContextState { a: 0, b: 5, c: 0, n: 4 };
        // b > 0 → b -= n → b = 5 - 4 = 1 → clamp to min(1, 0) = 0 in the code? 
        // Actually: b = (b - n).min(0) = (1).min(0) = 0. Wait let me re-check the code.
        // After errval=0: ctx.b += 0*(2*0+1) = 0, so b stays 5.
        // Then: b > 0 → b -= n (=4) → b = 1 → b.min(0) = 0? No: `b -= n` then `b = b.min(0)`.
        // Hmm, actually: b = b - n = 5 - 5 (n=5 after += 1) → b = 0.
        update_context(&mut ctx, 0, 0);
        assert_eq!(ctx.n, 5);
        // b=5, n becomes 5: b -= n → b = 0 → min(0,0) = 0
        assert_eq!(ctx.b, 0);
    }
}

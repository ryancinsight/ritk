//! ISO 14495-1 context model for JPEG-LS lossless decoding.
//!
//! # Context Selection (ISO 14495-1 Â§A.2)
//! For each sample, three local gradients D1, D2, D3 are computed from the
//! causal neighborhood. Each gradient is quantized into one of 9 levels
//! [-4..4] using thresholds T1, T2, T3. The triplet is sign-normalized
//! (ensuring the first non-zero component is positive) and mapped to a
//! context index in [0, 365).
//!
//! # Context State (ISO 14495-1 Â§A.4â€“Â§A.5)
//! Each context tracks:
//! - `a`: accumulated absolute error sum (initialized to `a_init = max(2, (RANGE+32)/64)`)
//! - `b`: accumulated signed error sum for bias tracking
//! - `c`: bias correction applied to predictor (clamped to [-128, 127])
//! - `n`: sample count (initialized to 1 to avoid division by zero)

/// ISO 14495-1 context state for one regular or run-interrupt context.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextState {
    /// Accumulated absolute error magnitude (A in ISO 14495-1 Â§A.4).
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
        Self {
            a: 0,
            b: 0,
            c: 0,
            n: 1,
        }
    }
}

/// ISO 14495-1 regular context count: 5Ã—9Ã—9 / 2 + 1 = 365 (sign-normalized triplets).
pub(crate) const CONTEXTS: usize = 365;

/// Context model reset threshold (ISO 14495-1 Â§A.5: RESET = 64).
const RESET: u32 = 64;

/// Minimum bias correction value (ISO 14495-1 Â§A.6: MIN_C = âˆ’128).
pub(super) const MIN_C: i32 = -128;

/// Maximum bias correction value (ISO 14495-1 Â§A.6: MAX_C = 127).
pub(super) const MAX_C: i32 = 127;

/// ISO 14495-1 context model: 365 regular contexts + 1 run-interrupt context.
pub(crate) struct ContextModel {
    /// Regular-mode context states indexed by `context_index()`.
    pub(crate) regular: [ContextState; CONTEXTS],
    /// Run-interrupt context for `RItype = 0` (Â§A.6).
    pub(crate) run_int_diff: RunInterruptionContext,
    /// Run-interrupt context for `RItype = 1` (Â§A.6).
    pub(crate) run_int_same: RunInterruptionContext,
    /// Run-mode Golomb J-table index per component (Â§A.6.4, Rl).
    pub(crate) run_index: usize,
}

impl ContextModel {
    /// Initialize all context states.
    ///
    /// `a_init = max(2, (RANGE + 32) / 64)` where `RANGE = MAXVAL + 1`.
    /// From ISO 14495-1 Â§A.4: initial `A[q]` = a_init for all q.
    pub(crate) fn new(a_init: u32) -> Self {
        let s = ContextState {
            a: a_init,
            b: 0,
            c: 0,
            n: 1,
        };
        Self {
            regular: [s; CONTEXTS],
            run_int_diff: RunInterruptionContext::new(0, a_init),
            run_int_same: RunInterruptionContext::new(1, a_init),
            run_index: 0,
        }
    }
}

/// ISO 14495-1 run-interruption context state for indices 365 and 366.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunInterruptionContext {
    run_interruption_type: u32,
    a: u32,
    n: u32,
    nn: u32,
}

impl RunInterruptionContext {
    pub(crate) const fn new(run_interruption_type: u32, a_init: u32) -> Self {
        Self {
            run_interruption_type,
            a: a_init,
            n: 1,
            nn: 0,
        }
    }

    #[inline(always)]
    pub(crate) const fn run_interruption_type(self) -> u32 {
        self.run_interruption_type
    }

    #[inline(always)]
    pub(crate) fn compute_k(self, qbpp: u32) -> u32 {
        let temp = self.a + (self.n >> 1) * self.run_interruption_type;
        let mut n_test = self.n;
        let mut k = 0;
        while n_test < temp && k < qbpp {
            n_test <<= 1;
            k += 1;
        }
        k
    }

    /// Sign-mapping condition used by both the decoder's `compute_error_value`
    /// and the encoder's forward mapping (ISO 14495-1 Â§A.7.2): when `true`,
    /// a negative error value maps to an odd `temp`.
    #[inline(always)]
    pub(crate) fn negative_maps_to_odd(self, k: u32) -> bool {
        k != 0 || 2 * self.nn >= self.n
    }

    #[inline(always)]
    pub(crate) fn compute_error_value(self, temp: u32, k: u32) -> i32 {
        let map = temp & 1 != 0;
        let error_value_abs = ((temp + u32::from(map)) / 2) as i32;
        if self.negative_maps_to_odd(k) == map {
            -error_value_abs
        } else {
            error_value_abs
        }
    }

    #[inline(always)]
    pub(crate) fn update(&mut self, error_value: i32, mapped_error_value: u32) {
        if error_value < 0 {
            self.nn += 1;
        }
        self.a += (mapped_error_value + 1 - self.run_interruption_type) >> 1;
        if self.n == RESET {
            self.a >>= 1;
            self.n >>= 1;
            self.nn >>= 1;
        }
        self.n += 1;
    }
}

/// Update a regular-mode context state after decoding one sample.
///
/// Updates A, B, N with optional renormalization (RESET), and adjusts C.
/// ISO 14495-1 Â§A.5.
#[inline(always)]
pub(crate) fn update_context(ctx: &mut ContextState, errval: i32, near: u32) {
    ctx.a += errval.unsigned_abs();
    ctx.b += errval * (2 * near as i32 + 1);
    if ctx.n == RESET {
        ctx.a >>= 1;
        ctx.b /= 2;
        ctx.n >>= 1;
    }
    ctx.n += 1;

    // Bias correction update
    if ctx.b + ctx.n as i32 <= 0 {
        ctx.b += ctx.n as i32;
        if ctx.b <= -(ctx.n as i32) {
            ctx.b = -(ctx.n as i32) + 1;
        }
        if ctx.c > MIN_C {
            ctx.c -= 1;
        }
    } else if ctx.b > 0 {
        ctx.b -= ctx.n as i32;
        if ctx.b > 0 {
            ctx.b = 0;
        }
        if ctx.c < MAX_C {
            ctx.c += 1;
        }
    }
}

#[inline(always)]
pub(crate) fn error_correction(ctx: &ContextState, k_or_near: u32) -> i32 {
    if k_or_near != 0 {
        0
    } else if 2 * ctx.b + ctx.n as i32 - 1 < 0 {
        -1
    } else {
        0
    }
}

/// Compute Golomb-Rice parameter k = floor(logâ‚‚(A/N)), clamped to [0, qbpp].
///
/// Invariant: NÃ—2^k â‰¤ A < NÃ—2^(k+1).
/// ISO 14495-1 Â§A.3.
#[inline(always)]
pub(crate) fn compute_k(a: u32, n: u32, qbpp: u32) -> u32 {
    let mut k = 0u32;
    // Smallest k such that N << k >= A (equivalent to k = ceil(log2(A/N)) but floored)
    while (n << k) < a && k < qbpp {
        k += 1;
    }
    k
}

/// Quantize gradient `d` into one of 9 levels {âˆ’4, âˆ’3, âˆ’2, âˆ’1, 0, 1, 2, 3, 4}
/// using thresholds T1, T2, T3 (where NEAR < T1 â‰¤ T2 â‰¤ T3) and the NEAR
/// dead-zone (`|d| â‰¤ NEAR â†’ 0`; NEAR = 0 for lossless).
///
/// ISO 14495-1 Â§A.3.3 (code segment A.4).
#[inline(always)]
pub(crate) fn quant(d: i32, t1: i32, t2: i32, t3: i32, near: i32) -> i32 {
    if d <= -t3 {
        -4
    } else if d <= -t2 {
        -3
    } else if d <= -t1 {
        -2
    } else if d < -near {
        -1
    } else if d <= near {
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
/// Returns (Q1', Q2', Q3', sign) where sign âˆˆ {1, âˆ’1}.
/// ISO 14495-1 Â§A.2.
#[inline(always)]
pub(crate) fn sign_normalize(q1: i32, q2: i32, q3: i32) -> (i32, i32, i32, i32) {
    if q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0) {
        (-q1, -q2, -q3, -1)
    } else {
        (q1, q2, q3, 1)
    }
}

/// Compute context index from sign-normalized (Q1, Q2, Q3) âˆˆ [0, 365).
///
/// Partition:
/// - Q1=0, Q2=0: indices [0, 5) (Q3 âˆˆ {0..4})
/// - Q1=0, Q2>0: indices [5, 41) (Q2 âˆˆ {1..4}, Q3 âˆˆ {âˆ’4..4}: 4Ã—9=36)
/// - Q1>0: indices [41, 365) (Q1 âˆˆ {1..4}, Q2 âˆˆ {âˆ’4..4}, Q3 âˆˆ {âˆ’4..4}: 4Ã—81=324)
///
/// ISO 14495-1 Â§A.2.
#[inline(always)]
pub(crate) fn context_index(q1: i32, q2: i32, q3: i32) -> usize {
    if q1 == 0 && q2 == 0 {
        // Q3 âˆˆ [0..4]: 5 contexts
        q3 as usize
    } else if q1 == 0 {
        // Q2 âˆˆ [1..4], Q3 âˆˆ [-4..4]: 4Ã—9 = 36 contexts starting at 5
        5 + (q2 - 1) as usize * 9 + (q3 + 4) as usize
    } else {
        // Q1 âˆˆ [1..4], Q2 âˆˆ [-4..4], Q3 âˆˆ [-4..4]: 4Ã—81 = 324 starting at 41
        41 + (q1 - 1) as usize * 81 + (q2 + 4) as usize * 9 + (q3 + 4) as usize
    }
}

/// Compute default gradient thresholds T1, T2, T3 from MAXVAL and NEAR.
///
/// ISO 14495-1 Â§C.2.4.1.1.1 default preset parameters (no LSE marker):
/// `FACTOR = (min(MAXVAL, 4095) + 128) / 256`, then
/// `T1 = FACTORÂ·(BASIC_T1 âˆ’ 2) + 2 + 3Â·NEAR`,
/// `T2 = FACTORÂ·(BASIC_T2 âˆ’ 3) + 3 + 5Â·NEAR`,
/// `T3 = FACTORÂ·(BASIC_T3 âˆ’ 4) + 4 + 7Â·NEAR`,
/// with `BASIC = (3, 7, 21)`, each clamped to `[NEAR + 1 / T1 / T2, MAXVAL]`.
/// For `MAXVAL < 128` the FACTOR term scales the basic values down instead;
/// DICOM data is â‰¥ 8-bit so the high branch is the production path.
pub(crate) fn default_thresholds(maxval: i32, near: i32) -> (i32, i32, i32) {
    if maxval >= 128 {
        let factor = (maxval.min(4095) + 128) / 256;
        let t1 = (factor + 2 + 3 * near).clamp(near + 1, maxval);
        let t2 = (4 * factor + 3 + 5 * near).clamp(t1, maxval);
        let t3 = (17 * factor + 4 + 7 * near).clamp(t2, maxval);
        (t1, t2, t3)
    } else {
        // Low-precision branch (ISO 14495-1 Â§C.2.4.1.1.1, MAXVAL < 128):
        // FACTOR = 256 / (MAXVAL + 1) divides the basic thresholds.
        let factor = 256 / (maxval + 1);
        let t1 = (3 / factor + 3 * near).max(2).clamp(near + 1, maxval);
        let t2 = (7 / factor + 5 * near).max(3).clamp(t1, maxval);
        let t3 = (21 / factor + 7 * near).max(4).clamp(t2, maxval);
        (t1, t2, t3)
    }
}

/// Coding parameters derived from the sample precision and the NEAR bound
/// (ISO 14495-1 Â§A.2.1) â€” shared by the encoder and the scan decoder.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CodingParams {
    /// Maximum sample value `2^bpp âˆ’ 1`.
    pub(crate) maxval: i32,
    /// Range of quantized prediction errors; `2^bpp` for lossless.
    pub(crate) range: i32,
    /// `âŒˆlogâ‚‚ RANGEâŒ‰`; equals `bpp` for lossless.
    pub(crate) qbpp: u32,
    /// Golomb code length limit `2Â·(bpp + max(8, bpp))`.
    pub(crate) limit: u32,
    /// Context initialisation value `max(2, (RANGE + 32)/64)`.
    pub(crate) a_init: u32,
    /// NEAR bound (0 = lossless).
    pub(crate) near: i32,
}

impl CodingParams {
    pub(crate) fn new(bpp: u32, near: i32) -> Self {
        let maxval = (1i32 << bpp) - 1;
        let range = (maxval + 2 * near) / (2 * near + 1) + 1;
        let qbpp = u32::BITS - ((range - 1) as u32).leading_zeros();
        let limit = 2 * (bpp + bpp.max(8));
        let a_init = ((range as u32 + 32) >> 6).max(2);
        Self {
            maxval,
            range,
            qbpp,
            limit,
            a_init,
            near,
        }
    }
}

/// Reduce a quantized prediction error to the canonical modulo-RANGE
/// representative in `[âˆ’RANGE/2, (RANGE+1)/2)` (ISO 14495-1 Â§A.4.5).
#[inline(always)]
pub(crate) fn modulo_reduce(mut errval: i32, range: i32) -> i32 {
    if errval < 0 {
        errval += range;
    }
    if errval >= (range + 1) / 2 {
        errval -= range;
    }
    errval
}

/// Quantize a raw prediction error by the NEAR dead-zone (ISO 14495-1 Â§A.4.4):
/// `q = sign(e) Â· (|e| + NEAR) / (2Â·NEAR + 1)`. Identity for NEAR = 0.
#[inline(always)]
pub(crate) fn quantize_error(e: i32, near: i32) -> i32 {
    if e > 0 {
        (e + near) / (2 * near + 1)
    } else {
        -((near - e) / (2 * near + 1))
    }
}

/// Dequantize and reconstruct a sample (ISO 14495-1 Â§A.4.4 / Â§A.8.2): wrap
/// into `[âˆ’NEAR, MAXVAL + NEAR]` by RANGEÂ·(2Â·NEAR+1) steps, then clamp to
/// `[0, MAXVAL]`. For NEAR = 0 this is the modulo-RANGE reconstruction.
#[inline(always)]
pub(crate) fn reconstruct(pred: i32, errval_q: i32, p: &CodingParams) -> i32 {
    let mut rx = pred + errval_q * (2 * p.near + 1);
    if rx < -p.near {
        rx += p.range * (2 * p.near + 1);
    } else if rx > p.maxval + p.near {
        rx -= p.range * (2 * p.near + 1);
    }
    rx.clamp(0, p.maxval)
}

/// Inverse error mapping: MErrval (â‰¥0) â†’ signed errval.
///
/// Forward mapping: errval â‰¥ 0 â†’ MErrval = 2Ã—errval (even); errval < 0 â†’ MErrval = âˆ’2Ã—errvalâˆ’1 (odd).
/// Inverse: even MErrval â†’ errval = MErrval/2; odd â†’ errval = âˆ’(MErrval+1)/2.
/// ISO 14495-1 Â§A.3.
#[inline(always)]
pub(crate) fn inverse_map(me: u32) -> i32 {
    if me & 1 == 0 {
        (me >> 1) as i32
    } else {
        -(((me + 1) >> 1) as i32)
    }
}

#[cfg(test)]
mod tests;

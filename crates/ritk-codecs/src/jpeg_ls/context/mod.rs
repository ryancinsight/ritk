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
        Self {
            a: 0,
            b: 0,
            c: 0,
            n: 1,
        }
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
    /// Run-interrupt context for `RItype = 0` (§A.6).
    pub(crate) run_int_diff: RunInterruptionContext,
    /// Run-interrupt context for `RItype = 1` (§A.6).
    pub(crate) run_int_same: RunInterruptionContext,
    /// Run-mode Golomb J-table index per component (§A.6.4, Rl).
    pub(crate) run_index: usize,
}

impl ContextModel {
    /// Initialize all context states.
    ///
    /// `a_init = max(2, (RANGE + 32) / 64)` where `RANGE = MAXVAL + 1`.
    /// From ISO 14495-1 §A.4: initial A[q] = a_init for all q.
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

    #[inline(always)]
    pub(crate) fn compute_error_value(self, temp: u32, k: u32) -> i32 {
        let map = temp & 1 != 0;
        let error_value_abs = ((temp + u32::from(map)) / 2) as i32;
        if (k != 0 || 2 * self.nn >= self.n) == map {
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
/// ISO 14495-1 §A.5.
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

/// Compute Golomb-Rice parameter k = floor(log₂(A/N)), clamped to [0, qbpp].
///
/// Invariant: N×2^k ≤ A < N×2^(k+1).
/// ISO 14495-1 §A.3.
#[inline(always)]
pub(crate) fn compute_k(a: u32, n: u32, qbpp: u32) -> u32 {
    let mut k = 0u32;
    // Smallest k such that N << k >= A (equivalent to k = ceil(log2(A/N)) but floored)
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
/// - Q1=0, Q2=0: indices [0, 5) (Q3 ∈ {0..4})
/// - Q1=0, Q2>0: indices [5, 41) (Q2 ∈ {1..4}, Q3 ∈ {−4..4}: 4×9=36)
/// - Q1>0: indices [41, 365) (Q1 ∈ {1..4}, Q2 ∈ {−4..4}, Q3 ∈ {−4..4}: 4×81=324)
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
mod tests;

//! Forward and inverse discrete wavelet transform — 9/7 irreversible (lossy).
//!
//! The Daubechies 9/7 floating-point wavelet (ISO 15444-1 §F.3.8.2, Table F.4)
//! is the irreversible transform used by lossy JPEG 2000 (transfer syntax
//! 1.2.840.10008.1.2.4.91).  Unlike the integer 5/3 lifting in [`super::wavelet`],
//! the 9/7 lifting uses real-valued predict/update steps followed by a pair of
//! scaling steps, so the transform is not bit-exact and the coefficients are
//! quantised (see [`super::quantization`]) before entropy coding.
//!
//! The 1-D lifting on an interleaved buffer (even index = low/`s`, odd = high/`d`)
//! applies, for analysis:
//! 1. predict  `d ← d + α(s_l + s_r)`
//! 2. update   `s ← s + β(d_l + d_r)`
//! 3. predict  `d ← d + γ(s_l + s_r)`
//! 4. update   `s ← s + δ(d_l + d_r)`
//! 5. scale    `s ← s/K`,  `d ← d·K`
//!
//! Synthesis applies the exact algebraic inverse in reverse order.  Boundaries
//! use whole-sample-symmetric (WSS) reflection, matching the 5/3 path and the
//! ISO periodic-symmetric extension.

use anyhow::{bail, Result};

use super::wavelet::ceil_div_pow2;

// The lifting coefficients below are the exact ISO 15444-1 Table F.4 reference
// values; they carry more digits than f32 can represent, but the compiler rounds
// each to the same deterministic f32 constant used by every conformant codec, so
// the documented precision is intentional.
#[allow(clippy::excessive_precision)]
mod coeffs {
    pub const ALPHA: f32 = -1.586_134_342_059_924;
    pub const BETA: f32 = -0.052_980_118_572_961;
    pub const GAMMA: f32 = 0.882_911_075_530_934;
    pub const DELTA: f32 = 0.443_506_852_043_971;
    pub const K: f32 = 1.230_174_104_914_001;
}
use coeffs::{ALPHA, BETA, DELTA, GAMMA, K};

/// Whole-sample-symmetric mirror of a (possibly out-of-range) index into
/// `[0, n-1]`.  `f[-1] = f[1]`, `f[n] = f[n-2]`, extended periodically.
#[inline]
fn reflect(i: isize, n: usize) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    j as usize
}

/// Apply `num_levels` of the inverse 9/7 DWT to a row-major Mallat-layout buffer
/// in-place.  After the call, `samples[y * width + x]` holds the reconstructed
/// (still DC-shifted, pre-rounding) coefficient for pixel `(x, y)`.
///
/// # Errors
/// Returns an error when `samples.len() != width × height`.
pub fn inverse_dwt_9_7(
    samples: &mut [f32],
    width: usize,
    height: usize,
    num_levels: u8,
) -> Result<()> {
    if samples.len() != width * height {
        bail!(
            "inverse_dwt_9_7: samples.len()={} != width({})×height({})",
            samples.len(),
            width,
            height
        );
    }
    for lvl in (1..=u32::from(num_levels)).rev() {
        let cur_w = ceil_div_pow2(width, lvl - 1);
        let cur_h = ceil_div_pow2(height, lvl - 1);
        synth_2d_9_7(samples, width, cur_w, cur_h);
    }
    Ok(())
}

/// Apply `num_levels` of the forward 9/7 DWT to a row-major buffer in-place,
/// producing the Mallat coefficient layout consumed by [`inverse_dwt_9_7`].
///
/// # Errors
/// Returns an error when `samples.len() != width × height`.
pub fn forward_dwt_9_7(
    samples: &mut [f32],
    width: usize,
    height: usize,
    num_levels: u8,
) -> Result<()> {
    if samples.len() != width * height {
        bail!(
            "forward_dwt_9_7: samples.len()={} != width({})×height({})",
            samples.len(),
            width,
            height
        );
    }
    for lvl in 1..=u32::from(num_levels) {
        let cur_w = ceil_div_pow2(width, lvl - 1);
        let cur_h = ceil_div_pow2(height, lvl - 1);
        analyse_2d_9_7(samples, width, cur_w, cur_h);
    }
    Ok(())
}

// ── 2-D per-level passes ──────────────────────────────────────────────────────

/// One synthesis level on the `[0..cur_h, 0..cur_w]` rectangle: rows first
/// (interleave [low|high] halves → inverse 1-D), then columns — the exact
/// inverse of [`analyse_2d_9_7`].
fn synth_2d_9_7(samples: &mut [f32], stride: usize, cur_w: usize, cur_h: usize) {
    let sn_x = cur_w.div_ceil(2);
    let sn_y = cur_h.div_ceil(2);
    let mut buf = vec![0f32; cur_w.max(cur_h)];

    // Horizontal pass.
    for y in 0..cur_h {
        let row = &samples[y * stride..y * stride + cur_w];
        interleave(row, sn_x, &mut buf[..cur_w]);
        idwt_1d_9_7(&mut buf[..cur_w]);
        samples[y * stride..y * stride + cur_w].copy_from_slice(&buf[..cur_w]);
    }

    // Vertical pass.
    let mut col = vec![0f32; cur_h];
    for x in 0..cur_w {
        for y in 0..cur_h {
            col[y] = samples[y * stride + x];
        }
        interleave(&col, sn_y, &mut buf[..cur_h]);
        idwt_1d_9_7(&mut buf[..cur_h]);
        for y in 0..cur_h {
            samples[y * stride + x] = buf[y];
        }
    }
}

/// One analysis level on the `[0..cur_h, 0..cur_w]` rectangle: columns first,
/// then rows, de-interleaving the lifted output into [low|high] halves.
fn analyse_2d_9_7(samples: &mut [f32], stride: usize, cur_w: usize, cur_h: usize) {
    let sn_x = cur_w.div_ceil(2);
    let sn_y = cur_h.div_ceil(2);
    let mut buf = vec![0f32; cur_w.max(cur_h)];

    // Vertical pass.
    let mut col = vec![0f32; cur_h];
    for x in 0..cur_w {
        for y in 0..cur_h {
            col[y] = samples[y * stride + x];
        }
        fdwt_1d_9_7(&mut col);
        deinterleave(&col, sn_y, &mut buf[..cur_h]);
        for y in 0..cur_h {
            samples[y * stride + x] = buf[y];
        }
    }

    // Horizontal pass.
    let mut row = vec![0f32; cur_w];
    for y in 0..cur_h {
        row.copy_from_slice(&samples[y * stride..y * stride + cur_w]);
        fdwt_1d_9_7(&mut row);
        deinterleave(&row, sn_x, &mut buf[..cur_w]);
        samples[y * stride..y * stride + cur_w].copy_from_slice(&buf[..cur_w]);
    }
}

/// `[low half | high half] → interleaved (even = low, odd = high)`.
#[inline]
fn interleave(halves: &[f32], sn: usize, out: &mut [f32]) {
    for (i, &v) in halves[..sn].iter().enumerate() {
        out[2 * i] = v;
    }
    for (i, &v) in halves[sn..].iter().enumerate() {
        out[2 * i + 1] = v;
    }
}

/// `interleaved (even = low, odd = high) → [low half | high half]`.
#[inline]
fn deinterleave(inter: &[f32], sn: usize, out: &mut [f32]) {
    for i in 0..sn {
        out[i] = inter[2 * i];
    }
    for i in 0..inter.len() - sn {
        out[sn + i] = inter[2 * i + 1];
    }
}

// ── 1-D lifting kernels ───────────────────────────────────────────────────────

/// 1-D forward 9/7 lifting on an interleaved buffer (even = low, odd = high).
fn fdwt_1d_9_7(buf: &mut [f32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let nb = |buf: &[f32], k: usize| -> f32 {
        buf[reflect(k as isize - 1, n)] + buf[reflect(k as isize + 1, n)]
    };
    // 1. predict α (odd), 2. update β (even), 3. predict γ (odd), 4. update δ (even).
    for k in (1..n).step_by(2) {
        buf[k] += ALPHA * nb(buf, k);
    }
    for k in (0..n).step_by(2) {
        buf[k] += BETA * nb(buf, k);
    }
    for k in (1..n).step_by(2) {
        buf[k] += GAMMA * nb(buf, k);
    }
    for k in (0..n).step_by(2) {
        buf[k] += DELTA * nb(buf, k);
    }
    // 5. scale: low ← s/K, high ← d·K.
    for k in (0..n).step_by(2) {
        buf[k] /= K;
    }
    for k in (1..n).step_by(2) {
        buf[k] *= K;
    }
}

/// 1-D inverse 9/7 lifting — exact algebraic inverse of [`fdwt_1d_9_7`].
fn idwt_1d_9_7(buf: &mut [f32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let nb = |buf: &[f32], k: usize| -> f32 {
        buf[reflect(k as isize - 1, n)] + buf[reflect(k as isize + 1, n)]
    };
    // Undo scale, then undo δ, γ, β, α in reverse order.
    for k in (0..n).step_by(2) {
        buf[k] *= K;
    }
    for k in (1..n).step_by(2) {
        buf[k] /= K;
    }
    for k in (0..n).step_by(2) {
        buf[k] -= DELTA * nb(buf, k);
    }
    for k in (1..n).step_by(2) {
        buf[k] -= GAMMA * nb(buf, k);
    }
    for k in (0..n).step_by(2) {
        buf[k] -= BETA * nb(buf, k);
    }
    for k in (1..n).step_by(2) {
        buf[k] -= ALPHA * nb(buf, k);
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn reflect_is_whole_sample_symmetric() {
        // n=5: indices mirror about 0 and 4 -> 0 1 2 3 4 3 2 1 0 1 ...
        let expect = [0usize, 1, 2, 3, 4, 3, 2, 1, 0];
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(reflect(i as isize, 5), e, "reflect({i}, 5)");
        }
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-2, 5), 2);
    }

    #[test]
    fn dwt_1d_round_trip_recovers_signal() {
        for n in [2usize, 3, 4, 5, 8, 9, 16, 17, 31, 64] {
            let signal: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
            let mut buf = signal.clone();
            fdwt_1d_9_7(&mut buf);
            idwt_1d_9_7(&mut buf);
            assert!(
                max_abs_err(&buf, &signal) < 1e-3,
                "9/7 1-D round-trip n={n} err={}",
                max_abs_err(&buf, &signal)
            );
        }
    }

    #[test]
    fn dwt_2d_multilevel_round_trip() {
        let (w, h) = (24usize, 20usize);
        let orig: Vec<f32> = (0..w * h)
            .map(|i| ((i * 131 % 251) as f32) - 125.0)
            .collect();
        for levels in [1u8, 2, 3] {
            let mut buf = orig.clone();
            forward_dwt_9_7(&mut buf, w, h, levels).unwrap();
            inverse_dwt_9_7(&mut buf, w, h, levels).unwrap();
            let err = max_abs_err(&buf, &orig);
            assert!(err < 5e-2, "9/7 2-D round-trip levels={levels} err={err}");
        }
    }

    #[test]
    fn forward_concentrates_energy_in_ll() {
        // A smooth ramp should put most of its energy in the LL (top-left) band.
        let (w, h) = (16usize, 16usize);
        let img: Vec<f32> = (0..w * h)
            .map(|i| (i % w) as f32 + (i / w) as f32)
            .collect();
        let mut buf = img.clone();
        forward_dwt_9_7(&mut buf, w, h, 1).unwrap();
        let (sn_x, sn_y) = (w.div_ceil(2), h.div_ceil(2));
        let mut ll = 0f32;
        let mut rest = 0f32;
        for y in 0..h {
            for x in 0..w {
                let e = buf[y * w + x].powi(2);
                if x < sn_x && y < sn_y {
                    ll += e;
                } else {
                    rest += e;
                }
            }
        }
        assert!(
            ll > rest,
            "LL energy {ll} should exceed detail energy {rest}"
        );
    }
}

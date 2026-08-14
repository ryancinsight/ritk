//! Forward and inverse discrete wavelet transform — 5/3 reversible (lossless).
//!
//! # Specification (ISO 15444-1 Annex F)
//! The 5/3 integer lifting steps for the **inverse** 1-D transform are:
//!
//! **Step 1 (undo update / even synthesis)**:
//! ```text
//! x[2n]   = s[n] − floor((d[n−1] + d[n] + 2) / 4)
//! ```
//! **Step 2 (undo predict / odd synthesis)**:
//! ```text
//! x[2n+1] = d[n] + floor((x[2n] + x[2n+2]) / 2)
//! ```
//!
//! # Layout
//! Coefficients are stored in the standard **Mallat layout**: at each level the
//! current rectangle is split into LL (top-left), HL (top-right: x-high,
//! y-low), LH (bottom-left), and HH (bottom-right); the next level recurses on
//! the LL quadrant. Per level, synthesis interleaves and inverts rows first,
//! then columns (matching OpenJPEG's H-then-V order); analysis applies the
//! exact reverse (columns first, then rows).
//!
//! # Limitation
//! Reversible 5/3 only. The irreversible 9/7 wavelet (lossy) is tracked as
//! backlog item J2K-LOSSY-97.

use anyhow::{bail, Context, Result};

/// `ceil(n / 2^k)`.
#[inline]
pub(crate) fn ceil_div_pow2(n: usize, k: u32) -> usize {
    (n + (1usize << k) - 1) >> k
}

/// Apply N levels of the inverse 5/3 DWT to a row-major Mallat-layout buffer
/// in-place. After the call, `samples[y * width + x]` holds the reconstructed
/// DC-shifted integer sample for pixel `(x, y)`.
///
/// # Errors
/// Returns an error when the dimensions overflow or do not match
/// `samples.len()`, nonzero levels accompany a zero dimension, the
/// decomposition depth exceeds the platform geometry width, or the reusable
/// transform workspace length overflows.
pub fn inverse_dwt_5_3(
    samples: &mut [i32],
    width: usize,
    height: usize,
    num_levels: u8,
) -> Result<()> {
    validate_transform_geometry(samples.len(), width, height, num_levels, "inverse_dwt_5_3")?;
    if num_levels == 0 {
        return Ok(());
    }

    let max_dimension = width.max(height);
    let mut workspace = transform_workspace(max_dimension, "inverse_dwt_5_3")?;
    let (buf, line) = workspace.split_at_mut(max_dimension);

    // Synthesise level by level from the coarsest rectangle up to full size.
    for lvl in (1..=u32::from(num_levels)).rev() {
        let cur_w = ceil_div_pow2(width, lvl - 1);
        let cur_h = ceil_div_pow2(height, lvl - 1);
        synth_2d_5_3(samples, width, cur_w, cur_h, buf, line);
    }
    Ok(())
}

/// Apply N levels of the forward 5/3 DWT to a row-major buffer in-place,
/// producing the Mallat coefficient layout consumed by [`inverse_dwt_5_3`].
///
/// # Errors
/// Returns an error when the dimensions overflow or do not match
/// `samples.len()`, nonzero levels accompany a zero dimension, the
/// decomposition depth exceeds the platform geometry width, or the reusable
/// transform workspace length overflows.
pub fn forward_dwt_5_3(
    samples: &mut [i32],
    width: usize,
    height: usize,
    num_levels: u8,
) -> Result<()> {
    validate_transform_geometry(samples.len(), width, height, num_levels, "forward_dwt_5_3")?;
    if num_levels == 0 {
        return Ok(());
    }

    let max_dimension = width.max(height);
    let mut workspace = transform_workspace(max_dimension, "forward_dwt_5_3")?;
    let (buf, line) = workspace.split_at_mut(max_dimension);

    for lvl in 1..=u32::from(num_levels) {
        let cur_w = ceil_div_pow2(width, lvl - 1);
        let cur_h = ceil_div_pow2(height, lvl - 1);
        analyse_2d_5_3(samples, width, cur_w, cur_h, buf, line);
    }
    Ok(())
}

fn validate_transform_geometry(
    sample_count: usize,
    width: usize,
    height: usize,
    num_levels: u8,
    operation: &str,
) -> Result<()> {
    let expected = width
        .checked_mul(height)
        .with_context(|| format!("{operation}: width({width})×height({height}) overflows usize"))?;
    if sample_count != expected {
        bail!("{operation}: samples.len()={sample_count} != width({width})×height({height})");
    }
    if num_levels > 0 && (width == 0 || height == 0) {
        bail!("{operation}: nonzero decomposition levels require positive dimensions");
    }
    // `>=`, not `>`: shifting by exactly the width of `usize` is already
    // undefined, so the boundary value has to be rejected too.
    if u32::from(num_levels) >= usize::BITS {
        bail!(
            "{operation}: decomposition levels ({num_levels}) reach the {}-bit geometry limit",
            usize::BITS
        );
    }
    Ok(())
}

fn transform_workspace(max_dimension: usize, operation: &str) -> Result<Vec<i32>> {
    let workspace_len = max_dimension
        .checked_mul(2)
        .with_context(|| format!("{operation}: workspace length overflow"))?;
    Ok(vec![0i32; workspace_len])
}

// ── 2-D per-level passes ─────────────────────────────────────────────────────

/// One synthesis level on the `[0..cur_h, 0..cur_w]` rectangle (stride
/// `stride`): rows first (interleave [low|high] halves → inverse 1-D), then
/// columns.
fn synth_2d_5_3(
    samples: &mut [i32],
    stride: usize,
    cur_w: usize,
    cur_h: usize,
    buf: &mut [i32],
    line: &mut [i32],
) {
    let sn_x = cur_w.div_ceil(2);
    let sn_y = cur_h.div_ceil(2);

    // Horizontal pass: each row holds [lows (0..sn_x) | highs (sn_x..cur_w)].
    for y in 0..cur_h {
        let row = &samples[y * stride..y * stride + cur_w];
        interleave(row, sn_x, &mut buf[..cur_w]);
        idwt_1d_5_3(&mut buf[..cur_w]);
        samples[y * stride..y * stride + cur_w].copy_from_slice(&buf[..cur_w]);
    }

    // Vertical pass: each column holds [lows (0..sn_y) | highs (sn_y..cur_h)].
    for x in 0..cur_w {
        for y in 0..cur_h {
            line[y] = samples[y * stride + x];
        }
        interleave(&line[..cur_h], sn_y, &mut buf[..cur_h]);
        idwt_1d_5_3(&mut buf[..cur_h]);
        for y in 0..cur_h {
            samples[y * stride + x] = buf[y];
        }
    }
}

/// One analysis level on the `[0..cur_h, 0..cur_w]` rectangle — the exact
/// inverse of [`synth_2d_5_3`]: columns first, then rows, de-interleaving the
/// lifted output into [low|high] halves.
fn analyse_2d_5_3(
    samples: &mut [i32],
    stride: usize,
    cur_w: usize,
    cur_h: usize,
    buf: &mut [i32],
    line: &mut [i32],
) {
    let sn_x = cur_w.div_ceil(2);
    let sn_y = cur_h.div_ceil(2);

    // Vertical pass.
    for x in 0..cur_w {
        for y in 0..cur_h {
            line[y] = samples[y * stride + x];
        }
        fdwt_1d_5_3(&mut line[..cur_h]);
        deinterleave(&line[..cur_h], sn_y, &mut buf[..cur_h]);
        for y in 0..cur_h {
            samples[y * stride + x] = buf[y];
        }
    }

    // Horizontal pass.
    for y in 0..cur_h {
        line[..cur_w].copy_from_slice(&samples[y * stride..y * stride + cur_w]);
        fdwt_1d_5_3(&mut line[..cur_w]);
        deinterleave(&line[..cur_w], sn_x, &mut buf[..cur_w]);
        samples[y * stride..y * stride + cur_w].copy_from_slice(&buf[..cur_w]);
    }
}

/// `[low half | high half] → interleaved (even = low, odd = high)`.
#[inline]
fn interleave(halves: &[i32], sn: usize, out: &mut [i32]) {
    for (i, &v) in halves[..sn].iter().enumerate() {
        out[2 * i] = v;
    }
    for (i, &v) in halves[sn..].iter().enumerate() {
        out[2 * i + 1] = v;
    }
}

/// `interleaved (even = low, odd = high) → [low half | high half]`.
#[inline]
fn deinterleave(inter: &[i32], sn: usize, out: &mut [i32]) {
    for i in 0..sn {
        out[i] = inter[2 * i];
    }
    for i in 0..inter.len() - sn {
        out[sn + i] = inter[2 * i + 1];
    }
}

// ── 1-D lifting kernels ───────────────────────────────────────────────────────

/// 1-D inverse 5/3 integer lifting (ISO 15444-1 §F.3.8.2) on an interleaved
/// buffer (even indices = `s[i]` lows, odd = `d[i]` highs). Whole-sample-symmetric
/// boundary extension.
fn idwt_1d_5_3(buf: &mut [i32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    let sn = n.div_ceil(2); // number of low (even-index) samples
    let dn = n / 2; // number of high (odd-index) samples

    // Each phase reads only the opposite parity, so the consumed parity can
    // be overwritten in place without changing a later input.
    // Step 1: undo update — restore even (low) samples.
    // x[2i] = s[i] − floor((d[i−1] + d[i] + 2) / 4); clamp d-index to [0, dn-1].
    for i in 0..sn {
        let dl = if i == 0 { buf[1] } else { buf[2 * i - 1] };
        let dr = if i >= dn {
            buf[2 * (dn - 1) + 1]
        } else {
            buf[2 * i + 1]
        };
        buf[2 * i] -= (dl + dr + 2) >> 2;
    }

    // Step 2: undo predict — restore odd (high) samples from restored lows.
    // x[2i+1] = d[i] + floor((x[2i] + x[2i+2]) / 2); clamp s-index to [0, sn-1].
    for i in 0..dn {
        let sr = if i + 1 >= sn {
            buf[2 * (sn - 1)]
        } else {
            buf[2 * (i + 1)]
        };
        buf[2 * i + 1] += (buf[2 * i] + sr) >> 1;
    }
}

/// 1-D forward 5/3 integer lifting — exact inverse of [`idwt_1d_5_3`]; output
/// is interleaved (even = low, odd = high).
fn fdwt_1d_5_3(buf: &mut [i32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    let sn = n.div_ceil(2);
    let dn = n / 2;

    // Each phase reads only the opposite parity, so the consumed parity can
    // be overwritten in place without changing a later input.
    // Step 1: predict — d[i] = x[2i+1] − floor((x[2i] + x[2i+2]) / 2).
    for i in 0..dn {
        let sr = if i + 1 >= sn {
            buf[2 * (sn - 1)]
        } else {
            buf[2 * (i + 1)]
        };
        buf[2 * i + 1] -= (buf[2 * i] + sr) >> 1;
    }

    // Step 2: update — s[i] = x[2i] + floor((d[i−1] + d[i] + 2) / 4).
    for i in 0..sn {
        let dl = if i == 0 { buf[1] } else { buf[2 * i - 1] };
        let dr = if i >= dn {
            buf[2 * (dn - 1) + 1]
        } else {
            buf[2 * i + 1]
        };
        buf[2 * i] += (dl + dr + 2) >> 2;
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;

    fn round_trip_1d(signal: &[i32]) {
        let mut buf = signal.to_vec();
        fdwt_1d_5_3(&mut buf);
        idwt_1d_5_3(&mut buf);
        assert_eq!(&buf, signal, "5/3 1-D round-trip failed");
    }

    #[test]
    fn idwt_1d_constant_signal() {
        round_trip_1d(&[5, 5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn idwt_1d_ramp() {
        round_trip_1d(&[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn idwt_1d_arbitrary_8_elements() {
        round_trip_1d(&[-4, 10, 3, -7, 0, 127, -128, 64]);
    }

    #[test]
    fn idwt_1d_odd_length() {
        round_trip_1d(&[9, -3, 17, 0, 5, 22, -11]);
    }

    #[test]
    fn idwt_1d_single_element_is_noop() {
        let mut buf = [42i32];
        idwt_1d_5_3(&mut buf);
        assert_eq!(buf, [42]);
    }

    #[test]
    fn inverse_dwt_5_3_zero_levels_is_noop() {
        let mut samples = vec![1i32, 2, 3, 4];
        inverse_dwt_5_3(&mut samples, 2, 2, 0).unwrap();
        assert_eq!(samples, vec![1, 2, 3, 4]);
    }

    #[test]
    fn dwt_rejects_zero_geometry_with_nonzero_levels() {
        let mut samples = Vec::new();
        let error = forward_dwt_5_3(&mut samples, usize::MAX, 0, 1)
            .expect_err("nonzero levels require positive dimensions");
        assert!(
            format!("{error:#}").contains("require positive dimensions"),
            "got: {error:#}"
        );
    }

    #[test]
    fn dwt_rejects_dimension_product_overflow() {
        let mut samples = Vec::new();
        let error = forward_dwt_5_3(&mut samples, usize::MAX, 2, 0)
            .expect_err("overflowing dimensions must fail");
        assert!(
            format!("{error:#}").contains("overflows usize"),
            "got: {error:#}"
        );
    }

    #[test]
    fn dwt_rejects_levels_wider_than_geometry_arithmetic() {
        let mut samples = [7i32];
        let levels = u8::try_from(usize::BITS + 1).expect("supported usize width fits u8");
        let error = inverse_dwt_5_3(&mut samples, 1, 1, levels)
            .expect_err("oversized decomposition depth must fail");
        assert!(
            format!("{error:#}").contains("geometry limit"),
            "got: {error:#}"
        );
    }

    fn round_trip_2d(w: usize, h: usize, levels: u8) {
        // Deterministic LCG image.
        let mut state = 0xDEADBEEFu64;
        let original: Vec<i32> = (0..w * h)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) & 0xFF) as i32 - 128
            })
            .collect();
        let mut buf = original.clone();
        forward_dwt_5_3(&mut buf, w, h, levels).unwrap();
        if levels > 0 && w > 1 && h > 1 {
            assert_ne!(buf, original, "forward DWT must alter coefficients");
        }
        inverse_dwt_5_3(&mut buf, w, h, levels).unwrap();
        assert_eq!(
            buf, original,
            "2-D 5/3 round-trip failed ({w}×{h}, L{levels})"
        );
    }

    #[test]
    fn dwt_2d_round_trip_even_dims() {
        for levels in 0..=3 {
            round_trip_2d(16, 8, levels);
        }
    }

    #[test]
    fn dwt_2d_round_trip_odd_dims() {
        for levels in 0..=3 {
            round_trip_2d(7, 5, levels);
            round_trip_2d(13, 9, levels);
        }
    }

    #[test]
    fn dwt_2d_round_trip_single_row_and_column() {
        round_trip_2d(9, 1, 2);
        round_trip_2d(1, 9, 2);
    }
}

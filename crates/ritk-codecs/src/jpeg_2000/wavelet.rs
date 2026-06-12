//! Inverse discrete wavelet transform — 5/3 reversible (lossless) only.
//!
//! # Specification (ISO 15444-1 §F.3.2)
//! The 5/3 integer lifting steps for the **inverse** 1-D transform are:
//!
//! **Step 1 (undo predict / odd synthesis)**:
//! ```text
//! x[2n+1] = d[n] + floor((s[n] + s[n+1]) / 2)
//! ```
//! **Step 2 (undo update / even synthesis)**:
//! ```text
//! x[2n]   = s[n] − floor((x[2n−1] + x[2n+1] + 2) / 4)
//! ```
//! The 2-D inverse DWT applies the 1-D transform to rows then columns.
//!
//! # Limitation
//! This module implements the **reversible 5/3** wavelet only.  The irreversible
//! 9/7 wavelet (lossy) is planned for a future sprint.

#![allow(dead_code)] // Public entry-point; consumed when DWT decoding is wired into image.rs.

use anyhow::{bail, Result};

// ── Public entry point ────────────────────────────────────────────────────────

/// Apply N levels of the inverse 5/3 DWT to a row-major tile buffer in-place.
///
/// After decoding, `samples[y * width + x]` holds the reconstructed DC-shifted
/// integer sample for pixel `(x, y)`.
///
/// `num_levels` is the number of DWT decomposition levels that were applied
/// during encoding (`cod.num_decomp_levels`).  0 means no DWT was applied and
/// this function is a no-op.
///
/// # Errors
/// Returns an error if `num_levels > 0` and the wavelet reconstruction fails.
pub fn inverse_dwt_5_3(
    samples: &mut [i32],
    width: usize,
    height: usize,
    num_levels: u8,
) -> Result<()> {
    if num_levels == 0 {
        return Ok(());
    }
    if samples.len() != width * height {
        bail!(
            "inverse_dwt_5_3: samples.len()={} != width({})×height({})",
            samples.len(),
            width,
            height
        );
    }

    // Apply inverse DWT level by level from coarsest to finest.
    // At level `l`, the LL band occupies the top-left
    //   ceil(width / 2^l) × ceil(height / 2^l) corner.
    // We reconstruct level by level from level `num_levels` down to 1.

    let mut ll_w = ceil_div_pow2(width, num_levels as u32);
    let mut ll_h = ceil_div_pow2(height, num_levels as u32);

    for _lvl in (1..=num_levels).rev() {
        // The current resolution doubles each time.
        let dst_w = (ll_w * 2).min(width);
        let dst_h = (ll_h * 2).min(height);

        // Inverse 2-D 5/3 DWT on the region [0..dst_h, 0..dst_w] of `samples`.
        inverse_2d_5_3(samples, width, dst_w, dst_h);

        ll_w = dst_w;
        ll_h = dst_h;
    }

    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// `ceil(n / 2^k)`.
fn ceil_div_pow2(n: usize, k: u32) -> usize {
    (n + (1usize << k) - 1) >> k
}

/// 2-D inverse 5/3 DWT on the sub-region `[0..roi_h, 0..roi_w]` of a row-major
/// buffer with row stride `stride`.
///
/// The sub-region contains the LL band in the top-left quarter and the HL, LH,
/// HH bands in the other three quadrants (standard J2K subband layout).
fn inverse_2d_5_3(samples: &mut [i32], stride: usize, roi_w: usize, roi_h: usize) {
    // 1. Apply inverse 1-D transform to each ROW of the ROI.
    let mut row_buf = vec![0i32; roi_w];
    for y in 0..roi_h {
        for x in 0..roi_w {
            row_buf[x] = samples[y * stride + x];
        }
        idwt_1d_5_3(&mut row_buf);
        for x in 0..roi_w {
            samples[y * stride + x] = row_buf[x];
        }
    }

    // 2. Apply inverse 1-D transform to each COLUMN of the ROI.
    let mut col_buf = vec![0i32; roi_h];
    for x in 0..roi_w {
        for y in 0..roi_h {
            col_buf[y] = samples[y * stride + x];
        }
        idwt_1d_5_3(&mut col_buf);
        for y in 0..roi_h {
            samples[y * stride + x] = col_buf[y];
        }
    }
}

/// 1-D inverse 5/3 integer lifting (ISO 15444-1 §F.3.2).
///
/// The input buffer contains the **interleaved** low-pass (even indices = s[i])
/// and high-pass (odd indices = d[i]) subband coefficients produced by the
/// forward DWT.  After the transform the buffer holds the reconstructed
/// spatial-domain signal.
///
/// Boundary extension follows whole-sample symmetry:
/// - for the undo-update step (reads HL neighbours of an LL sample):
///   d[-1] → d[0]; d[dn] → d[dn-1]
/// - for the undo-predict step (reads LL neighbours of an HL sample):
///   s[-1] → s[0]; s[sn] → s[sn-1]
fn idwt_1d_5_3(buf: &mut [i32]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    let sn = n.div_ceil(2); // number of LL (even-index) samples
    let dn = n / 2; // number of HL (odd-index) samples

    let mut tmp = buf.to_vec();

    // Step 1: undo update — restore even (LL) samples.
    // x[2i] = s[i] − floor((d[i−1] + d[i] + 2) / 4)
    // d[k] lives at tmp[2k+1]; clamp k to [0, dn-1].
    for i in 0..sn {
        let dl = if i == 0 { tmp[1] } else { tmp[2 * i - 1] }; // d[max(0, i-1)]
        let dr = if i >= dn {
            tmp[2 * (dn - 1) + 1]
        } else {
            tmp[2 * i + 1]
        }; // d[min(dn-1, i)]
        tmp[2 * i] -= (dl + dr + 2) >> 2;
    }

    // Step 2: undo predict — restore odd (HL) samples using restored LL.
    // x[2i+1] = d[i] + floor((s[i] + s[i+1]) / 2)
    // s[k] lives at tmp[2k] (already restored above); clamp k+1 to [0, sn-1].
    for i in 0..dn {
        let sr = if i + 1 >= sn {
            tmp[2 * (sn - 1)]
        } else {
            tmp[2 * (i + 1)]
        }; // s[min(sn-1, i+1)]
        tmp[2 * i + 1] += (tmp[2 * i] + sr) >> 1;
    }

    buf.copy_from_slice(&tmp);
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward 5/3 DWT for test purposes (spatial → interleaved LL/HL coefficients).
    fn fdwt_1d_5_3(buf: &mut [i32]) {
        let n = buf.len();
        if n <= 1 {
            return;
        }

        let sn = n.div_ceil(2);
        let dn = n / 2;
        let mut tmp = buf.to_vec();

        // Step 1: predict — compute HL (odd) from original even neighbours.
        // d[i] = x[2i+1] - floor((x[2i] + x[2i+2]) / 2)
        // x[2i+2] = buf[2*(i+1)]; clamp i+1 to [0, sn-1].
        for i in 0..dn {
            let sr = if i + 1 >= sn {
                buf[2 * (sn - 1)]
            } else {
                buf[2 * (i + 1)]
            };
            tmp[2 * i + 1] = buf[2 * i + 1] - ((buf[2 * i] + sr) >> 1);
        }

        // Step 2: update — compute LL (even) using predicted HL neighbours.
        // s'[i] = x[2i] + floor((d[i-1] + d[i] + 2) / 4)
        // d[k] lives at tmp[2k+1]; clamp k to [0, dn-1].
        for i in 0..sn {
            let dl = if i == 0 { tmp[1] } else { tmp[2 * i - 1] };
            let dr = if i >= dn {
                tmp[2 * (dn - 1) + 1]
            } else {
                tmp[2 * i + 1]
            };
            tmp[2 * i] = buf[2 * i] + ((dl + dr + 2) >> 2);
        }

        buf.copy_from_slice(&tmp);
    }

    fn forward_then_inverse_round_trips(signal: &[i32]) {
        let mut buf = signal.to_vec();
        fdwt_1d_5_3(&mut buf);
        idwt_1d_5_3(&mut buf);
        assert_eq!(&buf, signal, "5/3 1-D round-trip failed");
    }

    #[test]
    fn idwt_1d_constant_signal() {
        forward_then_inverse_round_trips(&[5, 5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn idwt_1d_ramp() {
        forward_then_inverse_round_trips(&[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn idwt_1d_arbitrary_8_elements() {
        forward_then_inverse_round_trips(&[-4, 10, 3, -7, 0, 127, -128, 64]);
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
}

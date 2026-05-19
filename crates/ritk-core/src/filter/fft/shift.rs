//! FFT Shift filter.
//!
//! Rearranges the zero-frequency (DC) component from the corners to the centre
//! of a complex frequency-domain image by applying a cyclic roll of `H/2` rows
//! and `W/2` complex columns.
//!
//! # Complex image convention
//!
//! Complex images are stored with interleaved real/imaginary `f32` pairs:
//! - 2D: shape `[H, 2·W]` where `data[r·2W + 2c] = Re(F[r,c])`,
//!   `data[r·2W + 2c + 1] = Im(F[r,c])`
//! - 3D: shape `[D, H, 2·W]`
//!
//! # Mathematical contract
//!
//! For a complex image with shape `[H, 2·W]` and `W = cw/2`:
//! ```text
//! out[r, c] = in[(r + H/2) % H, (c + W/2) % W]
//! ```
//! This is a cyclic roll by `(H/2, W/2)` in complex-pixel coordinates.
//!
//! # Self-inverse property
//!
//! `shift(shift(x)) == x` for all even dimension sizes.
//!
//! **Proof.** Let `s_D = D/2`. Double-application maps depth `z` to
//! `(z + s_D + s_D) % D = (z + D) % D = z` for even D. The same
//! argument applies to `H/2` and `W/2` on the remaining axes. ∎

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Moves the zero-frequency (DC) component from the corners to the centre of
/// a complex frequency-domain image.
///
/// The operation is a cyclic roll by `(H/2, W/2)` in complex-pixel coordinates
/// and is self-inverse for even-dimension images.
///
/// # Example
/// ```ignore
/// let shifted = FftShiftFilter::new().apply_2d(&freq_image)?;
/// // … inspect centred spectrum …
/// let restored = FftShiftFilter::new().apply_2d(&shifted)?; // back to original
/// ```
pub struct FftShiftFilter;

impl FftShiftFilter {
    /// Construct a new `FftShiftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply FFT shift to a 2-D complex (frequency-domain) image.
    ///
    /// # Input / output
    /// Shape `[H, 2·W]` (unchanged): the zero-frequency component is
    /// cyclic-rolled to the centre of the array.
    ///
    /// # Errors
    /// Returns `Err` when the tensor data cannot be extracted as `f32`.
    pub fn apply_2d<B: Backend>(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, cw] = image.shape();
        let w = cw / 2;

        let (vals, _) = extract_vec(image)?;

        let h_shift = h / 2;
        let w_shift = w / 2;

        let mut out = vec![0.0_f32; h * cw];

        for r in 0..h {
            let src_r = (r + h_shift) % h;
            for c in 0..w {
                let src_c = (c + w_shift) % w;
                let src_idx = src_r * cw + 2 * src_c;
                let tgt_idx = r * cw + 2 * c;
                out[tgt_idx] = vals[src_idx];
                out[tgt_idx + 1] = vals[src_idx + 1];
            }
        }

        Ok(rebuild(out, [h, cw], image))
    }

    /// Apply FFT shift to a 3-D complex (frequency-domain) image.
    ///
    /// Performs a true 3-D cyclic roll by `(D/2, H/2, W/2)`, moving the
    /// zero-frequency component from the corners to the volumetric centre.
    ///
    /// Matches ITK's `itk::FFTShiftImageFilter` behaviour which shifts
    /// all axes (not just 2-D per-slice).
    ///
    /// # Input / output
    /// Shape `[D, H, 2·W]` (unchanged).
    ///
    /// # Self-inverse
    /// `apply_3d(apply_3d(x)) == x` for even D, H, W.
    ///
    /// # Errors
    /// Returns `Err` when the tensor data cannot be extracted as `f32`.
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let [depth, h, cw] = image.shape();
        let w = cw / 2;

        let (vals, _) = extract_vec(image)?;

        let d_shift = depth / 2;
        let h_shift = h / 2;
        let w_shift = w / 2;

        let mut out = vec![0.0_f32; depth * h * cw];

        for d in 0..depth {
            let src_d = (d + d_shift) % depth;
            for r in 0..h {
                let src_r = (r + h_shift) % h;
                for c in 0..w {
                    let src_c = (c + w_shift) % w;
                    let src_idx = src_d * h * cw + src_r * cw + 2 * src_c;
                    let tgt_idx = d * h * cw + r * cw + 2 * c;
                    out[tgt_idx] = vals[src_idx];
                    out[tgt_idx + 1] = vals[src_idx + 1];
                }
            }
        }

        Ok(rebuild(out, [depth, h, cw], image))
    }
}

impl Default for FftShiftFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_shift.rs"]
mod tests;

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

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Moves the zero-frequency (DC) component from the corners to the centre of
/// a complex frequency-domain image.
///
/// The operation is a cyclic roll by `(H/2, W/2)` in complex-pixel coordinates
/// and is self-inverse for even-dimension images.
///
/// # Example
/// ```ignore
/// let shifted = FftShiftFilter::new().apply::<2>(&freq_image)?;
/// // … inspect centred spectrum …
/// let restored = FftShiftFilter::new().apply::<2>(&shifted)?; // back to original
/// ```
pub struct FftShiftFilter;

impl FftShiftFilter {
    /// Construct a new `FftShiftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply FFT shift to a D-dimensional complex (frequency-domain) image.
    ///
    /// Performs a cyclic roll by `dim / 2` along every axis (in complex-pixel
    /// space for the innermost axis), moving the zero-frequency component from
    /// the corners to the centre.
    ///
    /// # Self-inverse
    /// `apply(apply(x)) == x` for even dimension sizes.
    ///
    /// # Errors
    /// Returns `Err` when the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Result<Image<B, D>> {
        Self::apply_inner(image)
    }

    fn apply_inner<B: Backend, const D: usize>(image: &Image<B, D>) -> Result<Image<B, D>> {
        let dims = image.shape();
        let cw = dims[D - 1]; // complex width = 2 * W
        let w = cw / 2;

        let (vals, _) = extract_vec(image)?;

        // Shift amounts for the D-1 outer spatial axes.
        let shifts: [usize; D] = core::array::from_fn(|a| dims[a] / 2);
        let w_shift = w / 2;

        // Row strides: stride[a] = number of *rows* (not f32 elements)
        // spanned by one step along axis a. The last outer axis (D-2)
        // has stride 1; earlier axes multiply by subsequent dims.
        let mut row_strides = [0usize; D];
        row_strides[D - 2] = 1;
        for i in (0..D - 2).rev() {
            row_strides[i] = row_strides[i + 1] * dims[i + 1];
        }

        // Total number of rows (outer index space).
        let row_count: usize = dims[..D - 1].iter().product();
        let mut out = vec![0.0_f32; row_count * cw];

        for flat_row in 0..row_count {
            // Decompose the row-major flat index into per-axis coordinates.
            let mut coords = [0usize; D];
            let mut remaining = flat_row;
            for a in 0..D - 1 {
                coords[a] = remaining / row_strides[a];
                remaining %= row_strides[a];
            }

            // Compute the source row-major flat index with cyclic shifts.
            let mut src_row = 0usize;
            for a in 0..D - 1 {
                let src_a = (coords[a] + shifts[a]) % dims[a];
                src_row += src_a * row_strides[a];
            }

            // Copy w complex pixels with cyclic shift in the inner dimension.
            let src_base = src_row * cw;
            let tgt_base = flat_row * cw;
            for c in 0..w {
                let src_c = (c + w_shift) % w;
                let src_idx = src_base + 2 * src_c;
                let tgt_idx = tgt_base + 2 * c;
                out[tgt_idx] = vals[src_idx];
                out[tgt_idx + 1] = vals[src_idx + 1];
            }
        }

        Ok(rebuild(out, dims, image))
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

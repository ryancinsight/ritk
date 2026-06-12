//! Inverse Fast Fourier Transform filter.
//!
//! Transforms a frequency-domain complex image back to its spatial representation.
//! This is the inverse of [`super::forward::ForwardFftFilter`].
//!
//! # Mathematical specification
//!
//! For a 2-D complex image F(u, v) with dimensions (H, W):
//!
//! ```text
//! f(x, y) = (1 / H·W) · Σ_{u=0}^{H-1} Σ_{v=0}^{W-1} F(u,v) · e^{+2πi(ux/H + vy/W)}
//! ```
//!
//! The transform is applied separably: 1-D IFFT along rows, then along columns.
//! For 3-D images an additional 1-D IFFT is applied along the depth axis.
//!
//! `rustfft` computes the unnormalized IFFT:
//!
//! ```text
//! IFFT_unnorm(F)[n] = Σ_{k} F[k] · e^{+2πi·k·n/N}
//! ```
//!
//! All IFFT passes are completed first; a single normalization by `1/N`
//! (N = product of all spatial dimensions) is applied afterwards. This
//! satisfies the round-trip identity `inverse(forward(f)) ≈ f` to within
//! f32 rounding error.
//!
//! # Input format (shared with ForwardFftFilter)
//!
//! Complex images are stored with interleaved (Re, Im) pairs in the last
//! dimension:
//!
//! - 2-D input shape `[H, 2·W]`:
//!   Re at flat index `r·2W + 2c`, Im at `r·2W + 2c + 1`
//! - 3-D input shape `[D, H, 2·W]`:
//!   Re at `d·H·2W + r·2W + 2c`, Im at `+1`

use crate::fft::convolution::{fft_nd, InverseFft};
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use ritk_image::Image;
use rustfft::{num_complex::Complex, FftPlanner};

// ── Struct ────────────────────────────────────────────────────────────────────

/// Inverse Fast Fourier Transform filter.
///
/// Transforms a complex-valued frequency-domain image (produced by
/// [`super::forward::ForwardFftFilter`]) back to the spatial domain.
/// Spatial metadata (origin, spacing, direction) is preserved from the
/// complex input image.
///
/// # Output
///
/// - 2-D: shape `[H, W]`, real-valued, normalized by `1/(H·W)`.
/// - 3-D: shape `[D, H, W]`, real-valued, normalized by `1/(D·H·W)`.
///
/// # Complexity
///
/// O(N log N) where N = product of spatial dimensions.
pub struct InverseFftFilter;

// ── impl ──────────────────────────────────────────────────────────────────────

impl InverseFftFilter {
    /// Create a new `InverseFftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply inverse FFT to a D-dimensional complex image.
    ///
    /// # Input
    ///
    /// Shape `[..., 2·W]` — Re at `...·2W + 2c`, Im at `...·2W + 2c + 1`.
    ///
    /// # Output
    ///
    /// Shape `[..., W]` — real-valued spatial image, normalized by `1/N`
    /// where N = product of spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the last dimension is odd (not a valid complex
    /// interleaved layout) or when the backend tensor cannot be converted to
    /// `f32`.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Result<Image<B, D>> {
        Self::apply_inner(image)
    }

    /// Dimension-generic inverse FFT implementation.
    ///
    /// Deinterleaves (Re, Im) pairs into a complex buffer, applies a
    /// separable N-D inverse FFT via [`fft_nd`], normalizes by `1/N`
    /// where N = product of spatial dimensions, and returns the real part.
    fn apply_inner<B: Backend, const D: usize>(image: &Image<B, D>) -> Result<Image<B, D>> {
        let dims = image.shape();
        let cw = dims[D - 1];
        if !cw.is_multiple_of(2) {
            return Err(anyhow!(
                "InverseFftFilter: {}D input last dimension must be even \
                 (got {}); expected interleaved complex layout",
                D,
                cw
            ));
        }
        let w = cw / 2;

        // Output spatial dimensions: same as input but last dim = W (not 2*W).
        let mut out_dims = dims;
        out_dims[D - 1] = w;

        let n_spatial: usize = out_dims.iter().product();
        let (vals, _) = extract_vec(image)?;

        // Deinterleave (Re, Im) pairs into a complex buffer of shape `out_dims`.
        let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n_spatial);

        // For 2D: iterate over h rows × w complex cols.
        // For 3D: iterate over d × h rows × w complex cols.
        // The general pattern: outer_dims = out_dims[..D-1], each outer slice
        // contains `cw` interleaved f32 values encoding `w` complex numbers.
        let outer_count: usize = out_dims[..D - 1].iter().product();
        for outer in 0..outer_count {
            let row_start = outer * cw;
            for c in 0..w {
                buf.push(Complex::new(
                    vals[row_start + 2 * c],
                    vals[row_start + 2 * c + 1],
                ));
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        fft_nd::<D, InverseFft>(&mut buf, &out_dims, &mut planner);

        // Normalize after all IFFT passes.
        // rustfft's IFFT is unnormalized; the factor 1/N accounts for
        // every axis pass combined.
        let scale = 1.0 / n_spatial as f32;
        let out: Vec<f32> = buf.iter().map(|z| z.re * scale).collect();

        Ok(rebuild(out, out_dims, image))
    }
}

impl Default for InverseFftFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_inverse.rs"]
mod tests;

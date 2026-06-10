//! Forward FFT filter.

//! Transforms a real-valued image to its complex frequency-domain representation.
//!
//! # Mathematical specification
//!
//! For a 2-D image f of shape [H, W], the forward DFT is:
//!
//! F(u, v) = Σ_{x=0}^{H-1} Σ_{y=0}^{W-1} f(x,y) · e^{-2πi(ux/H + vy/W)}
//!
//! Applied separably: row-wise DFT first, then column-wise DFT.
//! No normalization is applied in the forward direction (ITK convention).
//!
//! # Output format
//!
//! For a 2-D input of shape [H, W], the output is shape [H, 2*W]:
//! data[r * 2*W + 2*c] = Re(F[r, c])
//! data[r * 2*W + 2*c + 1] = Im(F[r, c])
//!
//! For a 3-D input of shape [D, H, W], the output is shape [D, H, 2*W]:
//! data[d*H*2*W + r*2*W + 2*c] = Re(F[d, r, c])
//! data[d*H*2*W + r*2*W + 2*c + 1] = Im(F[d, r, c])
//!
//! # Complexity
//! O(N log N), N = product of image dimensions.

use crate::filter::fft::convolution::{fft_nd, ForwardFft};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};

/// Forward Fast Fourier Transform filter.
///
/// Transforms a real-valued image into its frequency-domain complex representation.
/// Preserves spatial metadata (origin, spacing, direction) from the source image.
///
/// # Output layout
///
/// 2-D input `[H, W]` → output `[H, 2*W]`: each row contains `W` interleaved `(Re, Im)` pairs.
/// 3-D input `[D, H, W]` → output `[D, H, 2*W]`: same layout per depth slice.
///
/// DC (zero-frequency) component is at index `[0, 0]`.
/// Use [`super::shift::FftShiftFilter`] to move it to the centre.
///
/// # Complexity
/// O(N log N), N = product of image dimensions.
pub struct ForwardFftFilter;

impl ForwardFftFilter {
    /// Create a new `ForwardFftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply forward FFT to a 2-D real image.
    ///
    /// Input shape `[H, W]` → output shape `[H, 2*W]`.
    ///
    /// At row `r` and frequency column `c`:
    /// `out[r * 2*W + 2*c] = Re(F[r, c])`
    /// `out[r * 2*W + 2*c + 1] = Im(F[r, c])`
    pub fn apply_2d<B: Backend>(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        Self::apply::<B, 2>(image)
    }

    /// Apply forward FFT to a 3-D real image.
    ///
    /// Input shape `[D, H, W]` → output shape `[D, H, 2*W]`.
    ///
    /// At depth `d`, row `r`, and frequency column `c`:
    /// `out[d*H*2*W + r*2*W + 2*c] = Re(F[d, r, c])`
    /// `out[d*H*2*W + r*2*W + 2*c + 1] = Im(F[d, r, c])`
    ///
    /// The separable transform is applied row-wise, then column-wise (within each
    /// depth slice), then depth-wise (across all slices at each (row, col) position).
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        Self::apply::<B, 3>(image)
    }

    /// Dimension-generic forward FFT.
    ///
    /// Extracts real data into a complex buffer, applies a separable N-D forward
    /// FFT via [`fft_nd`], and interleaves the result as `(Re, Im)` pairs.
    fn apply<B: Backend, const D: usize>(image: &Image<B, D>) -> Result<Image<B, D>> {
        let dims = image.shape();
        let (vals, _) = extract_vec(image)?;

        let mut buf: Vec<Complex<f32>> = vals.into_iter().map(|v| Complex::new(v, 0.0)).collect();
        let mut planner = FftPlanner::<f32>::new();
        fft_nd::<D, ForwardFft>(&mut buf, &dims, &mut planner);

        // Interleave (Re, Im) pairs into a flat output buffer.
        // The last spatial dimension W becomes 2*W in output.
        let mut out_dims = dims;
        out_dims[D - 1] *= 2;

        let mut out = Vec::with_capacity(buf.len() * 2);
        for z in &buf {
            out.push(z.re);
            out.push(z.im);
        }

        Ok(rebuild(out, out_dims, image))
    }
}

impl Default for ForwardFftFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_forward.rs"]
mod tests;

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

use crate::fft::convolution::{fft_nd, ForwardFft};
use anyhow::Result;
use eunomia::Complex;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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

    /// Apply forward FFT to a D-dimensional real image.
    ///
    /// Input shape `[..., W]` → output shape `[..., 2*W]`.
    ///
    /// At each frequency column `c`:
    /// `out[..., 2*c] = Re(F[..., c])`
    /// `out[..., 2*c + 1] = Im(F[..., c])`
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<f32, B, D>,
    ) -> Result<Image<f32, B, D>> {
        Self::apply_inner(image)
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let (vals, _) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let mut buf: Vec<Complex<f32>> = vals.into_iter().map(|v| Complex::new(v, 0.0)).collect();
        fft_nd::<D, ForwardFft>(&mut buf, &dims);

        let mut out_dims = dims;
        out_dims[D - 1] *= 2;
        let mut out = Vec::with_capacity(buf.len() * 2);
        for z in &buf {
            out.push(z.re);
            out.push(z.im);
        }
        crate::native_support::rebuild_image(out, out_dims, image, backend)
    }

    fn apply_inner<B: Backend, const D: usize>(
        image: &Image<f32, B, D>,
    ) -> Result<Image<f32, B, D>> {
        let dims = image.shape();
        let (vals, _) = extract_vec(image)?;

        let mut buf: Vec<Complex<f32>> = vals.into_iter().map(|v| Complex::new(v, 0.0)).collect();
        fft_nd::<D, ForwardFft>(&mut buf, &dims);

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

/// Real-to-half-Hermitian forward FFT (`itk::RealToHalfHermitianForwardFFTImageFilter`).
///
/// The DFT of a real signal is Hermitian-symmetric, so the last-axis columns
/// `W/2+1 .. W-1` are conjugates of the retained columns and carry no new
/// information. This filter computes the full forward FFT and keeps only the
/// non-redundant half along the last axis: input `[..., W]` → output
/// `[..., 2*(W/2 + 1)]` interleaved `(Re, Im)`. The kept values are bitwise the
/// same as the corresponding columns of [`ForwardFftFilter`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RealToHalfHermitianForwardFftFilter;

impl RealToHalfHermitianForwardFftFilter {
    /// Create a new filter.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply the half-Hermitian forward FFT to a D-dimensional real image.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<f32, B, D>,
    ) -> Result<Image<f32, B, D>> {
        let dims = image.shape();
        let w = dims[D - 1];
        let half_cols = w / 2 + 1;

        // Full forward FFT, then retain the first `half_cols` complex columns of
        // every last-axis row (each column is two interleaved f32 values).
        let full = ForwardFftFilter::apply_inner::<B, D>(image)?;
        let (full_vals, full_dims) = extract_vec(&full)?;
        let row = full_dims[D - 1]; // == 2 * w
        let keep = 2 * half_cols;
        let rows = full_vals.len() / row;
        let mut out = Vec::with_capacity(rows * keep);
        for r in 0..rows {
            let base = r * row;
            out.extend_from_slice(&full_vals[base..base + keep]);
        }

        let mut out_dims = dims;
        out_dims[D - 1] = keep;
        Ok(rebuild(out, out_dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_forward.rs"]
mod tests;

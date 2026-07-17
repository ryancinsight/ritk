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
//! Apollo's inverse complex FFT path computes the unnormalized IFFT:
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
use eunomia::Complex;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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

    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let cw = dims[D - 1];
        if !cw.is_multiple_of(2) {
            anyhow::bail!(
                "InverseFftFilter: {}D input last dimension must be even (got {}); expected interleaved complex layout",
                D,
                cw
            );
        }
        let w = cw / 2;
        let mut out_dims = dims;
        out_dims[D - 1] = w;
        let n_spatial: usize = out_dims.iter().product();
        let (vals, _) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n_spatial);
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
        fft_nd::<D, InverseFft>(&mut buf, &out_dims);
        let scale = 1.0 / n_spatial as f32;
        let out: Vec<f32> = buf.iter().map(|z| z.re * scale).collect();
        crate::native_support::rebuild_image(out, out_dims, image, backend)
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

        fft_nd::<D, InverseFft>(&mut buf, &out_dims);

        // Normalize after all IFFT passes.
        // Apollo's inverse FFT path is unnormalized; the factor 1/N accounts for
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

/// Half-Hermitian-to-real inverse FFT
/// (`itk::HalfHermitianToRealInverseFFTImageFilter`).
///
/// Reconstructs a real image from the non-redundant half spectrum produced by
/// [`super::forward::RealToHalfHermitianForwardFftFilter`]. The full DFT is
/// rebuilt from the half via Hermitian symmetry
/// `F[k] = conj(F[(−k) mod N])` — last-axis columns `W/2+1 .. W−1` are the
/// conjugates of the retained columns reflected across **every** axis — and then
/// the standard inverse FFT (normalized by `1/N`) is applied.
///
/// The last-axis length `W` cannot be inferred from the half alone
/// (`half_cols = W/2 + 1` for both `W = 2(half_cols−1)` and
/// `W = 2(half_cols−1)+1`), so `actual_x_is_odd` selects the original parity,
/// mirroring ITK's `SetActualXDimensionIsOdd`.
#[derive(Debug, Clone, Copy)]
pub struct HalfHermitianToRealInverseFftFilter {
    /// Whether the original last-axis length `W` was odd.
    pub actual_x_is_odd: bool,
}

impl HalfHermitianToRealInverseFftFilter {
    /// Construct with the original last-axis parity.
    pub fn new(actual_x_is_odd: bool) -> Self {
        Self { actual_x_is_odd }
    }

    /// Apply the half-Hermitian inverse FFT to a D-dimensional half spectrum.
    ///
    /// Input shape `[..., 2·(W/2+1)]` (interleaved half) → output `[..., W]` real.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Result<Image<B, D>> {
        let dims = image.shape();
        let half_w2 = dims[D - 1];
        if !half_w2.is_multiple_of(2) {
            return Err(anyhow!(
                "HalfHermitianToRealInverseFft: last dimension {half_w2} must be even \
                 (interleaved complex layout)"
            ));
        }
        let half_cols = half_w2 / 2;
        if half_cols == 0 {
            return Err(anyhow!("HalfHermitianToRealInverseFft: empty spectrum"));
        }
        let w = 2 * (half_cols - 1) + usize::from(self.actual_x_is_odd);

        let (half, _) = extract_vec(image)?;
        let mut outer_dims = [0usize; D];
        let mut outer_count = 1usize;
        for (a, od) in outer_dims.iter_mut().enumerate().take(D - 1) {
            *od = dims[a];
            outer_count *= dims[a];
        }

        let hrow = 2 * half_cols;
        let frow = 2 * w;
        let mut full = vec![0.0f32; outer_count * frow];
        for o in 0..outer_count {
            // Decompose the row-major outer index, reflect every outer axis
            // (`c → (N−c) mod N`), recompose: the source row for the conjugate-
            // symmetric tail.
            let mut coords = [0usize; D];
            let mut rem = o;
            for a in (0..D - 1).rev() {
                coords[a] = rem % outer_dims[a];
                rem /= outer_dims[a];
            }
            let mut o_ref = 0usize;
            for a in 0..D - 1 {
                let rc = (outer_dims[a] - coords[a]) % outer_dims[a];
                o_ref = o_ref * outer_dims[a] + rc;
            }
            for wc in 0..w {
                let (re, im) = if wc < half_cols {
                    (half[o * hrow + 2 * wc], half[o * hrow + 2 * wc + 1])
                } else {
                    let sw = w - wc;
                    (
                        half[o_ref * hrow + 2 * sw],
                        -half[o_ref * hrow + 2 * sw + 1],
                    )
                };
                full[o * frow + 2 * wc] = re;
                full[o * frow + 2 * wc + 1] = im;
            }
        }

        let mut full_dims = dims;
        full_dims[D - 1] = frow;
        let full_img = rebuild(full, full_dims, image);
        InverseFftFilter::apply_inner::<B, D>(&full_img)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_inverse.rs"]
mod tests;

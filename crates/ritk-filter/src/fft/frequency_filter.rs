//! Frequency-domain filters (ideal low-pass/high-pass, Butterworth).
//!
//! These filters operate by applying a real-valued mask in the frequency domain
//! after a forward FFT + shift, then transforming back via inverse shift + IFFT.
//!
//! # Pipeline
//!
//! ```text
//! input → ForwardFFT → FFTShift → mask → FFTShift⁻¹ → InverseFFT → output
//! ```
//!
//! # Filter transfer functions
//!
//! Let `r = sqrt(fx² + fy² [+ fz²])` be the normalised frequency radius,
//! `c ∈ (0, 0.5]` the cutoff, and `n ∈ ℕ` the Butterworth order.
//!
//! | Filter | H(r) |
//! |---|---|
//! | Ideal low-pass | 1 if r ≤ c, else 0 |
//! | Ideal high-pass | 1 if r ≥ c, else 0 |
//! | Butterworth low-pass | 1 / (1 + (r/c)^(2n)) |
//! | Butterworth high-pass | 1 − 1 / (1 + (r/c)^(2n)) |
//!
//! # Normalised frequency convention
//!
//! After FFT shift, the DC component is at the centre of the array.
//! Frequencies are mapped to the range [−0.5, +0.5] in each axis,
//! where ±0.5 corresponds to the Nyquist frequency.
//!
//! # References
//!
//! - ITK \[`itk::FFTImageFilter`\] implementations for ideal / Butterworth
//! - Gonzalez & Woods, *Digital Image Processing*, 4th ed., ch. 4

use crate::fft::{FftShiftFilter, ForwardFftFilter, InverseFftFilter};
use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Filter type ───────────────────────────────────────────────────────────────

/// Frequency-domain filter transfer function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftFilterKind {
    /// Ideal (hard-threshold) low-pass.
    IdealLowPass,
    /// Ideal (hard-threshold) high-pass.
    IdealHighPass,
    /// Butterworth low-pass (smooth roll-off, configurable order).
    ButterworthLowPass,
    /// Butterworth high-pass (smooth roll-off, configurable order).
    ButterworthHighPass,
}

// ── Frequency response trait ──────────────────────────────────────────────────

/// Trait for frequency-domain transfer functions.
///
/// Each implementor is a zero-sized type (ZST) encoding the filter variant
/// at the type level. The trait provides the canonical `compute_mask` method
/// that generates a real-valued frequency mask for any dimensionality `D`.
pub trait FrequencyResponse: Default {
    /// Evaluate the transfer function `H(r)` at a given normalised radius.
    fn evaluate(radius: f64, cutoff: f64, order: usize) -> f32;

    /// Generate a D-dimensional real-valued frequency mask.
    ///
    /// The mask has shape matching `spatial_dims`. After FFT shift, the DC
    /// component is at the centre of the array; frequencies are mapped to
    /// `[−0.5, +0.5]` per axis.
    fn compute_mask<const D: usize>(
        spatial_dims: &[usize; D],
        cutoff: f64,
        order: usize,
    ) -> Vec<f32> {
        let n: usize = spatial_dims.iter().product();
        let mut mask = vec![0.0_f32; n];

        // Centre coordinates in normalised frequency space.
        let centers: [f64; D] = core::array::from_fn(|a| spatial_dims[a] as f64 / 2.0);

        // Strides for row-major indexing into the mask buffer.
        let mut strides = [0usize; D];
        strides[D - 1] = 1;
        for i in (0..D - 1).rev() {
            strides[i] = strides[i + 1] * spatial_dims[i + 1];
        }

        for (flat, mask_val) in mask.iter_mut().enumerate() {
            let mut remaining = flat;
            let mut radius_sq = 0.0_f64;
            for a in 0..D {
                let coord = remaining / strides[a];
                remaining %= strides[a];
                let f = (coord as f64 - centers[a]) / spatial_dims[a] as f64;
                radius_sq += f * f;
            }
            *mask_val = Self::evaluate(radius_sq.sqrt(), cutoff, order);
        }

        mask
    }
}

/// Ideal low-pass transfer function: H(r) = 1 if r ≤ c, else 0.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdealLowPass;

/// Ideal high-pass transfer function: H(r) = 1 if r ≥ c, else 0.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdealHighPass;

/// Butterworth low-pass transfer function: H(r) = 1 / (1 + (r/c)^(2n)).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ButterworthLowPass;

/// Butterworth high-pass transfer function: H(r) = 1 − 1 / (1 + (r/c)^(2n)).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ButterworthHighPass;

impl FrequencyResponse for IdealLowPass {
    fn evaluate(radius: f64, cutoff: f64, _order: usize) -> f32 {
        if radius <= cutoff {
            1.0
        } else {
            0.0
        }
    }
}

impl FrequencyResponse for IdealHighPass {
    fn evaluate(radius: f64, cutoff: f64, _order: usize) -> f32 {
        if radius >= cutoff {
            1.0
        } else {
            0.0
        }
    }
}

impl FrequencyResponse for ButterworthLowPass {
    fn evaluate(radius: f64, cutoff: f64, order: usize) -> f32 {
        let n = order.max(1) as i32;
        let ratio = radius / cutoff;
        (1.0 / (1.0 + ratio.powi(2 * n))) as f32
    }
}

impl FrequencyResponse for ButterworthHighPass {
    fn evaluate(radius: f64, cutoff: f64, order: usize) -> f32 {
        let n = order.max(1) as i32;
        let ratio = radius / cutoff;
        (1.0 - 1.0 / (1.0 + ratio.powi(2 * n))) as f32
    }
}

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Frequency-domain image filter.
///
/// Applies a real-valued mask in the shifted frequency domain to a **real**
/// input image. The full FFT round-trip (forward → shift → mask → unshift →
/// inverse) is handled internally.
///
/// # Example
///
/// ```ignore
/// use ritk_filter::fft::{FrequencyDomainFilter, FftFilterKind};
///
/// let low = FrequencyDomainFilter::new()
///     .apply::<2>(&image, FftFilterKind::IdealLowPass, 0.3, 2)?;
///
/// let high = FrequencyDomainFilter::new()
///     .apply::<3>(&volume, FftFilterKind::ButterworthHighPass, 0.25, 4)?;
/// ```
pub struct FrequencyDomainFilter;

impl FrequencyDomainFilter {
    /// Create a new `FrequencyDomainFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply a frequency-domain filter to a D-dimensional image.
    ///
    /// # Arguments
    ///
    /// * `image` — Real-valued input.
    /// * `kind` — Transfer function kind.
    /// * `cutoff` — Normalised cutoff frequency in `(0, 0.5]`.
    /// * `order` — Butterworth order (ignored for ideal filters).
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
        kind: FftFilterKind,
        cutoff: f64,
        order: usize,
    ) -> Result<Image<B, D>> {
        Self::validate_cutoff(cutoff)?;
        let spatial_dims = image.shape();

        // Forward FFT → shift.
        let freq = ForwardFftFilter::new().apply(image)?;
        let shifted = FftShiftFilter::new().apply(&freq)?;
        let cw = shifted.shape()[D - 1];

        // Generate and apply mask.
        let (vals, dims) = extract_vec(&shifted)?;
        let mask = Self::generate_mask_generic::<D>(&spatial_dims, kind, cutoff, order);

        // The mask is row-major over all D spatial dimensions.
        // The complex tensor has shape [..., 2*W] where the last axis is
        // interleaved (Re, Im). The outer D-1 axes index the rows.
        let w = spatial_dims[D - 1];
        let outer_count: usize = spatial_dims[..D - 1].iter().product();

        let mut out = Vec::with_capacity(vals.len());
        for outer in 0..outer_count {
            // Map flat row index to mask index.
            // Each row corresponds to a unique (D-1)-dimensional coordinate;
            // in the mask, each such coordinate spans `w` entries.
            let mask_base = outer * w;
            let row_base = outer * cw;

            for c in 0..w {
                let m = mask[mask_base + c];
                let idx = row_base + 2 * c;
                out.push(vals[idx] * m); // Re
                out.push(vals[idx + 1] * m); // Im
            }
        }
        let masked = rebuild(out, dims, &shifted);

        // Unshift → inverse FFT.
        let unshifted = FftShiftFilter::new().apply(&masked)?;
        InverseFftFilter::new().apply(&unshifted)
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        kind: FftFilterKind,
        cutoff: f64,
        order: usize,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        Self::validate_cutoff(cutoff)?;
        let spatial_dims = image.shape();
        let freq = ForwardFftFilter::new().apply_native(image, backend)?;
        let shifted = FftShiftFilter::new().apply_native(&freq, backend)?;
        let cw = shifted.shape()[D - 1];
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(&shifted)?;
        let mask = Self::generate_mask_generic::<D>(&spatial_dims, kind, cutoff, order);
        let w = spatial_dims[D - 1];
        let outer_count: usize = spatial_dims[..D - 1].iter().product();
        let mut out = Vec::with_capacity(vals.len());
        for outer in 0..outer_count {
            let mask_base = outer * w;
            let row_base = outer * cw;
            for c in 0..w {
                let m = mask[mask_base + c];
                let idx = row_base + 2 * c;
                out.push(vals[idx] * m);
                out.push(vals[idx + 1] * m);
            }
        }
        let masked = crate::native_support::rebuild_image(out, dims, &shifted, backend)?;
        let unshifted = FftShiftFilter::new().apply_native(&masked, backend)?;
        InverseFftFilter::new().apply_native(&unshifted, backend)
    }

    // ── Private helpers ────────────────────────────────────────────────

    fn validate_cutoff(cutoff: f64) -> Result<()> {
        if cutoff <= 0.0 || cutoff > 0.5 {
            bail!("FrequencyDomainFilter: cutoff must be in (0, 0.5], got {cutoff}");
        }
        Ok(())
    }

    /// Generate a D-dimensional frequency mask, dispatching on `FftFilterKind`.
    ///
    /// This is the const-generic entry point that delegates to the appropriate
    /// [`FrequencyResponse`] implementor's [`FrequencyResponse::compute_mask`] method.
    pub(crate) fn generate_mask_generic<const D: usize>(
        spatial_dims: &[usize; D],
        kind: FftFilterKind,
        cutoff: f64,
        order: usize,
    ) -> Vec<f32> {
        match kind {
            FftFilterKind::IdealLowPass => {
                IdealLowPass::compute_mask::<D>(spatial_dims, cutoff, order)
            }
            FftFilterKind::IdealHighPass => {
                IdealHighPass::compute_mask::<D>(spatial_dims, cutoff, order)
            }
            FftFilterKind::ButterworthLowPass => {
                ButterworthLowPass::compute_mask::<D>(spatial_dims, cutoff, order)
            }
            FftFilterKind::ButterworthHighPass => {
                ButterworthHighPass::compute_mask::<D>(spatial_dims, cutoff, order)
            }
        }
    }
}

impl Default for FrequencyDomainFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_frequency_filter.rs"]
mod tests;

//! Recursive Gaussian filter using the Deriche 4th-order IIR approximation
//! (matching ITK `RecursiveGaussianImageFilter` / SimpleITK
//! `SmoothingRecursiveGaussian`).
//!
//! # Mathematical Specification
//!
//! Implements a separable 4th-order recursive (IIR) Gaussian filter via the
//! Deriche (1993) approximation. For each dimension the 1-D Gaussian is the SUM
//! of a causal (forward) and an anticausal (backward) pass over the same input:
//!
//! Causal:     y_c\[n\] = Σ_{k=0..3} n_k·x\[n−k\] − Σ_{k=1..4} d_k·y_c\[n−k\]
//! Anticausal: y_a\[n\] = Σ_{k=1..4} m_k·x\[n+k\] − Σ_{k=1..4} d_k·y_a\[n+k\]
//! Output:     y\[n\] = y_c\[n\] + y_a\[n\]
//!
//! The coefficients (`n_k`, `d_k`, `m_k`) depend only on σ in pixel units and
//! are DC-normalised so the smoothing has unit gain (see [`iir::DericheCoefficients`]).
//! The interior is float-exact to SimpleITK; boundaries use constant (replicate)
//! extension.
//!
//! Derivative orders are computed by composing smoothing with finite
//! differences:
//!
//! - **Order 0 (smoothing)**: Two-pass IIR as described above.
//! - **Order 1 (first derivative)**: Smooth all axes separably, then compute
//!   gradient magnitude |∇I| = √(Σ_d (∂I/∂x_d)²) via central differences.
//! - **Order 2 (second derivative)**: Smooth all axes separably, then compute
//!   Laplacian ∇²I = Σ_d ∂²I/∂x_d² via second-order finite differences.
//!
//! Physical spacing is respected: `pixel_sigma = sigma / spacing[dim]`.
//!
//! # Complexity
//!
//! O(N) per dimension where N is the number of voxels along that axis,
//! applied separably across all D dimensions. Total: O(D · N_total).
//!
//! # References
//!
//! - Young, I.T. & van Vliet, L.J. (1995). Recursive implementation of the
//!   Gaussian filter. *Signal Processing* 44(2), pp. 139–151.
//! - van Vliet, L.J., Young, I.T., Verbeek, P.W. (1998). Recursive Gaussian
//!   derivative filters. *Proc. 14th ICPR*, pp. 509–514.

use crate::edge::GaussianSigma;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_tensor_ops::extract_vec;
use serde::{Deserialize, Serialize};

#[path = "iir.rs"]
mod iir;
use iir::*;

// ── Scale normalization enum ─────────────────────────────────────────────────

/// Whether to multiply the output by σ^order for comparable cross-scale magnitudes.
///
/// - `Skip`: no scale normalization (default for `RecursiveGaussianFilter::new`).
/// - `Normalize`: multiply output by σ^order so that responses are comparable
///   across scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScaleNormalization {
    /// No scale normalization.
    #[default]
    Skip,
    /// Multiply output by σ^order for cross-scale comparability.
    Normalize,
}

// ── Derivative order enum ─────────────────────────────────────────────────────

/// Derivative order for the recursive Gaussian filter.
///
/// Selects which derivative of the Gaussian kernel to approximate:
/// - `Zero`: Gaussian smoothing (zeroth derivative).
/// - `First`: First derivative — smoothing + gradient magnitude.
/// - `Second`: Second derivative — smoothing + Laplacian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOrder {
    /// Zeroth derivative — Gaussian smoothing.
    Zero,
    /// First derivative of Gaussian.
    First,
    /// Second derivative of Gaussian.
    Second,
}

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Recursive Gaussian filter using a 3rd-order Young–van Vliet IIR
/// approximation.
///
/// Applies a separable recursive Gaussian (or its first/second derivative)
/// along each spatial dimension of a 3-D image, respecting physical spacing.
#[derive(Debug, Clone)]
pub struct RecursiveGaussianFilter {
    /// Standard deviation of the Gaussian in physical units (mm).
    sigma: GaussianSigma,
    /// Which derivative order to approximate.
    derivative_order: DerivativeOrder,
    /// Scale normalization policy.
    scale_normalization: ScaleNormalization,
}

impl RecursiveGaussianFilter {
    /// Create a new recursive Gaussian filter with the given sigma (physical
    /// units).
    ///
    /// Defaults to smoothing (order 0), no scale normalization.
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma: GaussianSigma::new_unchecked(sigma),
            derivative_order: DerivativeOrder::Zero,
            scale_normalization: ScaleNormalization::Skip,
        }
    }

    /// Set the derivative order.
    pub fn with_derivative_order(mut self, order: DerivativeOrder) -> Self {
        self.derivative_order = order;
        self
    }

    /// Set scale normalization policy.
    pub fn with_scale_normalization(mut self, policy: ScaleNormalization) -> Self {
        self.scale_normalization = policy;
        self
    }

    /// Apply the recursive Gaussian filter to a 3-D image.
    ///
    /// Processing is separable: the 1-D IIR filter is applied along each of
    /// the three spatial axes in sequence.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (mut vals, dims) = extract_vec(image)?;

        let spacing = image.spacing();

        // Stage 1: Smooth all axes separably via the two-pass IIR
        for dim in 0..3 {
            let pixel_sigma = self.sigma.get() / spacing[dim];
            if pixel_sigma < 0.2 {
                continue;
            }
            let coeffs = DericheCoefficients::from_sigma(pixel_sigma);
            vals = apply_smooth_1d(&vals, dims, dim, &coeffs, pixel_sigma);
        }

        // Stage 2: Apply derivative operator across all axes combined.
        // Smoothing is already complete; the derivative is computed from
        // the smoothed data independently along each axis and then
        // combined (magnitude for order 1, sum for order 2).
        let sp = [spacing[0], spacing[1], spacing[2]];
        match self.derivative_order {
            DerivativeOrder::Zero => {}
            DerivativeOrder::First => {
                vals = gradient_magnitude_3d(&vals, dims, sp);
            }
            DerivativeOrder::Second => {
                vals = laplacian_3d(&vals, dims, sp);
            }
        }

        // Scale normalization: multiply by σ^order
        if let ScaleNormalization::Normalize = self.scale_normalization {
            let scale_factor = match self.derivative_order {
                DerivativeOrder::Zero => 1.0,
                DerivativeOrder::First => self.sigma.get(),
                DerivativeOrder::Second => self.sigma.get() * self.sigma.get(),
            };
            if (scale_factor - 1.0).abs() > 1e-12 {
                let sf = scale_factor as f32;
                for v in &mut vals {
                    *v *= sf;
                }
            }
        }

        let device = image.data().device();
        let out_td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}
// ── Combined derivative operators (gradient magnitude, Laplacian) ─────────────

/// Compute the gradient magnitude of a 3-D volume:
///
///   |∇I| = √(Σ_d (∂I/∂x_d / s_d)²)
///
/// Uses central differences at interior points and one-sided differences at
/// boundaries. Each component is divided by the physical spacing along that
/// axis so the result is in physical units (intensity / mm).
#[inline]
fn gradient_magnitude_3d(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = data.len();
    let mut sum_sq = vec![0.0_f32; n];
    let mut deriv_buf = vec![0.0_f32; n];
    for (dim, &s) in spacing.iter().enumerate() {
        apply_first_derivative_1d_into(data, dims, dim, &mut deriv_buf);
        let inv_s = (1.0 / s) as f32;
        for i in 0..n {
            let d = deriv_buf[i] * inv_s;
            sum_sq[i] += d * d;
        }
    }
    for v in &mut sum_sq {
        *v = v.sqrt();
    }
    sum_sq
}

/// Compute the Laplacian of a 3-D volume:
///
///   ∇²I = Σ_d ∂²I/∂x_d² / s_d²
///
/// Uses central second-order finite differences at interior points and
/// one-sided differences at boundaries. Each component is divided by the
/// squared physical spacing along that axis.
#[inline]
fn laplacian_3d(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = data.len();
    let mut result = vec![0.0_f32; n];
    let mut deriv_buf = vec![0.0_f32; n];
    for (dim, &s) in spacing.iter().enumerate() {
        apply_second_derivative_1d_into(data, dims, dim, &mut deriv_buf);
        let inv_s2 = (1.0 / (s * s)) as f32;
        for i in 0..n {
            result[i] += deriv_buf[i] * inv_s2;
        }
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_recursive_gaussian.rs"]
mod tests_recursive_gaussian;

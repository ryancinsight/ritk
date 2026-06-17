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
//! Derivative orders use the ITK/SimpleITK separable structure — the
//! corresponding-order Deriche recursion along the differentiated axis and the
//! zero-order (smoothing) recursion along the others, combined:
//!
//! - **Order 0 (smoothing)**: Two-pass IIR as described above.
//! - **Order 1 (gradient magnitude)**: for each axis `d`, first-order Deriche
//!   along `d` and zero-order along the others; `|∇I| = √(Σ_d (∂I/∂x_d / s_d)²)`.
//!   Float-exact to `GradientMagnitudeRecursiveGaussian`.
//! - **Order 2 (Laplacian)**: for each axis `d`, second-order Deriche along `d`
//!   and zero-order along the others; `∇²I = Σ_d ∂²I/∂x_d² / s_d²`. Float-exact
//!   to `LaplacianRecursiveGaussian`.
//!
//! Physical spacing is respected: `pixel_sigma = sigma / spacing[dim]`, and the
//! order-`k` term is divided by `spacing[d]^k` for the physical derivative.
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

        // Derivative order selects the separable Deriche structure that matches
        // ITK/SimpleITK: order 0 smooths all axes; orders 1 and 2 apply the
        // first/second-order Deriche recursion along each axis (zero-order along
        // the others) and combine — magnitude for order 1, sum for order 2.
        let sp = [spacing[0], spacing[1], spacing[2]];
        let sigma = self.sigma.get();
        vals = match self.derivative_order {
            DerivativeOrder::Zero => {
                for dim in 0..3 {
                    let pixel_sigma = sigma / spacing[dim];
                    if pixel_sigma < 0.2 {
                        continue;
                    }
                    let coeffs = DericheCoefficients::from_sigma(pixel_sigma);
                    vals = apply_deriche_1d(&vals, dims, dim, &coeffs, pixel_sigma);
                }
                vals
            }
            DerivativeOrder::First => gradient_magnitude_rg_vals(&vals, dims, sp, sigma),
            DerivativeOrder::Second => laplacian_rg_vals(&vals, dims, sp, sigma),
        };

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
// ── Recursive-Gaussian derivative operators (ITK structure) ───────────────────

/// `∇²(G_σ * I) = Σ_d ∂²/∂x_d²(G_σ * I)` as a flat z-major buffer. For each axis
/// `d` the volume is filtered with the **second-order** Deriche recursion along
/// `d` and the **zero-order** (smoothing) recursion along the others; each
/// per-axis second derivative is divided by `spacing[d]²` (physical units) and
/// summed. Float-exact to ITK/SimpleITK `LaplacianRecursiveGaussian`.
fn laplacian_rg_vals(vals: &[f32], dims: [usize; 3], spacing: [f64; 3], sigma: f64) -> Vec<f32> {
    let mut laplacian = vec![0.0f32; vals.len()];
    for d in 0..3 {
        let mut temp = vals.to_vec();
        for (ax, &s) in spacing.iter().enumerate() {
            let pixel_sigma = sigma / s;
            let coeffs = if ax == d {
                DericheCoefficients::second_order(pixel_sigma)
            } else {
                DericheCoefficients::from_sigma(pixel_sigma)
            };
            temp = apply_deriche_1d(&temp, dims, ax, &coeffs, pixel_sigma);
        }
        let inv_s2 = (1.0 / (spacing[d] * spacing[d])) as f32;
        for (acc, t) in laplacian.iter_mut().zip(temp.iter()) {
            *acc += t * inv_s2;
        }
    }
    laplacian
}

/// `|∇(G_σ * I)| = √(Σ_d (∂/∂x_d(G_σ * I))²)` as a flat z-major buffer. For each
/// axis `d` the volume is filtered with the **first-order** Deriche recursion
/// along `d` and the **zero-order** recursion along the others; each per-axis
/// first derivative is divided by `spacing[d]` (physical units), squared and
/// accumulated, then the square root is taken. Float-exact to ITK/SimpleITK
/// `GradientMagnitudeRecursiveGaussian`.
fn gradient_magnitude_rg_vals(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    sigma: f64,
) -> Vec<f32> {
    let mut sum_sq = vec![0.0f32; vals.len()];
    for d in 0..3 {
        let mut temp = vals.to_vec();
        for (ax, &s) in spacing.iter().enumerate() {
            let pixel_sigma = sigma / s;
            let coeffs = if ax == d {
                DericheCoefficients::first_order(pixel_sigma)
            } else {
                DericheCoefficients::from_sigma(pixel_sigma)
            };
            temp = apply_deriche_1d(&temp, dims, ax, &coeffs, pixel_sigma);
        }
        let inv_s = (1.0 / spacing[d]) as f32;
        for (acc, t) in sum_sq.iter_mut().zip(temp.iter()) {
            let g = t * inv_s;
            *acc += g * g;
        }
    }
    for v in &mut sum_sq {
        *v = v.sqrt();
    }
    sum_sq
}

/// Build an `Image` from a computed buffer, copying `src`'s spatial metadata.
fn image_from_vals<B: Backend>(src: &Image<B, 3>, vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = src.data().device();
    let out_td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(out_td, &device);
    Image::new(tensor, *src.origin(), *src.spacing(), *src.direction())
}

/// Compute `∇²(G_σ * I)` matching ITK/SimpleITK `LaplacianRecursiveGaussian`
/// (float-exact). See [`laplacian_rg_vals`].
///
/// # Errors
/// Returns `Err` if the tensor data cannot be extracted as `f32`.
pub fn laplacian_recursive_gaussian<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    let sp = image.spacing();
    let out = laplacian_rg_vals(&vals, dims, [sp[0], sp[1], sp[2]], sigma);
    Ok(image_from_vals(image, out, dims))
}

/// Compute `|∇(G_σ * I)|` matching ITK/SimpleITK `GradientMagnitudeRecursiveGaussian`
/// (float-exact). See [`gradient_magnitude_rg_vals`].
///
/// # Errors
/// Returns `Err` if the tensor data cannot be extracted as `f32`.
pub fn gradient_magnitude_recursive_gaussian<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    let sp = image.spacing();
    let out = gradient_magnitude_rg_vals(&vals, dims, [sp[0], sp[1], sp[2]], sigma);
    Ok(image_from_vals(image, out, dims))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_recursive_gaussian.rs"]
mod tests_recursive_gaussian;

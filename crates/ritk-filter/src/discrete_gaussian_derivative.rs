//! Discrete Gaussian derivative filter.
//!
//! # Mathematical Specification
//!
//! Matches `itk::DiscreteGaussianDerivativeImageFilter`. Along each axis `d` the
//! image is convolved with the 1-D `GaussianDerivativeOperator` of order
//! `order[d]`:
//!
//! ```text
//! GaussianDerivativeOperator(m) = trim_{2N}( pad_{2N-1}^{clamp}(G) âŠ› D^m )
//! ```
//!
//! where `G` is the discrete Gaussian operator ([`super::discrete_gaussian`]'s
//! Bessel kernel), `D^m` is the order-`m` central-difference derivative operator
//! (`DÂ¹ = [-Â½, 0, Â½]`, `DÂ² = [1, -2, 1]`, built by repeated convolution),
//! `N = (|D^m|-1)/2`, the Gaussian is edge-padded by `2N-1` before the
//! convolution (ITK's clamped boundary) and the full result trimmed by `2N` taps
//! each side. `use_image_spacing` enters only through the Gaussian width
//! (`pixel_variance = variance / spacingÂ²`); SimpleITK applies no extra derivative
//! norm. An order-0 axis is the plain Gaussian operator. The per-axis operators
//! are applied separably with zero-flux Neumann image boundaries (the same path as
//! `DiscreteGaussian`), float-exact to SimpleITK for voxel-unit (no-spacing)
//! variance across all derivative orders.

use super::discrete_gaussian::{convolve_separable, gaussian_operator_1d};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::rebuild;

/// Discrete Gaussian derivative filter.
#[derive(Debug, Clone)]
pub struct DiscreteGaussianDerivativeFilter {
    /// Variance (ÏƒÂ²) per axis, axis-major `[var_z, var_y, var_x]`. Scalar variance
    /// is broadcast.
    pub variance: f64,
    /// Derivative order per axis, axis-major `[order_z, order_y, order_x]`.
    pub order: [usize; 3],
    /// Maximum kernel error for the Gaussian operator (ITK default 0.01).
    pub maximum_error: f64,
    /// Divide each axis operator by `spacing^order` (ITK `UseImageSpacing`).
    pub use_image_spacing: bool,
}

impl DiscreteGaussianDerivativeFilter {
    /// Construct with explicit parameters.
    pub fn new(
        variance: f64,
        order: [usize; 3],
        maximum_error: f64,
        use_image_spacing: bool,
    ) -> Self {
        Self {
            variance,
            order,
            maximum_error,
            use_image_spacing,
        }
    }

    /// Apply the filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (tensor, origin, spacing, direction) = image.clone().into_parts();
        let dims: [usize; 3] = tensor
            .shape()
            .try_into()
            .expect("DiscreteGaussianDerivativeFilter requires rank three");

        let mut kernels: [Option<Vec<f32>>; 3] = std::array::from_fn(|_| None);
        for (d, slot) in kernels.iter_mut().enumerate() {
            let h = if self.use_image_spacing {
                spacing[d]
            } else {
                1.0
            };
            let pixel_variance = self.variance.max(0.0) / (h * h).max(1e-24);
            if pixel_variance < 1e-18 && self.order[d] == 0 {
                continue; // identity along this axis
            }
            *slot = Some(gaussian_derivative_operator_1d(
                pixel_variance,
                self.order[d],
                self.maximum_error,
            ));
        }

        if kernels.iter().all(|k| k.is_none()) {
            return Image::new(tensor, origin, spacing, direction);
        }

        let flat = tensor.to_vec();
        let result = convolve_separable(flat, dims, &kernels);
        rebuild(result, dims, image)
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (flat, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let spacing = *image.spacing();

        let mut kernels: [Option<Vec<f32>>; 3] = std::array::from_fn(|_| None);
        for (d, slot) in kernels.iter_mut().enumerate() {
            let h = if self.use_image_spacing {
                spacing[d]
            } else {
                1.0
            };
            let pixel_variance = self.variance.max(0.0) / (h * h).max(1e-24);
            if pixel_variance < 1e-18 && self.order[d] == 0 {
                continue; // identity along this axis
            }
            *slot = Some(gaussian_derivative_operator_1d(
                pixel_variance,
                self.order[d],
                self.maximum_error,
            ));
        }

        if kernels.iter().all(|k| k.is_none()) {
            return Ok(image.clone());
        }

        let result = convolve_separable(flat, dims, &kernels);
        crate::native_support::rebuild_image(result, dims, image, backend)
    }
}

/// Build the order-`m` central-difference derivative operator (`itk::Derivative-
/// Operator`): even orders fold in `[1, -2, 1]`, the odd order folds in
/// `[-Â½, 0, Â½]`, by repeated convolution. Order 0 returns `[1]`.
fn derivative_operator(order: usize) -> Vec<f64> {
    let mut coeff = vec![1.0f64];
    for _ in 0..(order / 2) {
        coeff = convolve_full(&coeff, &[1.0, -2.0, 1.0]);
    }
    if order % 2 == 1 {
        coeff = convolve_full(&coeff, &[-0.5, 0.0, 0.5]);
    }
    coeff
}

/// Full linear convolution of two coefficient vectors.
fn convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Build the 1-D `GaussianDerivativeOperator` of the given order for a pixel
/// variance: the Gaussian operator edge-padded by `2N-1` then convolved with the
/// order-`m` derivative operator, scaled by `1 / spacing^m`.
fn gaussian_derivative_operator_1d(
    pixel_variance: f64,
    order: usize,
    maximum_error: f64,
) -> Vec<f32> {
    let gauss: Vec<f64> = gaussian_operator_1d(pixel_variance, maximum_error)
        .into_iter()
        .map(f64::from)
        .collect();
    if order == 0 {
        return gauss.into_iter().map(|c| c as f32).collect();
    }
    let deriv = derivative_operator(order);
    let n = (deriv.len() - 1) / 2;
    // Edge-replicate (clamp) pad the Gaussian by 2N-1 each side, matching ITK.
    let pad = 2 * n - 1;
    let mut padded = Vec::with_capacity(gauss.len() + 2 * pad);
    let first = *gauss.first().expect("gaussian operator is non-empty");
    let last = *gauss.last().expect("gaussian operator is non-empty");
    padded.extend(std::iter::repeat_n(first, pad));
    padded.extend_from_slice(&gauss);
    padded.extend(std::iter::repeat_n(last, pad));
    let full = convolve_full(&padded, &deriv);
    // ITK's GaussianDerivativeOperator has radius `r + (n-1)` (the Gaussian radius
    // `r` plus the derivative operator's reach beyond a single tap). Padding by
    // `2n-1` then the radius-`n` derivative convolution gives a full radius of
    // `r + 3n - 1`, so trim `2n` taps each side.
    let trim = 2 * n;
    let op = &full[trim..full.len() - trim];
    // `use_image_spacing` enters only through `pixel_variance` (the Gaussian
    // width); SimpleITK applies no additional `1/spacing^order` derivative norm.
    // `convolve_separable` applies the kernel as a correlation, so its impulse
    // response is the kernel reversed â€” reverse here to apply ITK's operator.
    // (Even orders are symmetric â†’ no-op.)
    op.iter().rev().map(|&c| c as f32).collect()
}

#[cfg(test)]
#[path = "tests_discrete_gaussian_derivative.rs"]
mod tests_discrete_gaussian_derivative;

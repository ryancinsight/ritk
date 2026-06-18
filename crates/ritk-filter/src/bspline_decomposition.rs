//! B-spline decomposition filter (cubic).
//!
//! # Mathematical Specification
//!
//! Recovers the cubic B-spline interpolation coefficients `c` of an image whose
//! samples `s` satisfy `s = β³ ⊛ c` (the cubic basis `β³ = [1/6, 2/3, 1/6]`).
//! The coefficients are obtained by inverting that convolution via a causal +
//! anti-causal recursive filter with the single pole `z₁ = √3 − 2` and
//! whole-sample mirror boundary conditions, applied separably along every axis of
//! length `>= 2`. This matches `itk::BSplineDecompositionImageFilter` at spline
//! order 3 (its default).
//!
//! Delegates the numerics to the validated
//! [`ritk_interpolation::bspline_decomposition_coefficients`] used by the B-spline
//! interpolator's prefilter, so decomposition and interpolation stay consistent.

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Compute the cubic B-spline coefficient image of a 3-D image (mirror
/// boundary). The output shares the input geometry.
pub fn bspline_decomposition<B: Backend>(image: &Image<B, 3>) -> Result<Image<B, 3>> {
    let (vals, dims) = extract_vec_infallible(image);
    let coeffs = ritk_interpolation::bspline_decomposition_coefficients(&vals, &dims);
    Ok(rebuild(coeffs, dims, image))
}

#[cfg(test)]
#[path = "tests_bspline_decomposition.rs"]
mod tests_bspline_decomposition;

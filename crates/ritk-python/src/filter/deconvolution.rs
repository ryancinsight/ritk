//! Python bindings for image deconvolution / restoration filters (GAP-262-FLT-02).
//!
//! All four deconvolution filters operate natively on 3-D images via their
//! `apply_3d` methods. There is no single-slice restriction. Single-slice
//! images (shape `[1, H, W]`) are handled without special-casing.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    LandweberDeconvolution, RichardsonLucyDeconvolution, TikhonovDeconvolution, WienerDeconvolution,
};

/// Apply Wiener deconvolution to a 3-D image.
///
/// Restores a degraded image given the PSF kernel and noise-to-signal ratio K.
///
/// Matches `SimpleITK.WienerDeconvolution`. In the frequency domain:
/// ```text
/// U(ω) = G(ω) · H*(ω) / ( |H(ω)|² + Pn / (|G(ω)|² − Pn) )
/// ```
///
/// Args:
///     image: Degraded PyImage (any shape [Z, Y, X]).
///     kernel: 3-D PSF kernel PyImage (shape [kz, ky, kx]).
///     noise_variance: Noise power spectral density Pn (sitk noiseVariance,
///                     default 0.01). The regularisation is frequency-adaptive,
///                     using the input power spectrum to estimate the signal.
///
/// Returns:
///     Restored PyImage with the same shape as `image`.
#[pyfunction]
#[pyo3(signature = (image, kernel, noise_variance=0.01_f32))]
pub fn wiener_deconvolution(
    py: Python<'_>,
    image: &PyImage,
    kernel: &PyImage,
    noise_variance: f32,
) -> RitkResult<PyImage> {
    let img_ref = image.inner.clone();
    let ker_ref = kernel.inner.clone();
    let result = py.allow_threads(|| {
        WienerDeconvolution::new(noise_variance)
            .apply(img_ref.as_ref(), ker_ref.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply Tikhonov-regularized deconvolution to a 3-D image.
///
/// Matches `SimpleITK.TikhonovDeconvolution` — a constant-regularised inverse
/// filter. In the frequency domain:
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ)
/// ```
///
/// Args:
///     image: Degraded PyImage (any shape [Z, Y, X]).
///     kernel: 3-D PSF kernel PyImage.
///     lambda: Regularization parameter λ (default 0.01).
///
/// Returns:
///     Restored PyImage with the same shape as `image`.
#[pyfunction]
#[pyo3(signature = (image, kernel, lambda=0.01_f32))]
pub fn tikhonov_deconvolution(
    py: Python<'_>,
    image: &PyImage,
    kernel: &PyImage,
    lambda: f32,
) -> RitkResult<PyImage> {
    let img_ref = image.inner.clone();
    let ker_ref = kernel.inner.clone();
    let result = py.allow_threads(|| {
        TikhonovDeconvolution::new(lambda)
            .apply(img_ref.as_ref(), ker_ref.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply Richardson-Lucy iterative deconvolution to a 3-D image.
///
/// Expectation-maximization algorithm for Poisson-noise restoration:
///
/// ```text
/// u₀ = g
/// uₖ₊₁ = uₖ · (h* ⋆ (g / (h ⋆ uₖ)))
/// ```
///
/// Preserves non-negativity and total flux.
///
/// Args:
///     image: Degraded PyImage (any shape [Z, Y, X]).
///     kernel: 3-D PSF kernel PyImage.
///     max_iterations: Maximum EM iterations (default 30).
///     tolerance: Convergence tolerance for relative change (default 1e-6).
///
/// Returns:
///     Restored PyImage with the same shape as `image`.
#[pyfunction]
#[pyo3(signature = (image, kernel, max_iterations=30_usize, tolerance=1e-6_f32))]
pub fn richardson_lucy_deconvolution(
    py: Python<'_>,
    image: &PyImage,
    kernel: &PyImage,
    max_iterations: usize,
    tolerance: f32,
) -> RitkResult<PyImage> {
    let img_ref = image.inner.clone();
    let ker_ref = kernel.inner.clone();
    let result = py.allow_threads(|| {
        RichardsonLucyDeconvolution::new()
            .with_max_iterations(max_iterations)
            .with_tolerance(tolerance)
            .apply(img_ref.as_ref(), ker_ref.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply Landweber iterative deconvolution to a 3-D image.
///
/// Gradient descent minimization of `||g − h ∗ u||²`:
///
/// ```text
/// u₀ = g
/// uₖ₊₁ = uₖ + α · h* ⋆ (g − h ⋆ uₖ)
/// ```
///
/// Args:
///     image: Degraded PyImage (any shape [Z, Y, X]).
///     kernel: 3-D PSF kernel PyImage.
///     step_size: Gradient descent step size α (default 0.1).
///     max_iterations: Maximum iterations (default 100).
///     tolerance: Convergence tolerance (default 1e-6).
///
/// Returns:
///     Restored PyImage with the same shape as `image`.
#[pyfunction]
#[pyo3(signature = (image, kernel, step_size=0.1_f32, max_iterations=100_usize, tolerance=1e-6_f32))]
pub fn landweber_deconvolution(
    py: Python<'_>,
    image: &PyImage,
    kernel: &PyImage,
    step_size: f32,
    max_iterations: usize,
    tolerance: f32,
) -> RitkResult<PyImage> {
    let img_ref = image.inner.clone();
    let ker_ref = kernel.inner.clone();
    let result = py.allow_threads(|| {
        LandweberDeconvolution::new()
            .with_step_size(step_size)
            .with_max_iterations(max_iterations)
            .with_tolerance(tolerance)
            .apply(img_ref.as_ref(), ker_ref.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

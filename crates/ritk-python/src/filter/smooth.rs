//! Smoothing and diffusion filters: Gaussian, median, bilateral, N4, anisotropic diffusion.
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::bias::N4Config;
use ritk_core::filter::diffusion::{CoherenceConfig, CurvatureConfig};
use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
use ritk_core::filter::recursive_gaussian::DerivativeOrder;
use ritk_core::filter::{
    AnisotropicDiffusionFilter, BilateralFilter, BinShrinkImageFilter,
    CoherenceEnhancingDiffusionFilter, CurvatureAnisotropicDiffusionFilter, DiscreteGaussianFilter,
    GaussianFilter, MedianFilter, N4BiasFieldCorrectionFilter, RecursiveGaussianFilter,
};

/// Apply Gaussian smoothing to an image.
///
/// Uses `ritk_core::filter::GaussianFilter` (separable 1-D convolutions,
/// NdArray backend, CPU-only). The same sigma is applied along all three
/// axes (isotropic smoothing).
///
/// Args:
///     image: Input PyImage.
///     sigma: Standard deviation in physical units (mm).
///         The pixel-space sigma is computed as `sigma / spacing[d]`.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, sigma))]
pub fn gaussian_filter(py: Python<'_>, image: &PyImage, sigma: f64) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GaussianFilter::<Backend>::new(vec![sigma, sigma, sigma]);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Apply ITK-style discrete Gaussian smoothing parameterized by variance.
///
/// Uses `ritk_core::filter::DiscreteGaussianFilter` with separable 1-D
/// convolution, replicate padding, analytic kernel truncation from
/// `maximum_error`, and optional spacing-aware sigma conversion.
///
/// Args:
///     image: Input PyImage.
///     variance: Gaussian variance σ² in physical units².
///     maximum_error: Kernel truncation tolerance in (0, 1) (default 0.01).
///     use_image_spacing: If True, convert physical σ to pixel σ using image
///         spacing (default True).
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, variance, maximum_error=0.01, use_image_spacing=true))]
pub fn discrete_gaussian(
    py: Python<'_>,
    image: &PyImage,
    variance: f64,
    maximum_error: f64,
    use_image_spacing: bool,
) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = DiscreteGaussianFilter::<Backend>::new(vec![variance])
            .with_maximum_error(maximum_error)
            .with_use_image_spacing(use_image_spacing);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Apply a median (rank) filter for impulse-noise removal.
///
/// For each voxel the output is the median of all voxels within the
/// axis-aligned cube of half-width `radius` voxels. Out-of-bounds positions
/// are clamped to the nearest valid voxel (replicate padding).
///
/// Args:
///     image: Input PyImage.
///     radius: Neighbourhood half-width in voxels (default 1 → 3×3×3 cube).
///         The kernel contains `(2*radius + 1)^3` samples.
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn median_filter(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = MedianFilter::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply a bilateral filter (edge-preserving smoothing).
///
/// Each output voxel is a weighted average of its neighbourhood, where the
/// weight combines:
/// - A **spatial** Gaussian: `exp(−||p − q||² / (2 σ_s²))`
/// - A **range** Gaussian: `exp(−(I(p) − I(q))² / (2 σ_r²))`
///
/// Args:
///     image: Input PyImage.
///     spatial_sigma: Spatial Gaussian sigma in voxels.
///     range_sigma: Intensity range sigma (same units as voxel values).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
pub fn bilateral_filter(
    py: Python<'_>,
    image: &PyImage,
    spatial_sigma: f64,
    range_sigma: f64,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = BilateralFilter::new(spatial_sigma, range_sigma);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply N4 bias field correction to an MRI image.
///
/// Corrects low-frequency multiplicative intensity inhomogeneity caused by
/// RF coil non-uniformity. Based on Tustison et al. (2010),
/// *IEEE Trans. Med. Imaging* 29(6):1310–1320.
///
/// Args:
///     image: Input PyImage (must be f32, values > 0).
///     num_fitting_levels: Number of B-spline refinement levels (default 4).
///     num_iterations: Iterations per level (default 50).
///     noise_estimate: Fraction of intensity range modelling noise/bias spread
///         (default 0.01 for typical MRI; use 0.05–0.10 for
///         images with large bias fields).
///
/// Returns:
///     Bias-corrected PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, num_fitting_levels=4, num_iterations=50, noise_estimate=0.01))]
pub fn n4_bias_correction(
    py: Python<'_>,
    image: &PyImage,
    num_fitting_levels: usize,
    num_iterations: usize,
    noise_estimate: f64,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let config = N4Config {
            num_fitting_levels,
            num_iterations,
            noise_estimate,
            ..Default::default()
        };
        let filter = N4BiasFieldCorrectionFilter::new(config);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
///
/// Reduces noise while preserving edges via the PDE:
/// ∂I/∂t = div(c(|∇I|) · ∇I)
///
/// Args:
///     image: Input PyImage.
///     iterations: Number of explicit Euler time steps (default 20).
///     conductance: Edge-stopping parameter K (default 3.0; larger = more smoothing).
///     time_step: Euler step size Δt (default 0.0625; must be ≤ 1/6 for 3-D stability).
///     exponential: If True, use exponential conductance c(s)=exp(-(s/K)²);
///         if False, use quadratic c(s)=1/(1+(s/K)²) (default True).
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, conductance=3.0, time_step=0.0625, exponential=true))]
pub fn anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    conductance: f64,
    time_step: f64,
    exponential: bool,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let function = if exponential {
            ConductanceFunction::Exponential
        } else {
            ConductanceFunction::Quadratic
        };
        let config = DiffusionConfig {
            num_iterations: iterations,
            conductance: conductance as f32,
            time_step: time_step as f32,
            function,
        };
        let filter = AnisotropicDiffusionFilter::new(config);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply curvature anisotropic diffusion (Alvarez et al. 1992).
///
/// Evolves image level sets by mean curvature motion:
/// ∂I/∂t = |∇I| · div(∇I / |∇I|) = |∇I| · κ
///
/// Args:
///     image: Input PyImage.
///     iterations: Number of explicit Euler time steps (default 20).
///     time_step: Euler Δt (default 0.0625; stability requires Δt ≤ 1/6 for unit spacing).
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, time_step=0.0625))]
pub fn curvature_anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    time_step: f64,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply a recursive Gaussian (Young–van Vliet 3rd-order IIR) filter.
///
/// Separable IIR approximation of the Gaussian and its first/second
/// derivatives. Constant-time per voxel regardless of σ (no explicit kernel
/// construction).
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm, default 1.0).
///     order: Derivative order — 0 = smoothing, 1 = first derivative,
///         2 = second derivative (default 0).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     ValueError: if `order` is not in {0, 1, 2}.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, order=0))]
pub fn recursive_gaussian(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    order: usize,
) -> RitkResult<PyImage> {
    let derivative_order = match order {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::First,
        2 => DerivativeOrder::Second,
        _ => {
            return Err(RitkPyError::value(format!(
                "recursive_gaussian: order must be 0, 1, or 2, got {order}"
            )));
        }
    };
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = RecursiveGaussianFilter::new(sigma).with_derivative_order(derivative_order);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply coherence-enhancing diffusion (Weickert 1999).
///
/// Anisotropic diffusion that smooths along coherent structures (edges,
/// ridges) while preserving them across the structure orientation. Uses the
/// structure tensor to drive diffusion.
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian sigma for structure tensor smoothing (integration scale, default 3.0).
///     contrast: Contrast parameter C (default 1e-10).
///     alpha: Smoothing parameter in flat regions (default 0.001).
///     time_step: Euler step Δt (default 0.0625).
///     iterations: Number of iterations (default 10).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=3.0, contrast=1e-10, alpha=0.001, time_step=0.0625, iterations=10))]
pub fn coherence_enhancing_diffusion(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    contrast: f64,
    alpha: f64,
    time_step: f64,
    iterations: usize,
) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let config = CoherenceConfig {
            sigma,
            contrast,
            alpha,
            time_step,
            n_iterations: iterations,
        };
        let filter = CoherenceEnhancingDiffusionFilter::new(config);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Apply bin-shrink downsampling (integer sub-sampling by bin averaging).
///
/// Reduces image dimensions by integer factors by averaging all voxels
/// within each non-overlapping bin. This provides anti-aliasing compared
/// to naive sub-sampling (which just takes every Nth voxel).
///
/// Output shape[d] = floor(input_shape[d] / factor[d]).
/// Spacing is multiplied by the shrink factor.
///
/// Args:
///     image: Input PyImage.
///     factor_z: Shrink factor along Z axis (default 2).
///     factor_y: Shrink factor along Y axis (default 2).
///     factor_x: Shrink factor along X axis (default 2).
///
/// Returns:
///     Downsampled PyImage with reduced shape and scaled spacing.
#[pyfunction]
#[pyo3(signature = (image, factor_z=2, factor_y=2, factor_x=2))]
pub fn bin_shrink(
    py: Python<'_>,
    image: &PyImage,
    factor_z: usize,
    factor_y: usize,
    factor_x: usize,
) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = BinShrinkImageFilter::new(vec![factor_z, factor_y, factor_x]);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

//! Python-exposed image filtering functions.
//!
//! All filters delegate to `ritk-core::filter` implementations (SSOT).
//!
//! # Module structure
//! - `gaussian_filter`:       Separable Gaussian smoothing.
//! - `median_filter`:         Sliding-window median (rank filter).
//! - `bilateral_filter`:      Bilateral filter (edge-preserving smoothing).
//! - `n4_bias_correction`:    N4 bias field correction (Tustison 2010).
//! - `anisotropic_diffusion`: Perona-Malik anisotropic diffusion.
//! - `gradient_magnitude`:    Gradient magnitude via central differences.
//! - `laplacian`:             Discrete Laplacian filter.
//! - `frangi_vesselness`:     Frangi multiscale vesselness filter.
//! - `canny_edge_detect`:     Canny edge detector (Gaussian smooth → gradient → NMS → hysteresis).
//! - `laplacian_of_gaussian`: Laplacian of Gaussian (LoG) blob/edge detector.
//! - `recursive_gaussian`:    Young–van Vliet recursive Gaussian (IIR, order 0/1/2).
//! - `sobel_gradient`:        Sobel gradient magnitude.
//! - `grayscale_erosion`:     Grayscale morphological erosion (flat cubic SE).
//! - `grayscale_dilation`:    Grayscale morphological dilation (flat cubic SE).
//! - `curvature_anisotropic_diffusion`: Curvature anisotropic diffusion (Alvarez 1992).
//! - `sato_line_filter`:    Sato multi-scale line filter (Sato 1998).

use crate::image::{into_py_image, Backend, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::bias::N4Config;
use ritk_core::filter::diffusion::CurvatureConfig;
use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
use ritk_core::filter::recursive_gaussian::DerivativeOrder;
use ritk_core::filter::vesselness::FrangiConfig;
use ritk_core::filter::vesselness::SatoConfig;
use ritk_core::filter::{
    AnisotropicDiffusionFilter, BilateralFilter, BinaryThresholdImageFilter, CannyEdgeDetector,
    CurvatureAnisotropicDiffusionFilter, DiscreteGaussianFilter, FrangiVesselnessFilter,
    GaussianFilter, GradientMagnitudeFilter, GrayscaleDilation, GrayscaleErosion,
    IntensityWindowingFilter, LaplacianFilter, LaplacianOfGaussianFilter, MedianFilter,
    N4BiasFieldCorrectionFilter, RecursiveGaussianFilter, RescaleIntensityFilter, SatoLineFilter,
    SigmoidImageFilter, SobelFilter, ThresholdImageFilter,
};

use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::filter::{
    LabelClosing, LabelErosion, LabelOpening, MorphologicalReconstruction, ReconstructionMode,
    ResampleImageFilter,
};
use ritk_core::interpolation::linear::LinearInterpolator;
use ritk_core::interpolation::{
    BSplineInterpolator, Lanczos4Interpolator, NearestNeighborInterpolator,
};
use ritk_core::spatial::Spacing as CoreSpacing;
use ritk_core::transform::translation::TranslationTransform;

// ── gaussian_filter ───────────────────────────────────────────────────────────

/// Apply Gaussian smoothing to an image.
///
/// Uses `ritk_core::filter::GaussianFilter` (separable 1-D convolutions,
/// NdArray backend, CPU-only).  The same sigma is applied along all three
/// axes (isotropic smoothing).
///
/// Args:
///     image: Input PyImage.
///     sigma: Standard deviation in physical units (mm).
///            The pixel-space sigma is computed as `sigma / spacing[d]`.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma))]
pub fn gaussian_filter(py: Python<'_>, image: &PyImage, sigma: f64) -> PyResult<PyImage> {
    // All three axes use the same physical sigma; GaussianFilter scales each
    // axis by the corresponding spacing value internally.
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GaussianFilter::<Backend>::new(vec![sigma, sigma, sigma]);
        filter.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── discrete_gaussian ─────────────────────────────────────────────────────────

/// Apply ITK-style discrete Gaussian smoothing parameterized by variance.
///
/// Uses `ritk_core::filter::DiscreteGaussianFilter` with separable 1-D
/// convolution, replicate padding, analytic kernel truncation from
/// `maximum_error`, and optional spacing-aware sigma conversion.
///
/// Args:
///     image:             Input PyImage.
///     variance:          Gaussian variance σ² in physical units².
///     maximum_error:     Kernel truncation tolerance in (0, 1) (default 0.01).
///     use_image_spacing: If True, convert physical σ to pixel σ using image
///                        spacing (default True).
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
#[pyo3(signature = (image, variance, maximum_error=0.01, use_image_spacing=true))]
pub fn discrete_gaussian(
    py: Python<'_>,
    image: &PyImage,
    variance: f64,
    maximum_error: f64,
    use_image_spacing: bool,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = DiscreteGaussianFilter::<Backend>::new(vec![variance])
            .with_maximum_error(maximum_error)
            .with_use_image_spacing(use_image_spacing);
        filter.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── median_filter ─────────────────────────────────────────────────────────────

/// Apply a median (rank) filter for impulse-noise removal.
///
/// For each voxel the output is the median of all voxels within the
/// axis-aligned cube of half-width `radius` voxels.  Out-of-bounds positions
/// are clamped to the nearest valid voxel (replicate padding).
///
/// Args:
///     image:  Input PyImage.
///     radius: Neighbourhood half-width in voxels (default 1 → 3×3×3 cube).
///             The kernel contains `(2*radius + 1)^3` samples.
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal tensor operation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn median_filter(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = MedianFilter::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply a bilateral filter (edge-preserving smoothing).
///
/// Each output voxel is a weighted average of its neighbourhood, where the
/// weight combines:
/// - A **spatial** Gaussian: `exp(−||p − q||² / (2 σ_s²))`
/// - A **range**   Gaussian: `exp(−(I(p) − I(q))² / (2 σ_r²))`
///
/// The neighbourhood radius is `ceil(3 * spatial_sigma)` voxels.
/// Out-of-bounds positions are skipped (effectively zero-padded in weight, but
/// only present voxels contribute, so the estimator remains unbiased).
///
/// Args:
///     image:         Input PyImage.
///     spatial_sigma: Spatial Gaussian sigma in voxels.
///     range_sigma:   Intensity range sigma (same units as voxel values).
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
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = BilateralFilter::new(spatial_sigma, range_sigma);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── n4_bias_correction ────────────────────────────────────────────────────────

/// Apply N4 bias field correction to an MRI image.
///
/// Corrects low-frequency multiplicative intensity inhomogeneity caused by
/// RF coil non-uniformity.  Based on Tustison et al. (2010),
/// *IEEE Trans. Med. Imaging* 29(6):1310–1320.
///
/// Args:
///     image:              Input PyImage (must be f32, values > 0).
///     num_fitting_levels: Number of B-spline refinement levels (default 4).
///     num_iterations:     Iterations per level (default 50).
///     noise_estimate:     Fraction of intensity range modelling noise/bias spread
///                         (default 0.01 for typical MRI; use 0.05–0.10 for
///                         images with large bias fields).
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
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let config = N4Config {
            num_fitting_levels,
            num_iterations,
            noise_estimate,
            ..Default::default()
        };
        let filter = N4BiasFieldCorrectionFilter::new(config);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── anisotropic_diffusion ─────────────────────────────────────────────────────

/// Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
///
/// Reduces noise while preserving edges via the PDE:
///   ∂I/∂t = div(c(|∇I|) · ∇I)
///
/// Args:
///     image:       Input PyImage.
///     iterations:  Number of explicit Euler time steps (default 20).
///     conductance: Edge-stopping parameter K (default 3.0; larger = more smoothing).
///     time_step:   Euler step size Δt (default 0.0625; must be ≤ 1/6 for 3-D stability).
///     exponential: If True, use exponential conductance c(s)=exp(-(s/K)²);
///                  if False, use quadratic c(s)=1/(1+(s/K)²) (default True).
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
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
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
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── gradient_magnitude ────────────────────────────────────────────────────────

/// Compute the gradient magnitude |∇I| via central finite differences.
///
/// Each gradient component is divided by the corresponding physical spacing so
/// the result is in (intensity / mm) units.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of gradient magnitudes, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
pub fn gradient_magnitude(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = GradientMagnitudeFilter::new([spacing[0], spacing[1], spacing[2]]);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── laplacian ─────────────────────────────────────────────────────────────────

/// Compute the discrete Laplacian ∇²I = ∂²I/∂z² + ∂²I/∂y² + ∂²I/∂x².
///
/// Uses second-order central finite differences with the image's physical
/// spacing, so the result is in (intensity / mm²) units.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of Laplacian values, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
pub fn laplacian(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = LaplacianFilter::new([spacing[0], spacing[1], spacing[2]]);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── frangi_vesselness ─────────────────────────────────────────────────────────

/// Apply the Frangi multiscale vesselness filter.
///
/// Detects tubular structures (blood vessels, airways) by analysing Hessian
/// eigenvalues at multiple spatial scales.  Reference: Frangi et al. (1998),
/// *MICCAI* LNCS 1496:130–137.
///
/// Args:
///     image:          Input PyImage (should be pre-smoothed for noisy data).
///     scales:         List of σ values in mm (default [0.5, 1.0, 2.0]).
///     alpha:          Plate-vs-line anisotropy parameter (default 0.5).
///     beta:           Blobness parameter (default 0.5).
///     gamma:          Noise-suppression structureness threshold (default 15.0).
///     bright_vessels: If True, detect bright tubes on dark background (default True).
///
/// Returns:
///     PyImage of vesselness values in [0, 1], same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, beta=0.5, gamma=15.0, bright_vessels=true))]
pub fn frangi_vesselness(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    bright_vessels: bool,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let config = FrangiConfig {
            scales: scales.unwrap_or_else(|| vec![1.0, 2.0, 3.0]),
            alpha,
            beta,
            gamma,
            bright_vessels,
        };
        let filter = FrangiVesselnessFilter::new(config);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── canny_edge_detect ─────────────────────────────────────────────────────────

/// Apply the Canny edge detector to an image.
///
/// Pipeline: Gaussian smoothing (σ) → gradient magnitude → non-maximum
/// suppression → double-threshold hysteresis.  Reference: Canny, J. (1986),
/// *IEEE Trans. PAMI* 8(6):679–698.
///
/// Args:
///     image:          Input PyImage.
///     sigma:          Gaussian pre-smoothing σ in physical units (mm, default 1.0).
///     low_threshold:  Lower hysteresis threshold on gradient magnitude (default 0.1).
///     high_threshold: Upper hysteresis threshold on gradient magnitude (default 0.2).
///
/// Returns:
///     Binary edge PyImage (1.0 = edge, 0.0 = non-edge), same shape and metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, low_threshold=0.1, high_threshold=0.2))]
pub fn canny_edge_detect(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    low_threshold: f64,
    high_threshold: f64,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = CannyEdgeDetector::new(sigma, low_threshold, high_threshold);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── laplacian_of_gaussian ─────────────────────────────────────────────────────

/// Apply the Laplacian of Gaussian (LoG) filter.
///
/// Computes ∇²(G_σ * I) by first applying separable Gaussian smoothing with
/// standard deviation σ, then computing the discrete Laplacian via
/// second-order finite differences.  Useful for blob detection and
/// zero-crossing edge detection (Marr & Hildreth 1980).
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm, default 1.0).
///
/// Returns:
///     PyImage of LoG values, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0))]
pub fn laplacian_of_gaussian(py: Python<'_>, image: &PyImage, sigma: f64) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = LaplacianOfGaussianFilter::new(sigma);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── recursive_gaussian ────────────────────────────────────────────────────────

/// Apply a recursive Gaussian (Young–van Vliet 3rd-order IIR) filter.
///
/// Separable IIR approximation of the Gaussian and its first/second
/// derivatives.  Constant-time per voxel regardless of σ (no explicit kernel
/// construction).
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm, default 1.0).
///     order: Derivative order — 0 = smoothing, 1 = first derivative,
///            2 = second derivative (default 0).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     ValueError:    if `order` is not in {0, 1, 2}.
///     RuntimeError:  on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, order=0))]
pub fn recursive_gaussian(image: &PyImage, sigma: f64, order: usize) -> PyResult<PyImage> {
    let derivative_order = match order {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::First,
        2 => DerivativeOrder::Second,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "recursive_gaussian: order must be 0, 1, or 2, got {order}"
            )));
        }
    };
    let filter = RecursiveGaussianFilter::new(sigma).with_derivative_order(derivative_order);
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── sobel_gradient ────────────────────────────────────────────────────────────

/// Compute the Sobel gradient magnitude of an image.
///
/// Applies the 3×3×3 Sobel operator along each axis, scaled by the image's
/// physical spacing, then returns the Euclidean magnitude of the gradient
/// vector.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of gradient magnitudes in (intensity / mm) units, same shape
///     and metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
pub fn sobel_gradient(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = SobelFilter::new([spacing[0], spacing[1], spacing[2]]);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── grayscale_erosion ─────────────────────────────────────────────────────────

/// Apply grayscale morphological erosion with a flat cubic structuring element.
///
/// Each output voxel is the minimum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Eroded PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GrayscaleErosion::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── grayscale_dilation ────────────────────────────────────────────────────────

/// Apply grayscale morphological dilation with a flat cubic structuring element.
///
/// Each output voxel is the maximum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Dilated PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GrayscaleDilation::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── curvature_anisotropic_diffusion ───────────────────────────────────────────

/// Apply curvature anisotropic diffusion (Alvarez et al. 1992).
///
/// Evolves image level sets by mean curvature motion:
///   ∂I/∂t = |∇I| · div(∇I / |∇I|) = |∇I| · κ
///
/// Args:
///     image:      Input PyImage.
///     iterations: Number of explicit Euler time steps (default 20).
///     time_step:  Euler Δt (default 0.0625; stability requires Δt ≤ 1/6 for unit spacing).
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
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── sato_line_filter ──────────────────────────────────────────────────────────

/// Apply the Sato multi-scale line filter for curvilinear structure detection.
///
/// Detects tubular structures using multi-scale Hessian eigenvalue analysis
/// (Sato et al. 1998). The output is the per-voxel maximum response over all scales.
///
/// Args:
///     image:       Input PyImage.
///     scales:      List of Gaussian σ values (physical units, mm). Default [1.0, 2.0, 3.0].
///     alpha:       Cross-section anisotropy exponent [0.5, 2.0]. Default 0.5.
///     bright_tubes: If True detect bright tubes on dark background (default True).
///
/// Returns:
///     Line-enhanced PyImage with identical shape and metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, bright_tubes=true))]
pub fn sato_line_filter(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    bright_tubes: bool,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = SatoLineFilter::new(SatoConfig {
            scales: scales.unwrap_or_else(|| vec![1.0, 2.0, 3.0]),
            alpha,
            bright_tubes,
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── Submodule registration ────────────────────────────────────────────────────
// ── Submodule registration ────────────────────────────────────────────────────

// -- rescale_intensity ----------------------------------------------------------

/// Linearly rescale image intensity to [out_min, out_max].
///
/// output(x) = (I(x) - I_min) / (I_max - I_min) * (out_max - out_min) + out_min
///
/// Args:
///     image:   Input PyImage.
///     out_min: Minimum output value (default 0.0).
///     out_max: Maximum output value (default 1.0).
///
/// Returns:
///     Rescaled PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, out_min=0.0_f32, out_max=1.0_f32))]
pub fn rescale_intensity(
    py: Python<'_>,
    image: &PyImage,
    out_min: f32,
    out_max: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = RescaleIntensityFilter::new(out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Clamp to [window_min, window_max] then rescale to [out_min, out_max].
///
/// Pixels below window_min map to out_min; pixels above window_max map to out_max.
///
/// Args:
///     image:      Input PyImage.
///     window_min: Lower intensity window bound.
///     window_max: Upper intensity window bound.
///     out_min:    Minimum output value (default 0.0).
///     out_max:    Maximum output value (default 1.0).
#[pyfunction]
#[pyo3(signature = (image, window_min, window_max, out_min=0.0_f32, out_max=1.0_f32))]
pub fn intensity_windowing(
    py: Python<'_>,
    image: &PyImage,
    window_min: f32,
    window_max: f32,
    out_min: f32,
    out_max: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = IntensityWindowingFilter::new(window_min, window_max, out_min, out_max);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels strictly below threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_below(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::below(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels strictly above threshold to outside_value; keep others unchanged.
#[pyfunction]
#[pyo3(signature = (image, threshold, outside_value=0.0_f32))]
pub fn threshold_above(
    py: Python<'_>,
    image: &PyImage,
    threshold: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::above(threshold, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Set pixels outside [lower, upper] to outside_value; keep interior pixels unchanged.
#[pyfunction]
#[pyo3(signature = (image, lower, upper, outside_value=0.0_f32))]
pub fn threshold_outside(
    py: Python<'_>,
    image: &PyImage,
    lower: f32,
    upper: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = ThresholdImageFilter::outside(lower, upper, outside_value);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Sigmoid intensity transform: output = (max-min)/(1+exp(-(I-alpha)/beta)) + min.
///
/// Args:
///     image:      Input PyImage.
///     alpha:      Inflection point (input value where output = (max+min)/2).
///     beta:       Transition width (larger = more gradual sigmoid).
///     min_output: Minimum output value (default 0.0).
///     max_output: Maximum output value (default 1.0).
#[pyfunction]
#[pyo3(signature = (image, alpha, beta, min_output=0.0_f32, max_output=1.0_f32))]
pub fn sigmoid_filter(
    py: Python<'_>,
    image: &PyImage,
    alpha: f32,
    beta: f32,
    min_output: f32,
    max_output: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = SigmoidImageFilter::new(alpha, beta, min_output, max_output);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Binary threshold: foreground if I in [lower_threshold, upper_threshold], else background.
///
/// Args:
///     image:            Input PyImage.
///     lower_threshold:  Inclusive lower bound.
///     upper_threshold:  Inclusive upper bound.
///     foreground:       Value for pixels inside the interval (default 1.0).
///     background:       Value for pixels outside the interval (default 0.0).
#[pyfunction]
#[pyo3(signature = (image, lower_threshold, upper_threshold, foreground=1.0_f32, background=0.0_f32))]
pub fn binary_threshold(
    py: Python<'_>,
    image: &PyImage,
    lower_threshold: f32,
    upper_threshold: f32,
    foreground: f32,
    background: f32,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = BinaryThresholdImageFilter::new(
            lower_threshold,
            upper_threshold,
            foreground,
            background,
        );
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── label_erosion ──────────────────────────────────────────────────────────────

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelErosion::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── label_opening ─────────────────────────────────────────────────────────────

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelOpening::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── label_closing ─────────────────────────────────────────────────────────────

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelClosing::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── morphological_reconstruction ─────────────────────────────────────────────

/// Geodesic morphological reconstruction.
#[pyfunction]
#[pyo3(signature = (marker, mask, mode = "dilation"))]
pub fn morphological_reconstruction(
    py: Python<'_>,
    marker: &PyImage,
    mask: &PyImage,
    mode: &str,
) -> PyResult<PyImage> {
    let recon_mode = match mode {
        "dilation" => ReconstructionMode::Dilation,
        "erosion" => ReconstructionMode::Erosion,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown reconstruction mode '{}'. Use 'dilation' or 'erosion'.",
                other
            )))
        }
    };
    let marker_arc = std::sync::Arc::clone(&marker.inner);
    let mask_arc = std::sync::Arc::clone(&mask.inner);
    let result = py.allow_threads(|| {
        MorphologicalReconstruction::new(recon_mode)
            .apply(marker_arc.as_ref(), mask_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── resample_image ────────────────────────────────────────────────────────────

/// Resample a 3-D image to new voxel spacing.
///
/// Output size N_d_prime = max(1, round(N_d * S_d / S_d_prime)).
#[pyfunction]
#[pyo3(signature = (image, spacing_z=1.0_f64, spacing_y=1.0_f64, spacing_x=1.0_f64, mode="linear"))]
pub fn resample_image(
    py: Python<'_>,
    image: &PyImage,
    spacing_z: f64,
    spacing_y: f64,
    spacing_x: f64,
    mode: &str,
) -> PyResult<PyImage> {
    if spacing_z <= 0.0 || spacing_y <= 0.0 || spacing_x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "spacing values must be positive, got ({spacing_z},{spacing_y},{spacing_x})"
        )));
    }
    let mode = mode.to_string();
    let inner = std::sync::Arc::clone(&image.inner);

    let result = py
        .allow_threads(move || -> Result<_, String> {
            let orig_dims = inner.shape();
            let orig_sp = *inner.spacing();
            let orig_orig = *inner.origin();
            let orig_dir = *inner.direction();

            let new_nz = ((orig_dims[0] as f64 * orig_sp[0]) / spacing_z)
                .round()
                .max(1.0) as usize;
            let new_ny = ((orig_dims[1] as f64 * orig_sp[1]) / spacing_y)
                .round()
                .max(1.0) as usize;
            let new_nx = ((orig_dims[2] as f64 * orig_sp[2]) / spacing_x)
                .round()
                .max(1.0) as usize;

            let new_sp = CoreSpacing::new([spacing_z, spacing_y, spacing_x]);
            let device: <Backend as BurnBackend>::Device = Default::default();
            let zero_t = Tensor::<Backend, 1>::from_data(
                TensorData::new(vec![0.0f32; 3], Shape::new([3])),
                &device,
            );

            match mode.as_str() {
                "nearest" => Ok(ResampleImageFilter::new(
                    [new_nz, new_ny, new_nx],
                    orig_orig,
                    new_sp,
                    orig_dir,
                    TranslationTransform::<Backend, 3>::new(zero_t),
                    NearestNeighborInterpolator::new(),
                )
                .apply(inner.as_ref())),
                "linear" => Ok(ResampleImageFilter::new(
                    [new_nz, new_ny, new_nx],
                    orig_orig,
                    new_sp,
                    orig_dir,
                    TranslationTransform::<Backend, 3>::new(zero_t),
                    LinearInterpolator::new(),
                )
                .apply(inner.as_ref())),
                "bspline" => Ok(ResampleImageFilter::new(
                    [new_nz, new_ny, new_nx],
                    orig_orig,
                    new_sp,
                    orig_dir,
                    TranslationTransform::<Backend, 3>::new(zero_t),
                    BSplineInterpolator::new(),
                )
                .apply(inner.as_ref())),
                "lanczos4" => Ok(ResampleImageFilter::new(
                    [new_nz, new_ny, new_nx],
                    orig_orig,
                    new_sp,
                    orig_dir,
                    TranslationTransform::<Backend, 3>::new(zero_t),
                    Lanczos4Interpolator::new(),
                )
                .apply(inner.as_ref())),
                other => Err(format!(
                    "Unknown interpolation mode '{}'. Use: nearest, linear, bspline, lanczos4",
                    other
                )),
            }
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(into_py_image(result))
}

/// Register the `filter` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "filter")?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(discrete_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(n4_bias_correction, &m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(gradient_magnitude, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian, &m)?)?;
    m.add_function(wrap_pyfunction!(frangi_vesselness, &m)?)?;
    m.add_function(wrap_pyfunction!(canny_edge_detect, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_of_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(recursive_gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel_gradient, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(grayscale_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(curvature_anisotropic_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(sato_line_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(rescale_intensity, &m)?)?;
    m.add_function(wrap_pyfunction!(intensity_windowing, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_below, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_above, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_outside, &m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_filter, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold, &m)?)?;

    // -- white_top_hat

    #[pyfunction]
    pub fn white_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
        let image = std::sync::Arc::clone(&image.inner);
        let result = py.allow_threads(|| {
            ritk_core::filter::WhiteTopHatFilter::new(radius)
                .apply(image.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        });
        Ok(into_py_image(result?))
    }

    // -- black_top_hat

    #[pyfunction]
    pub fn black_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
        let image = std::sync::Arc::clone(&image.inner);
        let result = py.allow_threads(|| {
            ritk_core::filter::BlackTopHatFilter::new(radius)
                .apply(image.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        });
        Ok(into_py_image(result?))
    }

    // -- hit_or_miss

    #[pyfunction]
    pub fn hit_or_miss(
        py: Python<'_>,
        image: &PyImage,
        fg_radius: usize,
        bg_radius: usize,
    ) -> PyResult<PyImage> {
        let image = std::sync::Arc::clone(&image.inner);
        let result = py.allow_threads(|| {
            ritk_core::filter::HitOrMissTransform::new(fg_radius, bg_radius)
                .apply(image.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        });
        Ok(into_py_image(result?))
    }

    // -- label_dilation

    #[pyfunction]
    pub fn label_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
        let image = std::sync::Arc::clone(&image.inner);
        let result = py.allow_threads(|| {
            ritk_core::filter::LabelDilation::new(radius)
                .apply(image.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        });
        Ok(into_py_image(result?))
    }

    m.add_function(wrap_pyfunction!(white_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(black_top_hat, &m)?)?;
    m.add_function(wrap_pyfunction!(hit_or_miss, &m)?)?;
    m.add_function(wrap_pyfunction!(label_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(label_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(label_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(label_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(morphological_reconstruction, &m)?)?;
    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

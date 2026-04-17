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
use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
use ritk_core::filter::recursive_gaussian::DerivativeOrder;
use ritk_core::filter::vesselness::FrangiConfig;
use ritk_core::filter::{
    AnisotropicDiffusionFilter, BilateralFilter, CannyEdgeDetector, FrangiVesselnessFilter,
    GaussianFilter, GradientMagnitudeFilter, GrayscaleDilation, GrayscaleErosion, LaplacianFilter,
    LaplacianOfGaussianFilter, MedianFilter, N4BiasFieldCorrectionFilter, RecursiveGaussianFilter,
    SobelFilter,
};
use ritk_core::filter::diffusion::CurvatureConfig;
use ritk_core::filter::vesselness::SatoConfig;
use ritk_core::filter::{CurvatureAnisotropicDiffusionFilter, SatoLineFilter};

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

/// Register the `filter` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "filter")?;
    m.add_function(wrap_pyfunction!(gaussian_filter, &m)?)?;
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
    parent.add_submodule(&m)?;
    Ok(())
}

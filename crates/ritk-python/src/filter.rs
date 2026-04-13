//! Python-exposed image filtering functions.
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

use crate::image::{image_to_vec, into_py_image, vec_to_image_like, Backend, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::bias::N4Config;
use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
use ritk_core::filter::vesselness::FrangiConfig;
use ritk_core::filter::{
    AnisotropicDiffusionFilter, FrangiVesselnessFilter, GaussianFilter, GradientMagnitudeFilter,
    LaplacianFilter, N4BiasFieldCorrectionFilter,
};

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
pub fn gaussian_filter(image: &PyImage, sigma: f64) -> PyResult<PyImage> {
    // All three axes use the same physical sigma; GaussianFilter scales each
    // axis by the corresponding spacing value internally.
    let filter = GaussianFilter::<Backend>::new(vec![sigma, sigma, sigma]);
    let result = filter.apply(image.inner.as_ref());
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
pub fn median_filter(image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let (values, shape) = image_to_vec(image.inner.as_ref())?;
    let filtered = median_3d(&values, shape, radius);
    Ok(into_py_image(vec_to_image_like(
        filtered,
        shape,
        image.inner.as_ref(),
    )))
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
    image: &PyImage,
    spatial_sigma: f64,
    range_sigma: f64,
) -> PyResult<PyImage> {
    let (values, shape) = image_to_vec(image.inner.as_ref())?;
    let filtered = bilateral_3d(&values, shape, spatial_sigma, range_sigma);
    Ok(into_py_image(vec_to_image_like(
        filtered,
        shape,
        image.inner.as_ref(),
    )))
}

// ── Inline median implementation ──────────────────────────────────────────────

/// Sliding-window median on a 3-D volume stored in flat Z×Y×X order.
///
/// # Algorithm
/// For each voxel (iz, iy, ix), collect all values in the axis-aligned cube
/// `[iz±r, iy±r, ix±r]` (clamped to image bounds), sort them, and take the
/// middle element.
///
/// # Complexity
/// O(n · (2r+1)^3 · log((2r+1)^3)) per image.  For r=1: O(27n log 27).
fn median_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut neighbors: Vec<f32> = Vec::with_capacity((2 * radius + 1).pow(3));

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            // Replicate (clamp) padding.
                            let zz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                            let yy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                            let xx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            neighbors.push(data[zz * ny * nx + yy * nx + xx]);
                        }
                    }
                }

                // Partial sort: O(n log n) for correctness; could use
                // select_nth_unstable but sort is fine for typical radii.
                neighbors
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // For even-length neighborhoods take lower median (consistent
                // with most medical imaging toolkits).
                output[iz * ny * nx + iy * nx + ix] = neighbors[neighbors.len() / 2];
            }
        }
    }

    output
}

// ── Inline bilateral filter implementation ────────────────────────────────────

/// Bilateral filter on a 3-D volume stored in flat Z×Y×X order.
///
/// # Algorithm
/// For each center voxel p:
///   - Neighbourhood radius r = ⌈3 · σ_s⌉.
///   - For each neighbour q in [p±r]³:
///       w(p, q) = exp(−d_s(p,q)² / 2σ_s²) · exp(−d_r(p,q)² / 2σ_r²)
///   - Output(p) = Σ w(p,q)·I(q) / Σ w(p,q).
///
/// Out-of-bounds neighbours are skipped (not included in numerator or
/// denominator), which is equivalent to treating them as having zero weight.
/// When no valid neighbour exists (degenerate 0-voxel images), the input
/// value is returned unchanged.
///
/// # Precision
/// Accumulation is performed in f64 to avoid catastrophic cancellation.
fn bilateral_3d(data: &[f32], dims: [usize; 3], spatial_sigma: f64, range_sigma: f64) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);

    // Guard against degenerate sigma values.
    let spatial_sigma = spatial_sigma.max(1e-10);
    let range_sigma = range_sigma.max(1e-10);

    let r = (3.0 * spatial_sigma).ceil() as isize;
    let inv_two_ss2 = 1.0 / (2.0 * spatial_sigma * spatial_sigma);
    let inv_two_sr2 = 1.0 / (2.0 * range_sigma * range_sigma);

    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let center_flat = iz * ny * nx + iy * nx + ix;
                let center_val = data[center_flat] as f64;

                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let nz_i = iz as isize + dz;
                            let ny_i = iy as isize + dy;
                            let nx_i = ix as isize + dx;

                            // Skip out-of-bounds voxels entirely (zero contribution).
                            if nz_i < 0
                                || nz_i >= nz as isize
                                || ny_i < 0
                                || ny_i >= ny as isize
                                || nx_i < 0
                                || nx_i >= nx as isize
                            {
                                continue;
                            }

                            let n_flat =
                                nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                            let n_val = data[n_flat] as f64;

                            // Spatial distance squared (in voxel units).
                            let spatial_d2 = (dz * dz + dy * dy + dx * dx) as f64;
                            // Range distance squared.
                            let range_d2 = (center_val - n_val) * (center_val - n_val);

                            let w = (-spatial_d2 * inv_two_ss2 - range_d2 * inv_two_sr2).exp();

                            weighted_sum += w * n_val;
                            weight_total += w;
                        }
                    }
                }

                output[center_flat] = if weight_total > 1e-20 {
                    (weighted_sum / weight_total) as f32
                } else {
                    data[center_flat]
                };
            }
        }
    }

    output
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
    image: &PyImage,
    num_fitting_levels: usize,
    num_iterations: usize,
    noise_estimate: f64,
) -> PyResult<PyImage> {
    let config = N4Config {
        num_fitting_levels,
        num_iterations,
        noise_estimate,
        ..Default::default()
    };
    let filter = N4BiasFieldCorrectionFilter::new(config);
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
    image: &PyImage,
    iterations: usize,
    conductance: f64,
    time_step: f64,
    exponential: bool,
) -> PyResult<PyImage> {
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
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
pub fn gradient_magnitude(image: &PyImage) -> PyResult<PyImage> {
    let spacing = image.inner.spacing();
    let filter = GradientMagnitudeFilter::new([spacing[0], spacing[1], spacing[2]]);
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
pub fn laplacian(image: &PyImage) -> PyResult<PyImage> {
    let spacing = image.inner.spacing();
    let filter = LaplacianFilter::new([spacing[0], spacing[1], spacing[2]]);
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    bright_vessels: bool,
) -> PyResult<PyImage> {
    let scales = scales.unwrap_or_else(|| vec![0.5, 1.0, 2.0]);
    let config = FrangiConfig {
        scales,
        alpha,
        beta,
        gamma,
        bright_vessels,
    };
    let filter = FrangiVesselnessFilter { config };
    let result = filter
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

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
    parent.add_submodule(&m)?;
    Ok(())
}

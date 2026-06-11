//! Special-purpose filters: median, bilateral, N4 bias correction, and bin-shrink downsampling.
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::bias::N4Config;
use ritk_core::filter::{
    BilateralFilter, BinShrinkImageFilter, MedianFilter, N4BiasFieldCorrectionFilter,
};

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

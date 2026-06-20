use crate::errors::RitkResult;
use crate::errors::RitkPyError;
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::{
    slic_itk_segment as core_slic_itk_segment, SlicConfig, SlicSuperpixelFilter,
};
use std::sync::Arc;

/// SLIC super-pixel segmentation matching `SimpleITK.SLIC`.
///
/// Ports ITK's `SLICImageFilter`: shrink-grid centre initialisation, the
/// `(I-I_c)^2 + sum((p-c)*m/g)^2` distance, a fixed-count Lloyd iteration, and
/// the default `initializationPerturbation` + `enforceConnectivity` post-passes.
/// Label-for-label exact vs sitk for uniform `super_grid_size` (2-D `z==1`
/// volumes are handled as genuine 2-D images).
///
/// Args:
///     image: scalar image (`z==1` ⇒ 2-D).
///     super_grid_size: uniform per-axis grid step (sitk `superGridSize`).
///     spatial_proximity_weight: sitk `spatialProximityWeight` (default 10.0).
///     maximum_number_of_iterations: sitk `maximumNumberOfIterations` (default 5).
///     enforce_connectivity: sitk `enforceConnectivity` (default True).
///     initialization_perturbation: sitk `initializationPerturbation` (default True).
///
/// Returns:
///     label image (super-pixel indices as f32).
#[pyfunction]
#[pyo3(signature = (image, super_grid_size, spatial_proximity_weight=10.0,
    maximum_number_of_iterations=5, enforce_connectivity=true,
    initialization_perturbation=true))]
pub fn slic(
    py: Python<'_>,
    image: &PyImage,
    super_grid_size: usize,
    spatial_proximity_weight: f64,
    maximum_number_of_iterations: usize,
    enforce_connectivity: bool,
    initialization_perturbation: bool,
) -> PyImage {
    let arc = Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        core_slic_itk_segment(
            arc.as_ref(),
            super_grid_size,
            spatial_proximity_weight,
            maximum_number_of_iterations,
            initialization_perturbation,
            enforce_connectivity,
        )
    });
    into_py_image(out)
}

/// Segment a 3D image via SLIC super-pixel clustering (Achanta et al. 2012).
///
/// SLIC performs local clustering of voxels in a combined
/// intensity-spatial feature space, producing spatially compact
/// super-pixel regions. Uses k-means-style Lloyd iteration on a
/// regular grid initialization with search-window optimization.
///
/// Args:
///     image: Input PyImage.
///     n_superpixels: Number of desired superpixels (default 100).
///     compactness: Compactness parameter: higher = more regular shapes (default 10.0).
///     max_iterations: Maximum Lloyd iterations (default 10).
///     tolerance: Convergence tolerance on center shift (default 1e-3).
///     seed: Deterministic seed (default 42).
///     min_component_size: Minimum component size for connectivity enforcement (default 5).
///
/// Returns:
///     Label PyImage with superpixel indices in [0, K-1].
#[pyfunction]
#[pyo3(signature = (image, n_superpixels=100, compactness=10.0, max_iterations=10, tolerance=1e-3, seed=42, min_component_size=5))]
pub fn slic_superpixel(
    py: Python<'_>,
    image: &PyImage,
    n_superpixels: usize,
    compactness: f64,
    max_iterations: usize,
    tolerance: f64,
    seed: u64,
    min_component_size: usize,
) -> RitkResult<PyImage> {
    if n_superpixels < 1 {
        return Err(RitkPyError::value("n_superpixels must be >= 1"));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let config = SlicConfig {
            n_superpixels,
            compactness,
            max_iterations,
            tolerance,
            seed,
            min_component_size,
        };
        let filter = SlicSuperpixelFilter::new(config);
        filter.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

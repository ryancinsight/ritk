use crate::errors::RitkPyError;
use crate::errors::RitkResult;
use crate::image::{native_into_py_image, py_image_to_native, PyImage};
use coeus_core::SequentialBackend;
use pyo3::prelude::*;
use ritk_segmentation::{
    ConnectivityEnforcement, InitializationPerturbation, ItkSlicConfig, ItkSlicFilter, SlicConfig,
    SlicSuperpixelFilter };

/// SLIC super-pixel segmentation matching `SimpleITK.SLIC`.
///
/// Ports ITK's `SLICImageFilter`: shrink-grid centre initialisation, the
/// `(I-I_c)^2 + sum((p-c)*m/g)^2` distance, a fixed-count Lloyd iteration, and
/// the default `initializationPerturbation` + `enforceConnectivity` post-passes.
/// Label-for-label exact vs sitk for uniform `super_grid_size` (2-D `z==1`
/// volumes are handled as genuine 2-D images).
///
/// Args:
///     image: scalar image (`z==1` â‡’ 2-D).
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
) -> RitkResult<PyImage> {
    let perturbation = if initialization_perturbation {
        InitializationPerturbation::Enabled
    } else {
        InitializationPerturbation::Disabled
    };
    let connectivity = if enforce_connectivity {
        ConnectivityEnforcement::Enabled
    } else {
        ConnectivityEnforcement::Disabled
    };
    let config = ItkSlicConfig::new(super_grid_size)
        .and_then(|config| config.with_spatial_proximity_weight(spatial_proximity_weight))
        .and_then(|config| config.with_maximum_iterations(maximum_number_of_iterations))
        .map(|config| {
            config
                .with_initialization_perturbation(perturbation)
                .with_connectivity(connectivity)
        })
        .map_err(|error| RitkPyError::value(error.to_string()))?;
    let image = py_image_to_native(image)?;
    let output = py
        .allow_threads(|| ItkSlicFilter::new(config).apply_native(&image, &SequentialBackend))
        .map_err(|error| RitkPyError::value(error.to_string()))?;
    Ok(native_into_py_image(output))
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
///     min_component_size: Minimum component size for connectivity enforcement (default 5).
///
/// Returns:
///     Label PyImage with superpixel indices in [0, K-1].
#[pyfunction]
#[pyo3(signature = (image, n_superpixels=100, compactness=10.0, max_iterations=10, tolerance=1e-3, min_component_size=5))]
pub fn slic_superpixel(
    py: Python<'_>,
    image: &PyImage,
    n_superpixels: usize,
    compactness: f32,
    max_iterations: usize,
    tolerance: f32,
    min_component_size: usize,
) -> RitkResult<PyImage> {
    let config = SlicConfig::new(n_superpixels)
        .and_then(|config| config.with_compactness(compactness))
        .and_then(|config| config.with_max_iterations(max_iterations))
        .and_then(|config| config.with_tolerance(tolerance))
        .map(|config| config.with_min_component_size(min_component_size))
        .map_err(|error| RitkPyError::value(error.to_string()))?;
    let image = py_image_to_native(image)?;
    let result = py
        .allow_threads(|| {
            SlicSuperpixelFilter::new(config).apply_native(&image, &SequentialBackend)
        })
        .map_err(|error| RitkPyError::value(error.to_string()))?;
    Ok(native_into_py_image(result))
}

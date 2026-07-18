use crate::errors::RitkPyError;
use crate::errors::RitkResult;
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_segmentation::{
    labeling::Connectivity as SegConnectivity, ConnectedComponentsFilter, KMeansSegmentation };

/// Compute per-label shape statistics from a binary mask.
///
/// Returns per-component spatial statistics (voxel count, centroid, bounding box).
/// Background (label 0) is excluded from results.
///
/// Args:
///     mask:         Binary mask image (foreground > 0.5).
///     connectivity: Adjacency model (6 or 26; default 6).
///
/// Returns:
///     list of dicts, one per component, sorted by label ascending, each with keys:
///     label (int), voxel_count (int),
///     centroid (list\[float\]: [z, y, x] in index coordinates),
///     bounding_box_min (list\[int\]: [z, y, x]),
///     bounding_box_max (list\[int\]: [z, y, x]).
///
/// Raises:
///     ValueError: if connectivity is not 6 or 26.
#[pyfunction]
#[pyo3(signature = (mask, connectivity=6_u32))]
pub fn label_shape_statistics(
    py: Python<'_>,
    mask: &PyImage,
    connectivity: u32,
) -> RitkResult<Py<PyList>> {
    if connectivity != 6 && connectivity != 26 {
        return Err(RitkPyError::value(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }
    let mask_arc = py_image_to_burn(mask);
    let (_label_image, stats) = py.allow_threads(|| {
        let seg_conn = if connectivity == 6 {
            SegConnectivity::Six
        } else {
            SegConnectivity::TwentySix
        };
        ConnectedComponentsFilter::with_connectivity(seg_conn).apply(&mask_arc)
    });
    let list = PyList::empty_bound(py);
    for s in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", s.label)?;
        dict.set_item("voxel_count", s.voxel_count)?;
        let centroid: Vec<f64> = s.centroid.to_array().to_vec();
        dict.set_item("centroid", centroid)?;
        let (bb_min, bb_max) = s.bounding_box;
        let bb_min_list: Vec<i64> = bb_min.iter().map(|&v| v as i64).collect();
        let bb_max_list: Vec<i64> = bb_max.iter().map(|&v| v as i64).collect();
        dict.set_item("bounding_box_min", bb_min_list)?;
        dict.set_item("bounding_box_max", bb_max_list)?;
        list.append(dict)?;
    }
    Ok(list.into())
}

/// Segment a 3D image via Lloyd's K-Means clustering.
///
/// Delegates to `ritk_segmentation::KMeansSegmentation`. Voxel intensities
/// are treated as 1-D feature vectors; centroids are initialized via k-means++.
///
/// Args:
///     image:           Input PyImage.
///     k:               Number of clusters (â‰¥ 1).  Default 3.
///     max_iterations:  Maximum Lloyd iterations.  Default 100.
///     tolerance:       Centroid-displacement convergence tolerance.  Default 1e-6.
///     seed:            Deterministic seed for k-means++ initialization.  Default 42.
///
/// Returns:
///     Label PyImage with cluster indices in [0, kâˆ’1].
#[pyfunction]
#[pyo3(signature = (image, k=3, max_iterations=None, tolerance=None, seed=None))]
pub fn kmeans_segment(
    py: Python<'_>,
    image: &PyImage,
    k: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f32>,
    seed: Option<u64>,
) -> RitkResult<PyImage> {
    let mut segmentation =
        KMeansSegmentation::new(k).map_err(|error| RitkPyError::value(error.to_string()))?;
    if let Some(iterations) = max_iterations {
        segmentation = segmentation
            .with_max_iterations(iterations)
            .map_err(|error| RitkPyError::value(error.to_string()))?;
    }
    if let Some(tolerance) = tolerance {
        segmentation = segmentation
            .with_tolerance(tolerance)
            .map_err(|error| RitkPyError::value(error.to_string()))?;
    }
    if let Some(seed) = seed {
        segmentation = segmentation.with_seed(seed);
    }
    let image = py_image_to_burn(image);
    let result = py
        .allow_threads(|| segmentation.apply(&image))
        .map_err(|error| RitkPyError::value(error.to_string()))?;
    Ok(burn_into_py_image(result))
}

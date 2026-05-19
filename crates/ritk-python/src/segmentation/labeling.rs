//! Labeling, clustering, and watershed segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_core::segmentation::{
    connected_components as core_connected_components, ConnectedComponentsFilter,
    KMeansSegmentation, MarkerControlledWatershed, WatershedSegmentation,
};
use std::sync::Arc;

/// Label connected components in a binary mask.
///
/// Delegates to `ritk_core::segmentation::connected_components` (Hoshen-Kopelman
/// two-pass labeling).  Foreground voxels (value > 0.5) receive consecutive
/// integer labels [1, K] cast to f32; background voxels remain 0.0.
///
/// Args:
///     mask:         Binary mask PyImage (values 0 or 1).
///     connectivity: 6 (face-adjacent, default) or 26 (face + edge + corner).
///
/// Returns:
///     (labeled_image, num_components): label image and component count K.
///
/// Raises:
///     ValueError: if connectivity is not 6 or 26.
#[pyfunction]
#[pyo3(signature = (mask, connectivity=6))]
pub fn connected_components(
    py: Python<'_>,
    mask: &PyImage,
    connectivity: u32,
) -> RitkResult<(PyImage, usize)> {
    if connectivity != 6 && connectivity != 26 {
        return Err(RitkPyError::value(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }

    let mask = Arc::clone(&mask.inner);
    let (label_image, num_components) =
        py.allow_threads(|| core_connected_components(mask.as_ref(), connectivity));
    Ok((into_py_image(label_image), num_components))
}

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
///     centroid (list[float]: [z, y, x] in index coordinates),
///     bounding_box_min (list[int]: [z, y, x]),
///     bounding_box_max (list[int]: [z, y, x]).
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
    let mask_arc = Arc::clone(&mask.inner);
    let (_label_image, stats) = py.allow_threads(|| {
        ConnectedComponentsFilter::with_connectivity(connectivity).apply(mask_arc.as_ref())
    });
    let list = PyList::empty_bound(py);
    for s in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", s.label)?;
        dict.set_item("voxel_count", s.voxel_count)?;
        let centroid: Vec<f64> = s.centroid.to_vec();
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
/// Delegates to `ritk_core::segmentation::KMeansSegmentation`. Voxel intensities
/// are treated as 1-D feature vectors; centroids are initialized via k-means++.
///
/// Args:
///     image:           Input PyImage.
///     k:               Number of clusters (≥ 1).  Default 3.
///     max_iterations:  Maximum Lloyd iterations.  Default 100.
///     tolerance:       Centroid-displacement convergence tolerance.  Default 1e-6.
///     seed:            Deterministic seed for k-means++ initialization.  Default 42.
///
/// Returns:
///     Label PyImage with cluster indices in [0, k−1].
#[pyfunction]
#[pyo3(signature = (image, k=3, max_iterations=None, tolerance=None, seed=None))]
pub fn kmeans_segment(
    py: Python<'_>,
    image: &PyImage,
    k: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
) -> RitkResult<PyImage> {
    if k < 1 {
        return Err(RitkPyError::value("k must be ≥ 1"));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let mut seg = KMeansSegmentation::new(k);
        if let Some(mi) = max_iterations {
            seg.max_iterations = mi;
        }
        if let Some(tol) = tolerance {
            seg.tolerance = tol;
        }
        if let Some(s) = seed {
            seg.seed = s;
        }
        seg.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

/// Segment a 3D image via Meyer's flooding watershed algorithm.
///
/// Delegates to `ritk_core::segmentation::WatershedSegmentation`. The input
/// should be a gradient magnitude image. Each output voxel receives a basin
/// label (≥ 1) or 0 for watershed boundaries.
///
/// Args:
///     image: Input PyImage (typically gradient magnitude).
///
/// Returns:
///     Label PyImage with basin indices and watershed boundaries (0).
#[pyfunction]
pub fn watershed_segment(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let seg = WatershedSegmentation::new();
        seg.apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    }).map(into_py_image)
}

/// Run marker-controlled watershed segmentation on a gradient-magnitude image.
///
/// Delegates to `ritk_core::segmentation::MarkerControlledWatershed`.
///
/// Priority-queue flooding (Meyer algorithm): voxels are processed in
/// ascending gradient order. Each unlabeled voxel is assigned the label of
/// the lowest-gradient labeled neighbor. Watershed boundaries remain at zero.
///
/// Args:
///     gradient: 3D scalar gradient-magnitude image (f32). Drives flooding order.
///               Typically produced by a Sobel or Gaussian-derivative filter.
///     markers:  3D label image (f32). Non-zero integer values define basin seeds.
///               Zero voxels are unlabeled and will be flooded. Must be same shape
///               as `gradient`.
///
/// Returns:
///     PyImage (f32 label image) with the same shape and spatial metadata as
///     `gradient`. Non-zero values are basin labels from the markers; zero values
///     are watershed boundaries or voxels unreachable from any seed.
///
/// Raises:
///     RuntimeError: if gradient and markers have different shapes, or if the
///                   underlying tensor data cannot be read as f32.
#[pyfunction]
#[pyo3(signature = (gradient, markers))]
pub fn marker_watershed_segment(
    py: Python<'_>,
    gradient: &PyImage,
    markers: &PyImage,
) -> RitkResult<PyImage> {
    let grad_arc = Arc::clone(&gradient.inner);
    let mark_arc = Arc::clone(&markers.inner);
    let result = py.allow_threads(|| {
        MarkerControlledWatershed::new().apply(grad_arc.as_ref(), mark_arc.as_ref())
    });
    result
        .map(into_py_image)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
}

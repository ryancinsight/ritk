//! Labeling, clustering, and watershed segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::Backend;
use crate::image::{into_py_image, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_image::Image;
use ritk_segmentation::{
    connected_components as core_connected_components, labeling::Connectivity as SegConnectivity,
    scalar_connected_components as core_scalar_connected_components, ConnectedComponentsFilter,
    KMeansSegmentation, MarkerControlledWatershed, RelabelComponentFilter, SlicConfig,
    SlicSuperpixelFilter, ThresholdMaximumConnectedComponentsFilter, WatershedSegmentation,
};
use std::sync::Arc;

/// Label connected components in a binary mask.
///
/// Delegates to `ritk_segmentation::connected_components` (Hoshen-Kopelman
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
    let (label_image, num_components) = {
        let seg_conn = if connectivity == 6 {
            SegConnectivity::Six
        } else {
            SegConnectivity::TwentySix
        };
        py.allow_threads(|| core_connected_components(mask.as_ref(), seg_conn))
    };
    Ok((into_py_image(label_image), num_components))
}

/// Label scalar connected components: every voxel is labelled, and two
/// raster-adjacent voxels share a component when their intensities differ by at
/// most `distance_threshold`. Labels are consecutive `1..=K` in scan order.
///
/// ITK Parity: ScalarConnectedComponentImageFilter (`sitk.ScalarConnectedComponent`).
///
/// Args:
///     image:              Scalar PyImage.
///     distance_threshold: Maximum intensity difference for two neighbours to
///                         join the same component (default 0.0).
///     connectivity:       6 (face, default) or 26 (full).
#[pyfunction]
#[pyo3(signature = (image, distance_threshold=0.0, connectivity=6))]
pub fn scalar_connected_component(
    py: Python<'_>,
    image: &PyImage,
    distance_threshold: f32,
    connectivity: u32,
) -> RitkResult<PyImage> {
    if connectivity != 6 && connectivity != 26 {
        return Err(RitkPyError::value(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }
    let arc = Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        let dims = arc.shape();
        let vals = arc.data_slice().into_owned();
        let labels =
            core_scalar_connected_components(&vals, dims, distance_threshold, connectivity);
        let device = NdArrayDevice::default();
        let tensor =
            Tensor::<Backend, 3>::from_data(TensorData::new(labels, Shape::new(dims)), &device);
        Image::new(tensor, *arc.origin(), *arc.spacing(), *arc.direction())
    });
    Ok(into_py_image(out))
}

/// Relabel connected components by descending size: the largest object becomes
/// label 1, the next largest 2, and so on. Components smaller than
/// `minimum_object_size` voxels are removed (mapped to background 0).
///
/// ITK Parity: RelabelComponentImageFilter (`sitk.RelabelComponent` with
/// `sortByObjectSize=True`).
///
/// Args:
///     label_image: an integer label image (e.g. from `connected_components`).
///     minimum_object_size: components with fewer voxels are discarded (default 0).
///
/// Returns:
///     the relabelled image.
#[pyfunction]
#[pyo3(signature = (label_image, minimum_object_size=0))]
pub fn relabel_components(
    py: Python<'_>,
    label_image: &PyImage,
    minimum_object_size: usize,
) -> PyImage {
    let img = Arc::clone(&label_image.inner);
    let out = py.allow_threads(|| {
        RelabelComponentFilter::with_minimum_object_size(minimum_object_size)
            .apply(img.as_ref())
            .0
    });
    into_py_image(out)
}

/// Threshold an image at the lower value that maximizes the number of connected
/// components, matching `SimpleITK.ThresholdMaximumConnectedComponents`.
///
/// Binary-searches the threshold `T` maximizing the count of connected
/// components (size ≥ `minimum_object_size`, face connectivity) in the band
/// `T ≤ I ≤ upper_boundary`, then returns that binary mask (1 inside, 0 outside).
///
/// Args:
///     image: Input (integer-valued) PyImage.
///     minimum_object_size: Components smaller than this are not counted (default 0).
///     upper_boundary: Upper threshold bound; `None` uses the image maximum.
///
/// Returns:
///     Binary PyImage (1 inside the selected band, 0 outside).
#[pyfunction]
#[pyo3(signature = (image, minimum_object_size=0, upper_boundary=None))]
pub fn threshold_maximum_connected_components(
    py: Python<'_>,
    image: &PyImage,
    minimum_object_size: usize,
    upper_boundary: Option<i64>,
) -> PyImage {
    let img = Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        ThresholdMaximumConnectedComponentsFilter {
            minimum_object_size,
            upper_boundary,
            inside_value: 1.0,
            outside_value: 0.0,
        }
        .apply(img.as_ref())
    });
    into_py_image(out)
}

/// Remap label values according to a `{old: new}` change map. Voxels whose
/// (integral) value is not a key are left unchanged.
///
/// ITK Parity: ChangeLabelImageFilter (`sitk.ChangeLabel`).
///
/// Args:
///     label_image: an integer-valued label image.
///     change_map: dict mapping old label → new label.
///
/// Returns:
///     the remapped image (same shape and spatial metadata).
#[pyfunction]
pub fn change_label(
    py: Python<'_>,
    label_image: &PyImage,
    change_map: std::collections::HashMap<i64, i64>,
) -> PyImage {
    use burn::tensor::{Shape, Tensor, TensorData};
    let img = Arc::clone(&label_image.inner);
    let out = py.allow_threads(|| {
        let dims = img.shape();
        let out: Vec<f32> = img
            .data_slice()
            .iter()
            .map(|&v| {
                let k = v as i64;
                // Only remap exactly-integral values present in the map.
                if k as f32 == v {
                    change_map.get(&k).map(|&nv| nv as f32).unwrap_or(v)
                } else {
                    v
                }
            })
            .collect();
        let device = burn_ndarray::NdArrayDevice::default();
        let tensor = Tensor::<crate::image::Backend, 3>::from_data(
            TensorData::new(out, Shape::new(dims)),
            &device,
        );
        ritk_image::Image::new(tensor, *img.origin(), *img.spacing(), *img.direction())
    });
    into_py_image(out)
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
        let seg_conn = if connectivity == 6 {
            SegConnectivity::Six
        } else {
            SegConnectivity::TwentySix
        };
        ConnectedComponentsFilter::with_connectivity(seg_conn).apply(mask_arc.as_ref())
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
/// Delegates to `ritk_segmentation::WatershedSegmentation`. The input
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
    })
    .map(into_py_image)
}

/// Run marker-controlled watershed segmentation on a gradient-magnitude image.
///
/// Delegates to `ritk_segmentation::MarkerControlledWatershed`.
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

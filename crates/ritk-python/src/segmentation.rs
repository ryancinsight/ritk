//! Python-exposed segmentation functions delegating to `ritk_core::segmentation`.
//!
//! This module is a thin PyO3 binding layer.  All algorithmic work is performed
//! by the authoritative implementations in `ritk_core::segmentation`:
//!
//! - **Otsu thresholding** → `ritk_core::segmentation::OtsuThreshold`
//! - **Li thresholding** → `ritk_core::segmentation::LiThreshold`
//! - **Yen thresholding** → `ritk_core::segmentation::YenThreshold`
//! - **Kapur thresholding** → `ritk_core::segmentation::KapurThreshold`
//! - **Triangle thresholding** → `ritk_core::segmentation::TriangleThreshold`
//! - **Multi-Otsu thresholding** → `ritk_core::segmentation::MultiOtsuThreshold`
//! - **Connected-component labeling** → `ritk_core::segmentation::connected_components`
//! - **Connected-threshold region growing** → `ritk_core::segmentation::connected_threshold`
//! - **K-Means clustering** → `ritk_core::segmentation::KMeansSegmentation`
//! - **Watershed segmentation** → `ritk_core::segmentation::WatershedSegmentation`
//! - **Marker-controlled watershed** → `ritk_core::segmentation::MarkerControlledWatershed`
//! - **Binary erosion** → `ritk_core::segmentation::BinaryErosion`
//! - **Binary dilation** → `ritk_core::segmentation::BinaryDilation`
//! - **Binary opening** → `ritk_core::segmentation::BinaryOpening`
//! - **Binary closing** → `ritk_core::segmentation::BinaryClosing`
//! - **Chan-Vese level set** → `ritk_core::segmentation::ChanVeseSegmentation`
//! - **Geodesic Active Contour** → `ritk_core::segmentation::GeodesicActiveContourSegmentation`
//! - **Shape Detection level set** → `ritk_core::segmentation::ShapeDetectionSegmentation`
//! - **Threshold Level Set** → `ritk_core::segmentation::ThresholdLevelSet`
//! - **Confidence-connected region growing** → `ritk_core::segmentation::ConfidenceConnectedFilter`
//! - **Neighbourhood-connected region growing** → `ritk_core::segmentation::NeighborhoodConnectedFilter`
//! - **Skeletonization** → `ritk_core::segmentation::Skeletonization`
//! - **Binary fill holes** → `ritk_core::segmentation::BinaryFillHoles`
//! - **Morphological gradient** → `ritk_core::segmentation::MorphologicalGradient`
//!
//! No algorithm logic is duplicated here; SSOT is maintained in `ritk-core`.

use crate::image::{into_py_image, vec_to_image_like, with_tensor_slice, PyImage};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_core::segmentation::threshold::kapur::compute_kapur_threshold_from_slice;
use ritk_core::segmentation::threshold::li::compute_li_threshold_from_slice;
use ritk_core::segmentation::threshold::multi_otsu::compute_multi_otsu_thresholds_from_slice;
use ritk_core::segmentation::threshold::otsu::compute_otsu_threshold_from_slice;
use ritk_core::segmentation::threshold::triangle::compute_triangle_threshold_from_slice;
use ritk_core::segmentation::threshold::yen::compute_yen_threshold_from_slice;
use ritk_core::segmentation::{
    connected_components as core_connected_components,
    connected_threshold as core_connected_threshold, BinaryClosing, BinaryDilation, BinaryErosion,
    BinaryFillHoles, BinaryOpening, BinaryThreshold, ChanVeseSegmentation,
    ConfidenceConnectedFilter, ConnectedComponentsFilter, GeodesicActiveContourSegmentation,
    KMeansSegmentation, LaplacianLevelSet, MarkerControlledWatershed, MorphologicalGradient,
    MorphologicalOperation, NeighborhoodConnectedFilter, ShapeDetectionSegmentation,
    Skeletonization, ThresholdLevelSet, WatershedSegmentation,
};
use std::sync::Arc;

// ── Threshold: Otsu ───────────────────────────────────────────────────────────

/// Compute the Otsu threshold and produce a binary mask.
///
/// Delegates to `ritk_core::segmentation::OtsuThreshold` (256-bin histogram,
/// maximises between-class variance σ²_B).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn otsu_threshold(py: Python<'_>, image: &PyImage) -> PyResult<(f32, PyImage)> {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    // Zero-copy: single slice extraction replaces two clone().into_data() calls
    // (one inside OtsuThreshold::compute, one inside OtsuThreshold::apply).
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_otsu_threshold_from_slice(slice, 256);
            // Inline apply: avoids second data extraction inside OtsuThreshold::apply.
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── Threshold: Li ─────────────────────────────────────────────────────────────

/// Compute the Li minimum cross-entropy threshold and produce a binary mask.
///
/// Delegates to `ritk_core::segmentation::LiThreshold` (256-bin histogram,
/// iterative cross-entropy minimisation, Li & Tam 1998).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn li_threshold(py: Python<'_>, image: &PyImage) -> PyResult<(f32, PyImage)> {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_li_threshold_from_slice(slice, 256, 1000);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── Threshold: Yen ────────────────────────────────────────────────────────────

/// Compute the Yen maximum correlation threshold and produce a binary mask.
///
/// Delegates to `ritk_core::segmentation::YenThreshold` (256-bin histogram,
/// Yen, Chang & Chang 1995).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn yen_threshold(py: Python<'_>, image: &PyImage) -> PyResult<(f32, PyImage)> {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_yen_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── Threshold: Kapur ──────────────────────────────────────────────────────────

/// Compute the Kapur maximum entropy threshold and produce a binary mask.
///
/// Delegates to `ritk_core::segmentation::KapurThreshold` (256-bin histogram,
/// Kapur, Sahoo & Wong 1985).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn kapur_threshold(py: Python<'_>, image: &PyImage) -> PyResult<(f32, PyImage)> {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_kapur_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── Threshold: Triangle ───────────────────────────────────────────────────────

/// Compute the Triangle (Zack) threshold and produce a binary mask.
///
/// Delegates to `ritk_core::segmentation::TriangleThreshold` (256-bin histogram,
/// Zack, Rogers & Latt 1977).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn triangle_threshold(py: Python<'_>, image: &PyImage) -> PyResult<(f32, PyImage)> {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_triangle_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── Threshold: Multi-Otsu ─────────────────────────────────────────────────────

/// Compute multi-class Otsu thresholds and produce a labeled image.
///
/// Delegates to `ritk_core::segmentation::MultiOtsuThreshold`. Returns K−1
/// thresholds and a label image with class indices {0, 1, …, K−1} as f32.
///
/// Args:
///     image:       Input PyImage.
///     num_classes: Number of intensity classes (≥ 2). Default 3.
///
/// Returns:
///     (thresholds, labeled_image): list of K−1 threshold values and labeled PyImage.
#[pyfunction]
#[pyo3(signature = (image, num_classes=3))]
pub fn multi_otsu_threshold(
    py: Python<'_>,
    image: &PyImage,
    num_classes: usize,
) -> PyResult<(Vec<f32>, PyImage)> {
    if num_classes < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_classes must be ≥ 2",
        ));
    }
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (thresholds, label_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let thresholds = compute_multi_otsu_thresholds_from_slice(slice, num_classes, 256);
            let label_vals: Vec<f32> = slice
                .iter()
                .map(|&v| thresholds.iter().filter(|&&t| v >= t).count() as f32)
                .collect();
            (thresholds, label_vals)
        })
    });
    let labeled = vec_to_image_like(label_vals, dims, arc.as_ref());
    Ok((thresholds, into_py_image(labeled)))
}

// ── Connected components ──────────────────────────────────────────────────────

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
) -> PyResult<(PyImage, usize)> {
    if connectivity != 6 && connectivity != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }

    let mask = Arc::clone(&mask.inner);
    let (label_image, num_components) =
        py.allow_threads(|| core_connected_components(mask.as_ref(), connectivity));
    Ok((into_py_image(label_image), num_components))
}

// ── Connected-threshold region growing ────────────────────────────────────────

/// Segment a region by connected-threshold flood-fill from a seed voxel.
///
/// Delegates to `ritk_core::segmentation::connected_threshold`. Grows a
/// 6-connected region from `seed` including all reachable voxels with
/// intensity in [lower, upper].
///
/// Args:
///     image: Input PyImage.
///     seed:  Seed voxel as [z, y, x] indices.
///     lower: Inclusive lower intensity bound.
///     upper: Inclusive upper intensity bound.
///
/// Returns:
///     Binary mask PyImage (1.0 = included, 0.0 = excluded).
///
/// Raises:
///     ValueError: if lower > upper or seed is out of bounds.
#[pyfunction]
#[pyo3(signature = (image, seed, lower, upper))]
pub fn connected_threshold_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: [usize; 3],
    lower: f32,
    upper: f32,
) -> PyResult<PyImage> {
    if lower > upper {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lower bound ({lower}) must be ≤ upper bound ({upper})"
        )));
    }
    let shape = image.inner.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed {:?} is out of bounds for image shape {:?}",
            seed, shape
        )));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| core_connected_threshold(image.as_ref(), seed, lower, upper));
    Ok(into_py_image(result))
}

// ── K-Means clustering ───────────────────────────────────────────────────────

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
) -> PyResult<PyImage> {
    if k < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("k must be ≥ 1"));
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

// ── Watershed segmentation ───────────────────────────────────────────────────

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
pub fn watershed_segment(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let seg = WatershedSegmentation::new();
        seg.apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

// ── Binary erosion ───────────────────────────────────────────────────────────

/// Apply binary erosion with a box structuring element.
///
/// Delegates to `ritk_core::segmentation::BinaryErosion`. For each voxel p,
/// output[p] = 1.0 iff all voxels within the axis-aligned hypercube of
/// half-width `radius` centred at p are foreground.
///
/// Args:
///     image:  Binary mask PyImage.
///     radius: Half-width of the box structuring element in voxels. Default 1.
///
/// Returns:
///     Eroded binary mask PyImage.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn binary_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let op = BinaryErosion::new(radius);
        op.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── Binary dilation ──────────────────────────────────────────────────────────

/// Apply binary dilation with a box structuring element.
///
/// Delegates to `ritk_core::segmentation::BinaryDilation`. For each voxel p,
/// output[p] = 1.0 iff any voxel within the axis-aligned hypercube of
/// half-width `radius` centred at p is foreground.
///
/// Args:
///     image:  Binary mask PyImage.
///     radius: Half-width of the box structuring element in voxels. Default 1.
///
/// Returns:
///     Dilated binary mask PyImage.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn binary_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let op = BinaryDilation::new(radius);
        op.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── Binary opening ───────────────────────────────────────────────────────────

/// Apply binary opening (erosion followed by dilation).
///
/// Delegates to `ritk_core::segmentation::BinaryOpening`. Removes small
/// foreground regions while preserving the shape of larger structures.
///
/// Args:
///     image:  Binary mask PyImage.
///     radius: Half-width of the box structuring element in voxels. Default 1.
///
/// Returns:
///     Opened binary mask PyImage.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn binary_opening(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let op = BinaryOpening::new(radius);
        op.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── Binary closing ───────────────────────────────────────────────────────────

/// Apply binary closing (dilation followed by erosion).
///
/// Delegates to `ritk_core::segmentation::BinaryClosing`. Fills small
/// background holes while preserving the shape of the foreground.
///
/// Args:
///     image:  Binary mask PyImage.
///     radius: Half-width of the box structuring element in voxels. Default 1.
///
/// Returns:
///     Closed binary mask PyImage.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn binary_closing(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let op = BinaryClosing::new(radius);
        op.apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── Morphology: fill holes and gradient ───────────────────────────────────────────

/// Fill enclosed background holes in a binary mask.
///
/// Uses 6-connected border flood-fill to identify exterior background voxels.
/// All background voxels unreachable from any border face are set to foreground.
///
/// Args:
///     image: Binary mask PyImage (values in {0.0, 1.0}).
///
/// Returns:
///     Hole-filled binary mask, same shape and spatial metadata as input.
#[pyfunction]
pub fn binary_fill_holes(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || BinaryFillHoles.apply(inner.as_ref()));
    Ok(into_py_image(result))
}

/// Compute the morphological gradient (boundary extraction) of a binary mask.
///
/// Output is 1.0 at boundary voxels (in dilation but not erosion) and 0.0
/// at interior foreground, exterior background, and all other voxels.
///
/// Args:
///     image: Binary mask PyImage (values in {0.0, 1.0}).
///     radius: Structuring element ball radius (default: 1).
///
/// Returns:
///     Binary boundary mask, same shape and spatial metadata as input.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn morphological_gradient(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || MorphologicalGradient::new(radius).apply(inner.as_ref()));
    Ok(into_py_image(result))
}

// ── Chan-Vese level set ──────────────────────────────────────────────────────

/// Segment a 3D image via Chan-Vese level set evolution.
///
/// Delegates to `ritk_core::segmentation::ChanVeseSegmentation` (Active
/// Contours Without Edges, Chan & Vese 2001). Evolves a level set function
/// under an energy functional driven by region statistics (no edges required).
///
/// Args:
///     image:          Input PyImage.
///     mu:             Curvature (length) penalty weight. Default 0.25.
///     nu:             Area penalty weight. Default 0.0.
///     lambda1:        Data fidelity weight for inside region. Default 1.0.
///     lambda2:        Data fidelity weight for outside region. Default 1.0.
///     max_iterations: Maximum PDE evolution iterations. Default 200.
///     dt:             Euler forward time step. Default 0.1.
///     tolerance:      Convergence tolerance on max|Δφ|/dt. Default 1e-3.
///
/// Returns:
///     Binary mask PyImage (1.0 = inside, 0.0 = outside).
#[pyfunction]
#[pyo3(signature = (image, mu=0.25, nu=0.0, lambda1=1.0, lambda2=1.0, max_iterations=200, dt=0.1, tolerance=1e-3))]
pub fn chan_vese_segment(
    image: &PyImage,
    mu: f64,
    nu: f64,
    lambda1: f64,
    lambda2: f64,
    max_iterations: usize,
    dt: f64,
    tolerance: f64,
) -> PyResult<PyImage> {
    let mut seg = ChanVeseSegmentation::new();
    seg.mu = mu;
    seg.nu = nu;
    seg.lambda1 = lambda1;
    seg.lambda2 = lambda2;
    seg.max_iterations = max_iterations;
    seg.dt = dt;
    seg.tolerance = tolerance;
    let result = seg
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── Geodesic Active Contour ──────────────────────────────────────────────────

/// Segment a 3D image via Geodesic Active Contour level set evolution.
///
/// Delegates to `ritk_core::segmentation::GeodesicActiveContourSegmentation`
/// (Caselles, Kimmel & Sapiro 1997). Evolves an initial level set function
/// toward image edges using the GAC PDE.
///
/// Args:
///     image:              Input PyImage.
///     initial_phi:        Initial level set function PyImage (same shape as image).
///                         φ < 0 inside the initial contour, φ > 0 outside.
///     propagation_weight: Balloon force ν (expansion if > 0). Default 1.0.
///     curvature_weight:   Weight on curvature regularisation. Default 1.0.
///     advection_weight:   Weight on ∇g·∇φ edge attraction. Default 1.0.
///     edge_k:             Edge stopping sensitivity parameter k. Default 1.0.
///     sigma:              Gaussian pre-smoothing σ for gradient. Default 1.0.
///     dt:                 Euler forward time step Δt. Default 0.05.
///     max_iterations:     Maximum PDE iterations. Default 200.
///
/// Returns:
///     Binary mask PyImage (1.0 where φ < 0, 0.0 elsewhere).
///
/// Raises:
///     RuntimeError: if image and initial_phi shapes do not match.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, propagation_weight=1.0, curvature_weight=1.0, advection_weight=1.0, edge_k=1.0, sigma=1.0, dt=0.05, max_iterations=200))]
pub fn geodesic_active_contour_segment(
    image: &PyImage,
    initial_phi: &PyImage,
    propagation_weight: f64,
    curvature_weight: f64,
    advection_weight: f64,
    edge_k: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
) -> PyResult<PyImage> {
    let mut seg = GeodesicActiveContourSegmentation::new();
    seg.propagation_weight = propagation_weight;
    seg.curvature_weight = curvature_weight;
    seg.advection_weight = advection_weight;
    seg.edge_k = edge_k;
    seg.sigma = sigma;
    seg.dt = dt;
    seg.max_iterations = max_iterations;
    let result = seg
        .apply(image.inner.as_ref(), initial_phi.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── shape_detection_segment ──────────────────────────────────────────────────
/// Shape-detection level set segmentation.
///
/// Uses a speed function that slows evolution at edges detected by a
/// gradient-magnitude filter (Canny edges), enabling detection of topological
/// changes during iteration.
///
/// Args:
/// image: Input PyImage.
/// initial_phi: Initial level set function (signed distance).
/// curvature_weight: Weight of curvature term (default 1.0).
/// propagation_weight: Weight of propagation term (default 1.0).
/// advection_weight: Weight of advection term (default 1.0).
/// edge_k: K parameter for edge potential (default 1.0).
/// sigma: Smoothing sigma for gradient filter (default 1.0).
/// dt: Time step (default 0.05).
/// max_iterations: Maximum iterations (default 200).
/// tolerance: Convergence tolerance (default 1e-3).
///
/// Returns:
/// Evolved level set function as PyImage.
///
/// Raises:
/// RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, curvature_weight=1.0, propagation_weight=1.0, advection_weight=1.0, edge_k=1.0, sigma=1.0, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn shape_detection_segment(
    image: &PyImage,
    initial_phi: &PyImage,
    curvature_weight: f64,
    propagation_weight: f64,
    advection_weight: f64,
    edge_k: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let mut seg = ShapeDetectionSegmentation::new();
    seg.curvature_weight = curvature_weight;
    seg.propagation_weight = propagation_weight;
    seg.advection_weight = advection_weight;
    seg.edge_k = edge_k;
    seg.sigma = sigma;
    seg.dt = dt;
    seg.max_iterations = max_iterations;
    seg.tolerance = tolerance;
    let result = seg
        .apply(image.inner.as_ref(), initial_phi.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── threshold_level_set_segment ─────────────────────────────────────────────
/// Threshold-based level set segmentation.
///
/// Evolves a level set using a speed function derived from intensity
/// thresholds. The region between lower_threshold and upper_threshold has
/// zero speed; outside this band propagation occurs.
///
/// Args:
/// image: Input PyImage.
/// initial_phi: Initial level set function (signed distance).
/// lower_threshold: Lower intensity threshold.
/// upper_threshold: Upper intensity threshold.
/// propagation_weight: Weight of propagation term (default 1.0).
/// curvature_weight: Weight of curvature term (default 0.2).
/// dt: Time step (default 0.05).
/// max_iterations: Maximum iterations (default 200).
/// tolerance: Convergence tolerance (default 1e-3).
///
/// Returns:
/// Evolved level set function as PyImage.
///
/// Raises:
/// RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, lower_threshold, upper_threshold, propagation_weight=1.0, curvature_weight=0.2, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn threshold_level_set_segment(
    image: &PyImage,
    initial_phi: &PyImage,
    lower_threshold: f64,
    upper_threshold: f64,
    propagation_weight: f64,
    curvature_weight: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let mut seg = ThresholdLevelSet::new(lower_threshold, upper_threshold);
    seg.propagation_weight = propagation_weight;
    seg.curvature_weight = curvature_weight;
    seg.dt = dt;
    seg.max_iterations = max_iterations;
    seg.tolerance = tolerance;
    let result = seg
        .apply(image.inner.as_ref(), initial_phi.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// -- laplacian_level_set_segment -----------------------------------------

/// Laplacian level set segmentation.
///
/// Evolves a level set using a speed function derived from the image Laplacian.
/// Positive propagation speed is applied where L(I) < 0 (local bright maxima).
///
/// Args:
///     image: Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     propagation_weight: Weight of Laplacian propagation term (default 1.0).
///     curvature_weight: Weight of curvature regularisation term (default 0.2).
///     sigma: Gaussian pre-smoothing standard deviation (default 1.0).
///     dt: Euler time step (default 0.05).
///     max_iterations: Maximum PDE iterations (default 200).
///     tolerance: Convergence tolerance on max|delta phi|/dt (default 1e-3).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, propagation_weight=1.0, curvature_weight=0.2, sigma=1.0, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn laplacian_level_set_segment(
    image: &PyImage,
    initial_phi: &PyImage,
    propagation_weight: f64,
    curvature_weight: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let mut seg = LaplacianLevelSet::new();
    seg.propagation_weight = propagation_weight;
    seg.curvature_weight = curvature_weight;
    seg.sigma = sigma;
    seg.dt = dt;
    seg.max_iterations = max_iterations;
    seg.tolerance = tolerance;
    let result = seg
        .apply(image.inner.as_ref(), initial_phi.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(into_py_image(result))
}

// ── confidence_connected_segment ─────────────────────────────────────────────

/// Confidence-connected region growing (Yanowitz & Bruckstein 1989).
///
/// Iteratively grows a region from a seed voxel, adapting the intensity
/// window based on the running mean ± k·σ of currently-included voxels.
///
/// Args:
///     image:          Input PyImage.
///     seed:           Seed voxel as [z, y, x] integer list.
///     initial_lower:  Initial inclusive lower bound (first iteration, when σ=0).
///     initial_upper:  Initial inclusive upper bound (first iteration, when σ=0).
///     multiplier:     k for the adaptive k·σ window expansion (default 2.5).
///     max_iterations: Maximum region-growing iterations (default 15).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     ValueError:   if seed does not have exactly 3 elements.
///     RuntimeError: on computation failure.
#[pyfunction]
#[pyo3(signature = (image, seed, initial_lower, initial_upper, multiplier=2.5, max_iterations=15))]
pub fn confidence_connected_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: Vec<usize>,
    initial_lower: f32,
    initial_upper: f32,
    multiplier: f32,
    max_iterations: usize,
) -> PyResult<PyImage> {
    if seed.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed must have exactly 3 elements, got {}",
            seed.len()
        )));
    }
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || {
        ConfidenceConnectedFilter::new([seed[0], seed[1], seed[2]], initial_lower, initial_upper)
            .with_multiplier(multiplier)
            .with_max_iterations(max_iterations)
            .apply(inner.as_ref())
    });
    Ok(into_py_image(result))
}

// ── neighborhood_connected_segment ───────────────────────────────────────────

/// Neighbourhood-connected region growing.
///
/// Grows a region from a seed: admits voxels whose rectangular neighbourhood
/// (±radius in each direction) all satisfy the intensity bounds.
///
/// Args:
///     image:  Input PyImage.
///     seed:   Seed voxel as [z, y, x] integer list.
///     lower:  Inclusive lower intensity bound.
///     upper:  Inclusive upper intensity bound.
///     radius: Neighbourhood half-radius (uniform in all 3 axes, default 1 → 3×3×3).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     ValueError:   if seed does not have exactly 3 elements.
///     RuntimeError: on computation failure.
#[pyfunction]
#[pyo3(signature = (image, seed, lower, upper, radius=1))]
pub fn neighborhood_connected_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: Vec<usize>,
    lower: f32,
    upper: f32,
    radius: usize,
) -> PyResult<PyImage> {
    if seed.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed must have exactly 3 elements, got {}",
            seed.len()
        )));
    }
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || {
        NeighborhoodConnectedFilter::new([seed[0], seed[1], seed[2]], lower, upper)
            .with_radius([radius, radius, radius])
            .apply(inner.as_ref())
    });
    Ok(into_py_image(result))
}

// ── skeletonization ───────────────────────────────────────────────────────────

// ── Threshold: Binary ─────────────────────────────────────────────────────────

/// Apply user-specified binary threshold segmentation.
///
/// Classifies voxels in `[lower, upper]` as `inside_value` (default 1.0)
/// and all others as `outside_value` (default 0.0).
///
/// Delegates to `ritk_core::segmentation::BinaryThreshold`.
///
/// Args:
///     image:         Input PyImage.
///     lower:         Lower intensity bound (inclusive). Defaults to f32::NEG_INFINITY.
///     upper:         Upper intensity bound (inclusive). Defaults to f32::INFINITY.
///     inside_value:  Output value for voxels inside the band (default 1.0).
///     outside_value: Output value for voxels outside the band (default 0.0).
///
/// Returns:
///     Binary-labeled PyImage with the same shape as `image`.
///
/// Raises:
///     ValueError: if lower > upper, or if inside_value / outside_value is not finite.
#[pyfunction]
#[pyo3(signature = (image, lower=None, upper=None, inside_value=1.0, outside_value=0.0))]
pub fn binary_threshold_segment(
    py: Python<'_>,
    image: &PyImage,
    lower: Option<f32>,
    upper: Option<f32>,
    inside_value: f32,
    outside_value: f32,
) -> PyResult<PyImage> {
    let lower = lower.unwrap_or(f32::NEG_INFINITY);
    let upper = upper.unwrap_or(f32::INFINITY);
    if lower > upper {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lower bound ({lower}) must be ≤ upper bound ({upper})"
        )));
    }
    if !inside_value.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "inside_value must be finite, got {inside_value}"
        )));
    }
    if !outside_value.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "outside_value must be finite, got {outside_value}"
        )));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        BinaryThreshold::new(lower, upper)
            .with_values(inside_value, outside_value)
            .apply(image.as_ref())
    });
    Ok(into_py_image(result))
}

// ── Skeletonization ───────────────────────────────────────────────────────────

/// Topology-preserving morphological skeletonization.
///
/// Thins a binary mask to its medial axis (skeleton) while preserving
/// connectivity (Zhang-Suen 2D, directional sequential thinning 3D).
///
/// Args:
///     image: Binary mask PyImage (values in {0.0, 1.0}).
///
/// Returns:
///     Binary skeleton mask, same shape and spatial metadata as input.
///
/// Raises:
///     RuntimeError: on computation failure.
#[pyfunction]
pub fn skeletonization(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || Skeletonization::new().apply::<_, 3>(inner.as_ref()));
    Ok(into_py_image(result))
}

// ── label_shape_statistics ───────────────────────────────────────────────────

/// Compute per-label shape statistics from a binary mask.
///
/// Applies [] and returns per-component spatial
/// statistics (voxel count, centroid, bounding box).
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
) -> PyResult<Py<PyList>> {
    if connectivity != 6 && connectivity != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
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

// ── marker_watershed_segment ─────────────────────────────────────────────────

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
) -> PyResult<PyImage> {
    let grad_arc = Arc::clone(&gradient.inner);
    let mark_arc = Arc::clone(&markers.inner);
    let result = py.allow_threads(|| {
        MarkerControlledWatershed::new().apply(grad_arc.as_ref(), mark_arc.as_ref())
    });
    result
        .map(into_py_image)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `segmentation` submodule with all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "segmentation")?;

    // Thresholding
    m.add_function(wrap_pyfunction!(otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(li_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(yen_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(kapur_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(triangle_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(multi_otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold_segment, &m)?)?;

    // Labeling
    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;
    m.add_function(wrap_pyfunction!(label_shape_statistics, &m)?)?;

    // Region growing
    m.add_function(wrap_pyfunction!(connected_threshold_segment, &m)?)?;

    // Clustering
    m.add_function(wrap_pyfunction!(kmeans_segment, &m)?)?;

    // Watershed
    m.add_function(wrap_pyfunction!(watershed_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(marker_watershed_segment, &m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(binary_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_closing, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_fill_holes, &m)?)?;
    m.add_function(wrap_pyfunction!(morphological_gradient, &m)?)?;

    // Level set
    m.add_function(wrap_pyfunction!(chan_vese_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_active_contour_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(shape_detection_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold_level_set_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_level_set_segment, &m)?)?;

    // Region growing (confidence / neighbourhood)
    m.add_function(wrap_pyfunction!(confidence_connected_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_connected_segment, &m)?)?;

    // Skeletonization
    m.add_function(wrap_pyfunction!(skeletonization, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}

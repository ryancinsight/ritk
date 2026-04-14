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
//! - **Binary erosion** → `ritk_core::segmentation::BinaryErosion`
//! - **Binary dilation** → `ritk_core::segmentation::BinaryDilation`
//! - **Binary opening** → `ritk_core::segmentation::BinaryOpening`
//! - **Binary closing** → `ritk_core::segmentation::BinaryClosing`
//! - **Chan-Vese level set** → `ritk_core::segmentation::ChanVeseSegmentation`
//! - **Geodesic Active Contour** → `ritk_core::segmentation::GeodesicActiveContourSegmentation`
//!
//! No algorithm logic is duplicated here; SSOT is maintained in `ritk-core`.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::segmentation::{
    connected_components as core_connected_components,
    connected_threshold as core_connected_threshold, BinaryClosing, BinaryDilation, BinaryErosion,
    BinaryOpening, ChanVeseSegmentation, GeodesicActiveContourSegmentation, KMeansSegmentation,
    KapurThreshold, LiThreshold, MorphologicalOperation, MultiOtsuThreshold, OtsuThreshold,
    TriangleThreshold, WatershedSegmentation, YenThreshold,
};

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
pub fn otsu_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let filter = OtsuThreshold::new();
    let threshold = filter.compute(image.inner.as_ref());
    let mask = filter.apply(image.inner.as_ref());
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
pub fn li_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let filter = LiThreshold::new();
    let threshold = filter.compute(image.inner.as_ref());
    let mask = filter.apply(image.inner.as_ref());
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
pub fn yen_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let filter = YenThreshold::new();
    let threshold = filter.compute(image.inner.as_ref());
    let mask = filter.apply(image.inner.as_ref());
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
pub fn kapur_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let filter = KapurThreshold::new();
    let threshold = filter.compute(image.inner.as_ref());
    let mask = filter.apply(image.inner.as_ref());
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
pub fn triangle_threshold(image: &PyImage) -> PyResult<(f32, PyImage)> {
    let filter = TriangleThreshold::new();
    let threshold = filter.compute(image.inner.as_ref());
    let mask = filter.apply(image.inner.as_ref());
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
pub fn multi_otsu_threshold(image: &PyImage, num_classes: usize) -> PyResult<(Vec<f32>, PyImage)> {
    if num_classes < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_classes must be ≥ 2",
        ));
    }
    let filter = MultiOtsuThreshold::new(num_classes);
    let thresholds = filter.compute(image.inner.as_ref());
    let labeled = filter.apply(image.inner.as_ref());
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
pub fn connected_components(mask: &PyImage, connectivity: u32) -> PyResult<(PyImage, usize)> {
    if connectivity != 6 && connectivity != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "connectivity must be 6 or 26, got {connectivity}"
        )));
    }

    let (label_image, num_components) =
        core_connected_components(mask.inner.as_ref(), connectivity);
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
    let result = core_connected_threshold(image.inner.as_ref(), seed, lower, upper);
    Ok(into_py_image(result))
}

// ── K-Means clustering ───────────────────────────────────────────────────────

/// Segment an image into K clusters via K-Means (Lloyd's algorithm).
///
/// Delegates to `ritk_core::segmentation::KMeansSegmentation` with k-means++
/// initialization. Each output voxel contains its cluster index (0..K−1) as f32.
///
/// Args:
///     image: Input PyImage.
///     k:     Number of clusters (≥ 1). Default 3.
///
/// Returns:
///     Label PyImage with cluster indices.
#[pyfunction]
#[pyo3(signature = (image, k=3))]
pub fn kmeans_segment(image: &PyImage, k: usize) -> PyResult<PyImage> {
    if k < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err("k must be ≥ 1"));
    }
    let seg = KMeansSegmentation::new(k);
    let result = seg.apply(image.inner.as_ref());
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
pub fn watershed_segment(image: &PyImage) -> PyResult<PyImage> {
    let seg = WatershedSegmentation::new();
    let result = seg
        .apply(image.inner.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
pub fn binary_erosion(image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let op = BinaryErosion::new(radius);
    let result = op.apply(image.inner.as_ref());
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
pub fn binary_dilation(image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let op = BinaryDilation::new(radius);
    let result = op.apply(image.inner.as_ref());
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
pub fn binary_opening(image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let op = BinaryOpening::new(radius);
    let result = op.apply(image.inner.as_ref());
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
pub fn binary_closing(image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let op = BinaryClosing::new(radius);
    let result = op.apply(image.inner.as_ref());
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

    // Labeling
    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;

    // Region growing
    m.add_function(wrap_pyfunction!(connected_threshold_segment, &m)?)?;

    // Clustering
    m.add_function(wrap_pyfunction!(kmeans_segment, &m)?)?;

    // Watershed
    m.add_function(wrap_pyfunction!(watershed_segment, &m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(binary_erosion, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_dilation, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_opening, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_closing, &m)?)?;

    // Level set
    m.add_function(wrap_pyfunction!(chan_vese_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_active_contour_segment, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}

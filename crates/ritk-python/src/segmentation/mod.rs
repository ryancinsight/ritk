//! Python-exposed segmentation functions delegating to `ritk_segmentation`.
//!
//! All algorithmic work is performed by the authoritative implementations in
//! `ritk_segmentation`.  No algorithm logic is duplicated here; SSOT is
//! maintained in `ritk-core`.
//!
//! # Submodules
//! - `threshold`:  Otsu, Li, Yen, Kapur, Triangle, Multi-Otsu, binary threshold.
//! - `labeling`:   Connected components, label statistics, K-Means, watershed.
//! - `morphology`: Binary erosion/dilation/opening/closing, fill holes, gradient, skeletonization.
//! - `levelset`:   Chan-Vese, Geodesic Active Contour, Shape Detection, Threshold, Laplacian.
//! - `growing`:    Connected-threshold, confidence-connected, neighbourhood-connected.
//! - `ensemble`:   STAPLE EM consensus labeling and GrowCut interactive segmentation.

mod ensemble;
mod growing;
mod labeling;
mod levelset;
mod morphology;
mod threshold;

pub use ensemble::*;
pub use growing::*;
pub use labeling::*;
pub use levelset::*;
pub use morphology::*;
pub use threshold::*;

use pyo3::prelude::*;

/// Register the `segmentation` submodule with all exposed functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "segmentation")?;

    // Thresholding
    m.add_function(wrap_pyfunction!(otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(li_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(yen_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(kapur_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(triangle_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(isodata_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(moments_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(huang_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(intermodes_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(shanbhag_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(kittler_illingworth_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(renyi_entropy_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(multi_otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(binary_threshold_segment, &m)?)?;

    // Labeling
    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;
    m.add_function(wrap_pyfunction!(scalar_connected_component, &m)?)?;
    m.add_function(wrap_pyfunction!(relabel_components, &m)?)?;
    m.add_function(wrap_pyfunction!(relabel_label_map, &m)?)?;
    m.add_function(wrap_pyfunction!(merge_label_map, &m)?)?;
    m.add_function(wrap_pyfunction!(label_set_dilate, &m)?)?;
    m.add_function(wrap_pyfunction!(label_set_erode, &m)?)?;
    m.add_function(wrap_pyfunction!(toboggan, &m)?)?;
    m.add_function(wrap_pyfunction!(vector_connected_component, &m)?)?;
    m.add_function(wrap_pyfunction!(
        threshold_maximum_connected_components,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(morphological_watershed, &m)?)?;
    m.add_function(wrap_pyfunction!(change_label, &m)?)?;
    m.add_function(wrap_pyfunction!(label_shape_statistics, &m)?)?;

    // Region growing
    m.add_function(wrap_pyfunction!(connected_threshold_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(isolated_connected_segment, &m)?)?;

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
    m.add_class::<PyChanVeseOptions>()?;
    m.add_function(wrap_pyfunction!(chan_vese_segment, &m)?)?;
    m.add_class::<PyGacOptions>()?;
    m.add_function(wrap_pyfunction!(geodesic_active_contour_segment, &m)?)?;
    m.add_class::<PyShapeDetectionOptions>()?;
    m.add_function(wrap_pyfunction!(shape_detection_segment, &m)?)?;
    m.add_class::<PyThresholdLevelSetOptions>()?;
    m.add_function(wrap_pyfunction!(threshold_level_set_segment, &m)?)?;
    m.add_class::<PyLaplacianLevelSetOptions>()?;
    m.add_function(wrap_pyfunction!(laplacian_level_set_segment, &m)?)?;

    // Region growing (confidence / neighbourhood)
    m.add_function(wrap_pyfunction!(confidence_connected_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(vector_confidence_connected_segment, &m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_connected_segment, &m)?)?;

    // Skeletonization
    m.add_function(wrap_pyfunction!(skeletonization, &m)?)?;

    // Ensemble
    m.add_function(wrap_pyfunction!(staple_ensemble, &m)?)?;
    m.add_function(wrap_pyfunction!(multi_label_staple, &m)?)?;
    m.add_function(wrap_pyfunction!(growcut_segment, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}

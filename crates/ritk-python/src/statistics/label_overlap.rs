//! Python bindings for per-label overlap measures.

use crate::image::{with_image_slice, PyImage};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_statistics::label_overlap::label_overlap_measures_from_slices;

/// Compute per-label overlap measures between a predicted label map and ground truth.
///
/// Delegates to `ritk_statistics::label_overlap::label_overlap_measures_from_slices`.
/// Background label (0) is excluded from results.
///
/// Args:
///     prediction:   Label map PyImage (integer labels stored as f32; 0=background).
///     ground_truth: Reference label map PyImage (same shape as prediction).
///
/// Returns:
///     list of dicts sorted ascending by label, each with keys:
///     `label` (int), `dice` (float), `jaccard` (float),
///     `volume_similarity` (float), `false_negative_rate` (float),
///     `false_positive_rate` (float), `sensitivity` (float),
///     `specificity` (float), `predicted_volume` (int),
///     `ground_truth_volume` (int).
///
/// Raises:
///     RuntimeError: if images have different element counts.
#[pyfunction]
pub fn label_overlap_measures(
    py: Python<'_>,
    prediction: &PyImage,
    ground_truth: &PyImage,
) -> PyResult<Py<PyList>> {
    let measures = with_image_slice(prediction.inner.as_ref(), |pred_slice| {
        with_image_slice(ground_truth.inner.as_ref(), |gt_slice| {
            label_overlap_measures_from_slices(pred_slice, gt_slice)
        })
    });
    let list = PyList::empty_bound(py);
    for m in &measures {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", m.label)?;
        dict.set_item("dice", m.dice)?;
        dict.set_item("jaccard", m.jaccard)?;
        dict.set_item("volume_similarity", m.volume_similarity)?;
        dict.set_item("false_negative_rate", m.false_negative_rate)?;
        dict.set_item("false_positive_rate", m.false_positive_rate)?;
        dict.set_item("sensitivity", m.sensitivity)?;
        dict.set_item("specificity", m.specificity)?;
        dict.set_item("predicted_volume", m.predicted_volume)?;
        dict.set_item("ground_truth_volume", m.ground_truth_volume)?;
        list.append(dict)?;
    }
    Ok(list.into())
}

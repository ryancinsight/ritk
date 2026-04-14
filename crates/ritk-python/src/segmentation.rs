//! Python-exposed segmentation functions delegating to `ritk_core::segmentation`.
//!
//! This module is a thin PyO3 binding layer.  All algorithmic work is performed
//! by the authoritative implementations in `ritk_core::segmentation`:
//!
//! - **Otsu thresholding** → `ritk_core::segmentation::OtsuThreshold`
//! - **Connected-component labeling** → `ritk_core::segmentation::connected_components`
//!
//! No algorithm logic is duplicated here; SSOT is maintained in `ritk-core`.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::segmentation::{connected_components as core_connected_components, OtsuThreshold};

// ── otsu_threshold ────────────────────────────────────────────────────────────

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
    let otsu = OtsuThreshold::new();
    let threshold = otsu.compute(image.inner.as_ref());
    let mask = otsu.apply(image.inner.as_ref());
    Ok((threshold, into_py_image(mask)))
}

// ── connected_components ──────────────────────────────────────────────────────

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

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `segmentation` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "segmentation")?;
    m.add_function(wrap_pyfunction!(otsu_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

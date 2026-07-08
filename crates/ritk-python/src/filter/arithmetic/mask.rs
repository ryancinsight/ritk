use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::{MaskImageFilter, MaskNegatedImageFilter, MaskedAssignImageFilter};

/// Mask `image` by `mask`: keep where mask > 0, else `outside_value`.
/// ITK Parity: MaskImageFilter.
#[pyfunction]
#[pyo3(signature = (image, mask, outside_value = 0.0))]
pub fn mask_image(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    let msk = py_image_to_burn(mask);
    py.allow_threads(|| {
        MaskImageFilter::new()
            .with_outside_value(outside_value)
            .apply(&img, &msk)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Assign `assign_value` where `mask > 0`; keep `image` elsewhere (the role-
/// inverse of `mask_image`). ITK Parity: MaskedAssignImageFilter (`sitk.MaskedAssign`).
#[pyfunction]
#[pyo3(signature = (image, mask, assign_value = 0.0))]
pub fn masked_assign(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    assign_value: f32,
) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    let msk = py_image_to_burn(mask);
    py.allow_threads(|| {
        MaskedAssignImageFilter::new(assign_value)
            .apply(&img, &msk)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Mask `image` by the negation of `mask`: keep where mask ≤ 0, else
/// `outside_value`. ITK Parity: MaskNegatedImageFilter.
#[pyfunction]
#[pyo3(signature = (image, mask, outside_value = 0.0))]
pub fn mask_negated_image(
    py: Python<'_>,
    image: &PyImage,
    mask: &PyImage,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    let msk = py_image_to_burn(mask);
    py.allow_threads(|| {
        MaskNegatedImageFilter::new()
            .with_outside_value(outside_value)
            .apply(&img, &msk)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

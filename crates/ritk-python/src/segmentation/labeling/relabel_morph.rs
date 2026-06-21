use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::{
    label_set_morph as core_label_set_morph, merge_label_maps as core_merge_label_maps,
    relabel_consecutive as core_relabel_consecutive, LabelSetMorphOp, MergeLabelMethod,
    RelabelComponentFilter,
};
use std::sync::Arc;

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

/// Relabel non-zero labels to consecutive integers `1, 2, …, K` in ascending
/// original-label order (background 0 unchanged).
///
/// ITK Parity: matches `sitk.RelabelLabelMap` (via the LabelMap round-trip
/// `LabelMapToLabel(RelabelLabelMap(LabelImageToLabelMap(img)))`). Unlike
/// [`relabel_components`] (size-descending), this assigns new labels in the
/// order of ascending existing label values.
///
/// Args:
///     label_image: integer label image (0 = background).
///
/// Returns:
///     the relabelled image.
#[pyfunction]
#[pyo3(signature = (label_image))]
pub fn relabel_label_map(py: Python<'_>, label_image: &PyImage) -> PyImage {
    let img = Arc::clone(&label_image.inner);
    let out = py.allow_threads(|| core_relabel_consecutive(img.as_ref()));
    into_py_image(out)
}

/// Merge several label images into one, matching
/// `sitk.LabelMapToLabel(sitk.MergeLabelMap([…], method))`.
///
/// Each input's distinct non-zero values become label objects; the inputs are
/// folded into the first under one of four methods.
///
/// ITK Parity: `MergeLabelMapFilter` (`sitk.MergeLabelMap`).
///
/// Args:
///     label_images: list of integer label images, identical dimensions.
///     method: 0 = Keep, 1 = Aggregate, 2 = Pack, 3 = Strict (default 0).
///
/// Returns:
///     the merged label image.
#[pyfunction]
#[pyo3(signature = (label_images, method=0))]
pub fn merge_label_map(
    py: Python<'_>,
    label_images: Vec<PyRef<'_, PyImage>>,
    method: u8,
) -> RitkResult<PyImage> {
    let m = match method {
        0 => MergeLabelMethod::Keep,
        1 => MergeLabelMethod::Aggregate,
        2 => MergeLabelMethod::Pack,
        3 => MergeLabelMethod::Strict,
        other => {
            return Err(RitkPyError::value(format!(
                "merge_label_map: method must be 0..=3, got {other}"
            )))
        }
    };
    // Clone the Arc handles so the GIL can be released during compute.
    let arcs: Vec<Arc<ritk_image::Image<crate::image::Backend, 3>>> =
        label_images.iter().map(|p| Arc::clone(&p.inner)).collect();
    let out = py.allow_threads(|| {
        let refs: Vec<&ritk_image::Image<crate::image::Backend, 3>> =
            arcs.iter().map(|a| a.as_ref()).collect();
        core_merge_label_maps(&refs, m).map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok(into_py_image(out))
}

/// Label-preserving Euclidean dilation, matching `sitk.LabelSetDilate`.
///
/// Grows every label region by a Euclidean structuring element of per-axis
/// `radius` (SimpleITK order `[x, y, z]`); overlaps resolve to the nearer region.
///
/// ITK Parity: `LabelSetDilateImageFilter`.
///
/// Args:
///     label_image: integer label image (0 = background).
///     radius: per-axis radius; scalar broadcast or `[rx, ry, rz]` (default 1).
///     use_image_spacing: radius in world units (default True) or voxels.
#[pyfunction]
#[pyo3(signature = (label_image, radius=vec![1.0, 1.0, 1.0], use_image_spacing=true))]
pub fn label_set_dilate(
    py: Python<'_>,
    label_image: &PyImage,
    radius: Vec<f64>,
    use_image_spacing: bool,
) -> RitkResult<PyImage> {
    label_set_morph_py(
        py,
        label_image,
        radius,
        use_image_spacing,
        LabelSetMorphOp::Dilate,
    )
}

/// Label-preserving Euclidean erosion, matching `sitk.LabelSetErode`.
///
/// Shrinks every label region by a Euclidean structuring element of per-axis
/// `radius` (SimpleITK order `[x, y, z]`); only voxels at least `radius` from
/// their region boundary survive.
///
/// ITK Parity: `LabelSetErodeImageFilter`.
///
/// Args:
///     label_image: integer label image (0 = background).
///     radius: per-axis radius; scalar broadcast or `[rx, ry, rz]` (default 1).
///     use_image_spacing: radius in world units (default True) or voxels.
#[pyfunction]
#[pyo3(signature = (label_image, radius=vec![1.0, 1.0, 1.0], use_image_spacing=true))]
pub fn label_set_erode(
    py: Python<'_>,
    label_image: &PyImage,
    radius: Vec<f64>,
    use_image_spacing: bool,
) -> RitkResult<PyImage> {
    label_set_morph_py(
        py,
        label_image,
        radius,
        use_image_spacing,
        LabelSetMorphOp::Erode,
    )
}

/// Shared driver for [`label_set_dilate`]/[`label_set_erode`]: normalize the
/// SimpleITK `radius` argument (scalar or 2-/3-vector in `[x, y, z]` order) and
/// dispatch to the core filter.
fn label_set_morph_py(
    py: Python<'_>,
    label_image: &PyImage,
    radius: Vec<f64>,
    use_image_spacing: bool,
    op: LabelSetMorphOp,
) -> RitkResult<PyImage> {
    let radius_itk: [f64; 3] = match radius.len() {
        1 => [radius[0], radius[0], radius[0]],
        2 => [radius[0], radius[1], 0.0],
        3 => [radius[0], radius[1], radius[2]],
        other => {
            return Err(RitkPyError::value(format!(
                "label_set radius must have 1, 2, or 3 elements, got {other}"
            )))
        }
    };
    let img = Arc::clone(&label_image.inner);
    let out =
        py.allow_threads(|| core_label_set_morph(img.as_ref(), radius_itk, use_image_spacing, op));
    Ok(into_py_image(out))
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

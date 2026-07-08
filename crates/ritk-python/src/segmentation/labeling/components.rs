use crate::errors::{RitkPyError, RitkResult};
use crate::image::{
    burn_into_py_image, into_py_image, py_image_to_burn, vec_to_image_like, with_image_slice,
    PyImage,
};
use pyo3::prelude::*;
use ritk_segmentation::{
    connected_components as core_connected_components, labeling::Connectivity as SegConnectivity,
    scalar_connected_components as core_scalar_connected_components,
    vector_connected_components_image as core_vector_connected_components,
    ThresholdMaximumConnectedComponentsFilter,
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

    let mask = py_image_to_burn(mask);
    let (label_image, num_components) = {
        let seg_conn = if connectivity == 6 {
            SegConnectivity::Six
        } else {
            SegConnectivity::TwentySix
        };
        py.allow_threads(|| core_connected_components(&mask, seg_conn))
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
        let vals = with_image_slice(arc.as_ref(), |slice| slice.to_vec());
        let labels =
            core_scalar_connected_components(&vals, dims, distance_threshold, connectivity);
        vec_to_image_like(labels, dims, arc.as_ref())
    });
    Ok(into_py_image(out))
}

/// Vector connected-component labeling, matching `sitk.VectorConnectedComponent`.
///
/// Labels a multi-channel (vector) image: two face- or fully-connected
/// neighbours join when `1 − |a · b| ≤ distance_threshold` over their channel
/// vectors (ITK assumes the vectors are normalized).  The component **partition**
/// matches SimpleITK; label integers are renumbered consecutively (the standard
/// connected-component parity convention).
///
/// ITK Parity: `VectorConnectedComponentImageFilter`.
///
/// Args:
///     channels: list of scalar component images (one per vector component),
///         all identical dimensions.
///     distance_threshold: direction-similarity threshold (default 1.0).
///     fully_connected: 26-/8-connectivity if True, else face (default False).
///
/// Returns:
///     consecutive label image.
#[pyfunction]
#[pyo3(signature = (channels, distance_threshold=1.0, fully_connected=false))]
pub fn vector_connected_component(
    py: Python<'_>,
    channels: Vec<PyRef<'_, PyImage>>,
    distance_threshold: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    if channels.is_empty() {
        return Err(RitkPyError::value(
            "vector_connected_component: at least one channel is required",
        ));
    }
    let burns: Vec<_> = channels.iter().map(|p| py_image_to_burn(p)).collect();
    let conn = if fully_connected { 26 } else { 6 };
    let out = py.allow_threads(|| {
        let refs: Vec<_> = burns.iter().collect();
        core_vector_connected_components(&refs, distance_threshold, conn)
    });
    Ok(burn_into_py_image(out))
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
    let img = py_image_to_burn(image);
    let out = py.allow_threads(|| {
        ThresholdMaximumConnectedComponentsFilter {
            minimum_object_size,
            upper_boundary,
            inside_value: 1.0,
            outside_value: 0.0,
        }
        .apply(&img)
    });
    burn_into_py_image(out)
}

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{
    burn_into_py_image, native_into_py_image, py_image_to_burn, py_image_to_native, PyImage,
};
use coeus_core::SequentialBackend;
use pyo3::prelude::*;
use ritk_segmentation::{
    toboggan as core_toboggan, FloodConnectivity, MarkerControlledWatershed,
    MorphologicalWatershed, WatershedLinePolicy, WatershedSegmentation,
};

/// Toboggan watershed labeling, matching `sitk.Toboggan`.
///
/// Each voxel slides along a face-connected steepest-descent path to a local
/// minimum; voxels reaching the same minimum share a label (≥ 2, assigned in
/// raster discovery order).
///
/// ITK Parity: `TobogganImageFilter`.
///
/// Args:
///     image: scalar relief image (typically a gradient magnitude).
///
/// Returns:
///     label image (basin indices ≥ 2).
#[pyfunction]
pub fn toboggan(py: Python<'_>, image: &PyImage) -> PyImage {
    let arc = py_image_to_burn(image);
    let out = py.allow_threads(|| core_toboggan(&arc));
    burn_into_py_image(out)
}

/// Marker-less morphological watershed, matching
/// `SimpleITK.MorphologicalWatershed(level, markWatershedLine=True,
/// fullyConnected=False)`.
///
/// Floods the relief from its own regional minima (after suppressing minima
/// shallower than `level` via h-minima). Watershed-line voxels are label 0; face
/// connectivity. Bit-exact to sitk for the default markWatershedLine/connectivity.
///
/// Args:
///     image: Input relief (typically a gradient magnitude).
///     level: Depth below which shallow minima merge (default 0.0).
///
/// Returns:
///     Label PyImage (basin indices ≥ 1; 0 = watershed line / unreachable).
#[pyfunction]
#[pyo3(signature = (image, level=0.0_f32))]
pub fn morphological_watershed(py: Python<'_>, image: &PyImage, level: f32) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        MorphologicalWatershed::new(level)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
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
    let image = py_image_to_native(image)?;
    py.allow_threads(|| {
        let seg = WatershedSegmentation::new();
        seg.apply_native(&image, &SequentialBackend)
            .map_err(|e| RitkPyError::value(e.to_string()))
    })
    .map(native_into_py_image)
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
#[pyo3(signature = (gradient, markers, mark_watershed_line=true, fully_connected=false))]
pub fn marker_watershed_segment(
    py: Python<'_>,
    gradient: &PyImage,
    markers: &PyImage,
    mark_watershed_line: bool,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let gradient = py_image_to_native(gradient)?;
    let markers = py_image_to_native(markers)?;
    let connectivity = if fully_connected {
        FloodConnectivity::Full
    } else {
        FloodConnectivity::Face
    };
    let watershed_lines = if mark_watershed_line {
        WatershedLinePolicy::Mark
    } else {
        WatershedLinePolicy::Omit
    };
    let result = py.allow_threads(|| {
        MarkerControlledWatershed::new()
            .with_connectivity(connectivity)
            .with_watershed_lines(watershed_lines)
            .apply_native(&gradient, &markers, &SequentialBackend)
    });
    result
        .map(native_into_py_image)
        .map_err(|e| RitkPyError::value(e.to_string()))
}

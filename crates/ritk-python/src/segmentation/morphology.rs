//! Binary morphological operations: erosion, dilation, opening, closing, fill holes,
//! gradient, and skeletonization.

use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::{
    BinaryClosing, BinaryDilation, BinaryErosion, BinaryFillHoles, BinaryOpening,
    MorphologicalGradient, MorphologicalOperation, Skeletonization,
};

/// Apply binary erosion with a box structuring element.
///
/// Delegates to `ritk_segmentation::BinaryErosion`. For each voxel p,
/// `output[p]` = 1.0 iff all voxels within the axis-aligned hypercube of
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
pub fn binary_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyImage {
    let image = image_from_py(image);
    let result = py.allow_threads(|| {
        let op = BinaryErosion::new(radius);
        op.apply(&image)
    });
    into_py_image(result)
}

/// Apply binary dilation with a box structuring element.
///
/// Delegates to `ritk_segmentation::BinaryDilation`. For each voxel p,
/// `output[p]` = 1.0 iff any voxel within the axis-aligned hypercube of
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
pub fn binary_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyImage {
    let image = image_from_py(image);
    let result = py.allow_threads(|| {
        let op = BinaryDilation::new(radius);
        op.apply(&image)
    });
    into_py_image(result)
}

/// Apply binary opening (erosion followed by dilation).
///
/// Delegates to `ritk_segmentation::BinaryOpening`. Removes small
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
pub fn binary_opening(py: Python<'_>, image: &PyImage, radius: usize) -> PyImage {
    let image = image_from_py(image);
    let result = py.allow_threads(|| {
        let op = BinaryOpening::new(radius);
        op.apply(&image)
    });
    into_py_image(result)
}

/// Apply binary closing (dilation followed by erosion).
///
/// Delegates to `ritk_segmentation::BinaryClosing`. Fills small
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
pub fn binary_closing(py: Python<'_>, image: &PyImage, radius: usize) -> PyImage {
    let image = image_from_py(image);
    let result = py.allow_threads(|| {
        let op = BinaryClosing::new(radius);
        op.apply(&image)
    });
    into_py_image(result)
}

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
pub fn binary_fill_holes(py: Python<'_>, image: &PyImage) -> PyImage {
    let inner = image_from_py(image);
    let result = py.allow_threads(move || BinaryFillHoles.apply(&inner));
    into_py_image(result)
}

/// Compute the morphological gradient (boundary extraction) of a binary mask.
///
/// Output is 1.0 at boundary voxels (in dilation but not erosion) and 0.0
/// at interior foreground, exterior background, and all other voxels.
///
/// Uses a box (hypercube) structuring element of half-width `radius`, matching
/// `sitk.MorphologicalGradient(..., sitk.sitkBox)` exactly (the binary
/// dilation/erosion it is built from are box-SE, like the rest of ritk's binary
/// morphology — *not* a ball).
///
/// Args:
///     image: Binary mask PyImage (values in {0.0, 1.0}).
///     radius: Box structuring-element half-width in voxels (default: 1).
///
/// Returns:
///     Binary boundary mask, same shape and spatial metadata as input.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn morphological_gradient(py: Python<'_>, image: &PyImage, radius: usize) -> PyImage {
    let inner = image_from_py(image);
    let result = py.allow_threads(move || MorphologicalGradient::new(radius).apply(&inner));
    into_py_image(result)
}

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
pub fn skeletonization(py: Python<'_>, image: &PyImage) -> PyImage {
    let inner = image_from_py(image);
    let result = py.allow_threads(move || Skeletonization::new().apply::<_, 3>(&inner));
    into_py_image(result)
}

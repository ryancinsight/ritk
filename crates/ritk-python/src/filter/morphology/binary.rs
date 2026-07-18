use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, into_py_image, py_image_to_burn, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::{
    BinaryPruningFilter, BinaryThinningFilter, ErodeObjectMorphologyFilter, HitOrMissTransform,
    VotingBinaryHoleFillingImageFilter, VotingBinaryImageFilter };
use std::sync::Arc;

/// Thin a binary image to its 1-pixel-wide skeleton, matching
/// `SimpleITK.BinaryThinning` (2-D Gonzalez & Woods thinning).
///
/// Input is binarized (`â‰  0 â†’ 1`) and iteratively thinned per `z`-plane until
/// stable. Output is binary (`1.0` skeleton, `0.0` background). 2-D filter:
/// apply to a `z = 1` image (each plane is thinned independently otherwise).
///
/// Args:
///     image: Input binary PyImage.
///
/// Returns:
///     Skeleton PyImage, same shape and spatial metadata as input.
#[pyfunction]
pub fn binary_thinning(py: Python<'_>, image: &PyImage) -> PyImage {
    // TODO: BinaryThinningFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    let out = py.allow_threads(|| BinaryThinningFilter::new().apply(&arc));
    burn_into_py_image(out)
}

/// Prune short spurs from a binary skeleton, matching `SimpleITK.BinaryPruning`
/// (2-D). Each of `iteration` raster-order in-place sweeps removes foreground
/// pixels with fewer than two 8-neighbours (endpoints / isolated pixels). Output
/// is binary; apply to a `z = 1` image. ITK default `iteration = 3`.
///
/// Args:
///     image:     Input binary PyImage.
///     iteration: Number of pruning sweeps (default 3).
///
/// Returns:
///     Pruned PyImage, same shape and spatial metadata as input.
#[pyfunction]
#[pyo3(signature = (image, iteration=3))]
pub fn binary_pruning(py: Python<'_>, image: &PyImage, iteration: usize) -> PyImage {
    // TODO: BinaryPruningFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    let out = py.allow_threads(|| BinaryPruningFilter::new(iteration).apply(&arc));
    burn_into_py_image(out)
}

/// Erode an object's surface with a box structuring element, matching
/// `SimpleITK.ErodeObjectMorphology` (box kernel).
///
/// Object voxels (== `object_value`) on the object boundary â€” having any 3Ã—3Ã—3
/// neighbour that is not the object value, with out-of-image neighbours treated
/// as non-object â€” paint their `(2r+1)Â³` box footprint with `background_value`.
/// Unlike grayscale erosion, this erodes objects that touch the image border.
///
/// Args:
///     image:            Input PyImage.
///     radius:           Box structuring-element radius per axis (default 1).
///     object_value:     Pixel value identifying the object (default 1.0).
///     background_value: Value written to eroded voxels (default 0.0).
///
/// Returns:
///     Eroded PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, radius=1, object_value=1.0_f32, background_value=0.0_f32))]
pub fn erode_object_morphology(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    object_value: f32,
    background_value: f32,
) -> PyImage {
    // TODO: ErodeObjectMorphologyFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    let out = py.allow_threads(|| {
        ErodeObjectMorphologyFilter::new([radius, radius, radius], object_value, background_value)
            .apply(&arc)
    });
    burn_into_py_image(out)
}

/// Apply hit-or-miss transform for shape detection in binary images.
///
/// Args:
///     image:     Binary input PyImage.
///     fg_radius: Foreground structuring element radius.
///     bg_radius: Background structuring element radius.
#[pyfunction]
#[pyo3(signature = (image, fg_radius=1_usize, bg_radius=2_usize))]
pub fn hit_or_miss(
    py: Python<'_>,
    image: &PyImage,
    fg_radius: usize,
    bg_radius: usize,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        HitOrMissTransform::new(fg_radius, bg_radius)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// One voting (cellular-automaton) step on a binary image: a background voxel
/// becomes foreground if â‰¥ `birth_threshold` of its `(2Â·radius+1)Â³` neighbours
/// are foreground; a foreground voxel survives if â‰¥ `survival_threshold` are.
/// ITK Parity: VotingBinaryImageFilter (`sitk.VotingBinary`).
#[pyfunction]
#[pyo3(signature = (image, radius = 1, birth_threshold = 1, survival_threshold = 1, foreground_value = 1.0, background_value = 0.0))]
pub fn voting_binary(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    birth_threshold: usize,
    survival_threshold: usize,
    foreground_value: f32,
    background_value: f32,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        VotingBinaryImageFilter::new(
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value,
            background_value,
        )
        .apply_native(native.as_ref(), &backend)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Fill background holes by majority vote: a background voxel becomes foreground
/// when â‰¥ `(Wâˆ’1)/2 + majority_threshold` of its `(2Â·radius+1)Â³` neighbours
/// (clamp boundary, `W` = full window) are foreground; foreground always
/// survives. ITK Parity: VotingBinaryHoleFillingImageFilter (`sitk.VotingBinaryHoleFilling`).
#[pyfunction]
#[pyo3(signature = (image, radius = 1, majority_threshold = 1, foreground_value = 1.0, background_value = 0.0))]
pub fn voting_binary_hole_filling(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    majority_threshold: usize,
    foreground_value: f32,
    background_value: f32,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        VotingBinaryHoleFillingImageFilter::new(
            [radius, radius, radius],
            majority_threshold,
            foreground_value,
            background_value,
        )
        .apply_native(native.as_ref(), &backend)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Iteratively fill background holes by majority vote, repeating the
/// hole-filling pass up to `max_iterations` times (stopping early when nothing
/// changes). ITK Parity: VotingBinaryIterativeHoleFillingImageFilter
/// (`sitk.VotingBinaryIterativeHoleFilling`).
#[pyfunction]
#[pyo3(signature = (image, radius = 1, max_iterations = 10, majority_threshold = 1, foreground_value = 1.0, background_value = 0.0))]
pub fn voting_binary_iterative_hole_filling(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    max_iterations: usize,
    majority_threshold: usize,
    foreground_value: f32,
    background_value: f32,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        VotingBinaryHoleFillingImageFilter::new(
            [radius, radius, radius],
            majority_threshold,
            foreground_value,
            background_value,
        )
        .apply_iterative_native(native.as_ref(), max_iterations, &backend)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

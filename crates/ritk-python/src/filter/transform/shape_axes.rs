use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    FlipImageFilter, ShrinkImageFilter, ExpandImageFilter, CyclicShiftImageFilter,
    PermuteAxesImageFilter, OrientImageFilter,
};

/// Flip the image along any combination of the Z, Y, X axes.
/// ITK Parity: FlipImageFilter (`sitk.Flip` with `flipAxes` reversed to `[x,y,z]`).
#[pyfunction]
#[pyo3(signature = (image, flip_z = false, flip_y = false, flip_x = false))]
pub fn flip(
    py: Python<'_>,
    image: &PyImage,
    flip_z: bool,
    flip_y: bool,
    flip_x: bool,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        FlipImageFilter::from_bools([flip_z, flip_y, flip_x])
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Subsample (ITK `Shrink`) by integer per-axis factors. Keeps one voxel per
/// tile (offset `f/2`, `floor(N/f)` output), scales spacing, and shifts the
/// origin to the first tile centroid. No averaging (cf. `bin_shrink`).
/// ITK Parity: ShrinkImageFilter (`sitk.Shrink`, factors `[fx,fy,fz]`).
#[pyfunction]
#[pyo3(signature = (image, factor_z=2, factor_y=2, factor_x=2))]
pub fn shrink(
    py: Python<'_>,
    image: &PyImage,
    factor_z: usize,
    factor_y: usize,
    factor_x: usize,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ShrinkImageFilter::new([factor_z, factor_y, factor_x])
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Expand (upsample) the image by integer per-axis factors `(fz, fy, fx)` using
/// linear interpolation on the ITK Expand grid (spacing/factor, half-voxel
/// origin shift, edge-clamp). ITK Parity: ExpandImageFilter (`sitk.Expand`,
/// factors `[fx,fy,fz]`).
#[pyfunction]
pub fn expand(py: Python<'_>, image: &PyImage, factors: (usize, usize, usize)) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        ExpandImageFilter::new([factors.0, factors.1, factors.2]).apply(arc.as_ref())
    });
    into_py_image(out)
}

/// Cyclically roll the image by `shift = (z, y, x)` voxels (periodic wrap-around,
/// no data loss). ITK Parity: CyclicShiftImageFilter (`sitk.CyclicShift`, `[x,y,z]`).
#[pyfunction]
pub fn cyclic_shift(
    py: Python<'_>,
    image: &PyImage,
    shift: (i64, i64, i64),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        CyclicShiftImageFilter::new([shift.0, shift.1, shift.2]).apply(arc.as_ref())
    });
    Ok(into_py_image(out))
}

/// Permute the tensor axes. `order` is a permutation of `(0, 1, 2)` in `[z,y,x]`
/// tensor space. ITK Parity: PermuteAxesImageFilter (`sitk.PermuteAxes`).
#[pyfunction]
pub fn permute_axes(
    py: Python<'_>,
    image: &PyImage,
    order: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        PermuteAxesImageFilter::new([order.0, order.1, order.2])
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Reorient the image to a target DICOM orientation code (`"LPS"`, `"RAI"`, …),
/// relabeling the axes consistently across data, spacing, origin, and direction.
/// ITK Parity: DICOMOrientImageFilter (`sitk.DICOMOrient`).
#[pyfunction]
pub fn dicom_orient(py: Python<'_>, image: &PyImage, orientation: &str) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let filter =
        OrientImageFilter::from_code(orientation).map_err(|e| RitkPyError::value(e.to_string()))?;
    py.allow_threads(|| {
        filter
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

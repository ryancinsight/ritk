//! Geometric image-transform filters: flip, pad (constant/mirror/wrap),
//! shrink, region-of-interest (crop), permute-axes, paste.
//!
//! All index/size arguments use ritk's tensor-axis order `[z, y, x]` (the same
//! convention as `bin_shrink`); SimpleITK's equivalents take `[x, y, z]`, so a
//! caller bridging to sitk reverses the tuple. See the axis-order note.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    ConstantPadImageFilter, FlipImageFilter, MirrorPadImageFilter, Padding, PasteImageFilter,
    PermuteAxesImageFilter, RegionOfInterestImageFilter, WrapPadImageFilter,
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

/// Pad the image with a constant value. `lower`/`upper` are `(z, y, x)` voxel
/// counts. ITK Parity: ConstantPadImageFilter (`sitk.ConstantPad`).
#[pyfunction]
#[pyo3(signature = (image, lower, upper, constant = 0.0))]
pub fn constant_pad(
    py: Python<'_>,
    image: &PyImage,
    lower: (usize, usize, usize),
    upper: (usize, usize, usize),
    constant: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let lo = Padding([lower.0, lower.1, lower.2]);
    let up = Padding([upper.0, upper.1, upper.2]);
    py.allow_threads(|| {
        ConstantPadImageFilter::new(lo, up, constant)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pad the image by mirroring at the borders. ITK Parity: MirrorPadImageFilter
/// (`sitk.MirrorPad`).
#[pyfunction]
#[pyo3(signature = (image, lower, upper))]
pub fn mirror_pad(
    py: Python<'_>,
    image: &PyImage,
    lower: (usize, usize, usize),
    upper: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let lo = Padding([lower.0, lower.1, lower.2]);
    let up = Padding([upper.0, upper.1, upper.2]);
    py.allow_threads(|| {
        MirrorPadImageFilter::new(lo, up)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Pad the image by wrapping (periodic tiling) at the borders. ITK Parity:
/// WrapPadImageFilter (`sitk.WrapPad`).
#[pyfunction]
#[pyo3(signature = (image, lower, upper))]
pub fn wrap_pad(
    py: Python<'_>,
    image: &PyImage,
    lower: (usize, usize, usize),
    upper: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let lo = Padding([lower.0, lower.1, lower.2]);
    let up = Padding([upper.0, upper.1, upper.2]);
    py.allow_threads(|| {
        WrapPadImageFilter::new(lo, up)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Crop to a sub-region: `start` and `size` are `(z, y, x)` voxels. ITK Parity:
/// RegionOfInterestImageFilter (`sitk.RegionOfInterest` with `[x,y,z]`).
#[pyfunction]
#[pyo3(signature = (image, start, size))]
pub fn region_of_interest(
    py: Python<'_>,
    image: &PyImage,
    start: (usize, usize, usize),
    size: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        RegionOfInterestImageFilter::new([start.0, start.1, start.2], [size.0, size.1, size.2])
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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

/// Paste `source` into a copy of `dest` at `dest_start` `(z, y, x)`. ITK Parity:
/// PasteImageFilter (`sitk.Paste`).
#[pyfunction]
pub fn paste(
    py: Python<'_>,
    dest: &PyImage,
    source: &PyImage,
    dest_start: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let d = std::sync::Arc::clone(&dest.inner);
    let s = std::sync::Arc::clone(&source.inner);
    py.allow_threads(|| {
        PasteImageFilter::new([dest_start.0, dest_start.1, dest_start.2])
            .apply(d.as_ref(), s.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

//! Geometric image-transform filters: flip, pad (constant/mirror/wrap),
//! shrink, region-of-interest (crop), permute-axes, paste.
//!
//! All index/size arguments use ritk's tensor-axis order `[z, y, x]` (the same
//! convention as `bin_shrink`); SimpleITK's equivalents take `[x, y, z]`, so a
//! caller bridging to sitk reverses the tuple. See the axis-order note.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_filter::{
    ConstantPadImageFilter, CyclicShiftImageFilter, ExpandImageFilter, FlipImageFilter,
    MirrorPadImageFilter, Padding, PasteImageFilter, PermuteAxesImageFilter,
    RegionOfInterestImageFilter, WrapPadImageFilter,
};
use ritk_image::Image;

/// Stack a list of images along the Z axis (concatenate `[zᵢ, Y, X]` volumes
/// into `[Σzᵢ, Y, X]`). All inputs must share the same `Y`/`X` extent.
///
/// ITK Parity: JoinSeriesImageFilter (`sitk.JoinSeries`, which stacks N 2-D
/// slices into a 3-D volume along the new last axis = ritk's Z).
#[pyfunction]
pub fn join_series(py: Python<'_>, images: Vec<Py<PyImage>>) -> RitkResult<PyImage> {
    if images.is_empty() {
        return Err(RitkPyError::value("join_series: needs at least one image"));
    }
    let arcs: Vec<_> = images
        .iter()
        .map(|p| std::sync::Arc::clone(&p.bind(py).borrow().inner))
        .collect();
    let [_, ny, nx] = arcs[0].shape();
    let mut total_z = 0usize;
    let mut data: Vec<f32> = Vec::new();
    for (i, a) in arcs.iter().enumerate() {
        let [zi, yi, xi] = a.shape();
        if yi != ny || xi != nx {
            return Err(RitkPyError::value(format!(
                "join_series: image {i} has Y/X [{yi},{xi}], expected [{ny},{nx}]"
            )));
        }
        total_z += zi;
        data.extend_from_slice(&a.data_slice());
    }
    let device = NdArrayDevice::default();
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(data, Shape::new([total_z, ny, nx])),
        &device,
    );
    let out = Image::new(
        tensor,
        *arcs[0].origin(),
        *arcs[0].spacing(),
        *arcs[0].direction(),
    );
    Ok(into_py_image(out))
}

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

/// Crop the image by removing `lower` and `upper` voxels from each axis face.
/// `lower`/`upper` are `(z, y, x)` voxel counts. ITK Parity:
/// CropImageFilter (`sitk.Crop`, with `[x,y,z]` boundary sizes).
#[pyfunction]
#[pyo3(signature = (image, lower, upper))]
pub fn crop(
    py: Python<'_>,
    image: &PyImage,
    lower: (usize, usize, usize),
    upper: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let [nz, ny, nx] = image.inner.shape();
    let start = [lower.0, lower.1, lower.2];
    let (uz, uy, ux) = upper;
    if lower.0 + uz >= nz || lower.1 + uy >= ny || lower.2 + ux >= nx {
        return Err(RitkPyError::value(format!(
            "crop: lower+upper {:?}+{:?} leaves no extent in shape [{},{},{}]",
            start,
            [uz, uy, ux],
            nz,
            ny,
            nx
        )));
    }
    let size = [nz - lower.0 - uz, ny - lower.1 - uy, nx - lower.2 - ux];
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        RegionOfInterestImageFilter::new(start, size)
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

/// Expand (upsample) the image by integer per-axis factors `(fz, fy, fx)` using
/// linear interpolation on the ITK Expand grid (spacing/factor, half-voxel
/// origin shift, edge-clamp). ITK Parity: ExpandImageFilter (`sitk.Expand`,
/// factors `[fx,fy,fz]`).
#[pyfunction]
pub fn expand(py: Python<'_>, image: &PyImage, factors: (usize, usize, usize)) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py
        .allow_threads(|| ExpandImageFilter::new([factors.0, factors.1, factors.2]).apply(arc.as_ref()));
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

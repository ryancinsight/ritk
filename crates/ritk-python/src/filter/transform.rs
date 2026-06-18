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
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{
    gaussian_image_source as core_gaussian_image_source,
    grid_image_source as core_grid_image_source, ConstantPadImageFilter,
    CyclicShiftImageFilter, ExpandImageFilter, FlipImageFilter, MirrorPadImageFilter, Padding,
    PasteImageFilter, PermuteAxesImageFilter, RegionOfInterestImageFilter, ShrinkImageFilter,
    WrapPadImageFilter, ZeroFluxNeumannPadImageFilter,
};
use ritk_image::Image;

/// Extract a strided sub-region (numpy-style `start:stop:step` per axis).
/// `start`/`stop`/`step` are `(z, y, x)` in ritk tensor order; each axis keeps
/// indices `start, start+step, … < stop` (clamped to `[0, dim]`, `step ≥ 1`).
///
/// ITK Parity: SliceImageFilter (`sitk.Slice`, with `[x,y,z]` parameters).
#[pyfunction]
#[pyo3(signature = (image, start, stop, step))]
pub fn slice_image(
    py: Python<'_>,
    image: &PyImage,
    start: (i64, i64, i64),
    stop: (i64, i64, i64),
    step: (i64, i64, i64),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let [nz, ny, nx] = arc.shape();
    let axis_indices = |st: i64, sp: i64, stp: i64, n: usize| -> Vec<usize> {
        let st = st.clamp(0, n as i64);
        let sp = sp.clamp(0, n as i64);
        let stp = stp.max(1);
        let mut idx = Vec::new();
        let mut i = st;
        while i < sp {
            idx.push(i as usize);
            i += stp;
        }
        idx
    };
    let zi = axis_indices(start.0, stop.0, step.0, nz);
    let yi = axis_indices(start.1, stop.1, step.1, ny);
    let xi = axis_indices(start.2, stop.2, step.2, nx);
    let (oz, oy, ox) = (zi.len(), yi.len(), xi.len());
    if oz == 0 || oy == 0 || ox == 0 {
        return Err(RitkPyError::value(
            "slice: empty result (check start/stop/step)".to_string(),
        ));
    }
    let out = py.allow_threads(|| {
        let data = arc.data_slice();
        let mut out = vec![0.0_f32; oz * oy * ox];
        for (a, &z) in zi.iter().enumerate() {
            for (b, &y) in yi.iter().enumerate() {
                for (c, &x) in xi.iter().enumerate() {
                    out[a * oy * ox + b * ox + c] = data[z * ny * nx + y * nx + x];
                }
            }
        }
        out
    });
    // Shift the origin to the first kept voxel (per-axis spacing).
    let sp = arc.spacing().to_array();
    let orig = arc.origin().to_array();
    let new_origin = ritk_core::spatial::Point::new([
        orig[0] + zi[0] as f64 * sp[0],
        orig[1] + yi[0] as f64 * sp[1],
        orig[2] + xi[0] as f64 * sp[2],
    ]);
    let device = NdArrayDevice::default();
    let tensor =
        Tensor::<Backend, 3>::from_data(TensorData::new(out, Shape::new([oz, oy, ox])), &device);
    Ok(into_py_image(Image::new(
        tensor,
        new_origin,
        *arc.spacing(),
        *arc.direction(),
    )))
}

/// Combine two same-sized images in a checkerboard pattern: `pattern = (nx, ny,
/// nz)` gives the number of checker cells per axis (sitk x/y/z). A voxel takes
/// `image1` where the sum of its cell indices is even, else `image2`.
///
/// ITK Parity: CheckerBoardImageFilter (`sitk.CheckerBoard`).
#[pyfunction]
#[pyo3(signature = (image1, image2, pattern=(4, 4, 4)))]
pub fn checker_board(
    py: Python<'_>,
    image1: &PyImage,
    image2: &PyImage,
    pattern: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let a = std::sync::Arc::clone(&image1.inner);
    let b = std::sync::Arc::clone(&image2.inner);
    let [nz, ny, nx] = a.shape();
    if b.shape() != [nz, ny, nx] {
        return Err(RitkPyError::value(format!(
            "checker_board: shapes differ ({:?} vs {:?})",
            [nz, ny, nx],
            b.shape()
        )));
    }
    let (px, py_, pz) = pattern; // sitk x, y, z cells
    let out = py.allow_threads(|| {
        let da = a.data_slice();
        let db = b.data_slice();
        let mut out = vec![0.0_f32; nz * ny * nx];
        for z in 0..nz {
            let cz = z * pz / nz;
            for y in 0..ny {
                let cy = y * py_ / ny;
                for x in 0..nx {
                    let cx = x * px / nx;
                    let i = z * ny * nx + y * nx + x;
                    out[i] = if (cx + cy + cz) % 2 == 0 { da[i] } else { db[i] };
                }
            }
        }
        out
    });
    let device = NdArrayDevice::default();
    let tensor =
        Tensor::<Backend, 3>::from_data(TensorData::new(out, Shape::new([nz, ny, nx])), &device);
    Ok(into_py_image(Image::new(
        tensor,
        *a.origin(),
        *a.spacing(),
        *a.direction(),
    )))
}

/// Tile (montage) a list of same-sized images into a grid. `layout = (nx, ny, nz)`
/// gives the number of tiles along each axis (sitk x/y/z convention); images fill
/// the grid in x-then-y-then-z order. Empty cells take `default_value`.
///
/// ITK Parity: TileImageFilter (`sitk.Tile`, same-size inputs, explicit layout).
#[pyfunction]
#[pyo3(signature = (images, layout, default_value=0.0))]
pub fn tile(
    py: Python<'_>,
    images: Vec<Py<PyImage>>,
    layout: (usize, usize, usize),
    default_value: f32,
) -> RitkResult<PyImage> {
    if images.is_empty() {
        return Err(RitkPyError::value("tile: needs at least one image"));
    }
    let (nx, ny, nz) = layout;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(RitkPyError::value("tile: layout dimensions must be ≥ 1"));
    }
    let arcs: Vec<_> = images
        .iter()
        .map(|p| std::sync::Arc::clone(&p.bind(py).borrow().inner))
        .collect();
    let [hz, hy, hx] = arcs[0].shape();
    for (i, a) in arcs.iter().enumerate() {
        if a.shape() != [hz, hy, hx] {
            return Err(RitkPyError::value(format!(
                "tile: image {i} shape {:?} != {:?} (same-size tiling only)",
                a.shape(),
                [hz, hy, hx]
            )));
        }
    }
    // Output grid: nz tiles in z, ny in y, nx in x.
    let (oz, oy, ox) = (nz * hz, ny * hy, nx * hx);
    let out = py.allow_threads(|| {
        let mut out = vec![default_value; oz * oy * ox];
        for (i, a) in arcs.iter().enumerate() {
            let tz = i / (nx * ny);
            if tz >= nz {
                break; // beyond the grid capacity
            }
            let ty = (i / nx) % ny;
            let tx = i % nx;
            let data = a.data_slice();
            for z in 0..hz {
                for y in 0..hy {
                    for x in 0..hx {
                        let oi = (tz * hz + z) * oy * ox + (ty * hy + y) * ox + (tx * hx + x);
                        out[oi] = data[z * hy * hx + y * hx + x];
                    }
                }
            }
        }
        out
    });
    let device = NdArrayDevice::default();
    let tensor =
        Tensor::<Backend, 3>::from_data(TensorData::new(out, Shape::new([oz, oy, ox])), &device);
    Ok(into_py_image(Image::new(
        tensor,
        *arcs[0].origin(),
        *arcs[0].spacing(),
        *arcs[0].direction(),
    )))
}

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

/// Pad the image by replicating the edge voxel (zero-flux Neumann). ITK Parity:
/// ZeroFluxNeumannPadImageFilter (`sitk.ZeroFluxNeumannPad`).
#[pyfunction]
#[pyo3(signature = (image, lower, upper))]
pub fn zero_flux_neumann_pad(
    py: Python<'_>,
    image: &PyImage,
    lower: (usize, usize, usize),
    upper: (usize, usize, usize),
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let lo = Padding([lower.0, lower.1, lower.2]);
    let up = Padding([upper.0, upper.1, upper.2]);
    py.allow_threads(|| {
        ZeroFluxNeumannPadImageFilter::new(lo, up)
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

/// Generate a Gaussian blob image (`itk::GaussianImageSource` / `sitk.GaussianSource`).
///
/// `out(index) = scale · exp(−½ · Σ_d ((origin_d + index_d·spacing_d − mean_d)/sigma_d)²)`
/// (non-normalised; peak value = `scale`). All `(x, y, z)` tuples are in sitk
/// axis order; the produced image carries the given spacing/origin (identity
/// direction). ITK Parity: GaussianImageSource.
#[pyfunction]
#[pyo3(signature = (size, sigma, mean, scale=255.0, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)))]
pub fn gaussian_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    sigma: (f64, f64, f64),
    mean: (f64, f64, f64),
    scale: f64,
    origin: (f64, f64, f64),
    spacing: (f64, f64, f64),
) -> PyImage {
    let (buf, dims) = py.allow_threads(|| {
        core_gaussian_image_source(
            [size.0, size.1, size.2],
            [sigma.0, sigma.1, sigma.2],
            [mean.0, mean.1, mean.2],
            scale,
            [origin.0, origin.1, origin.2],
            [spacing.0, spacing.1, spacing.2],
        )
    });
    let device = NdArrayDevice::default();
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(buf, Shape::new(dims)), &device);
    // ritk metadata is axis-major [z, y, x]; reverse the sitk (x, y, z) tuples.
    let image = Image::new(
        tensor,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
    );
    into_py_image(image)
}

/// Generate a grid-pattern image (`itk::GridImageSource` / `sitk.GridSource`):
/// dark periodic Gaussian lines on a bright background,
/// `out = scale·Π_{selected d}(1 − Σ_lines exp(−(p_d−line)²/(2σ_d²)))`.
/// All `(x, y, z)` tuples are in sitk axis order. ITK Parity: GridImageSource.
#[pyfunction]
#[pyo3(signature = (size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), sigma=(0.5, 0.5, 0.5), grid_spacing=(4.0, 4.0, 4.0), grid_offset=(0.0, 0.0, 0.0), scale=255.0, which_dimensions=(true, true, true)))]
#[allow(clippy::too_many_arguments)]
pub fn grid_image_source(
    py: Python<'_>,
    size: (usize, usize, usize),
    spacing: (f64, f64, f64),
    origin: (f64, f64, f64),
    sigma: (f64, f64, f64),
    grid_spacing: (f64, f64, f64),
    grid_offset: (f64, f64, f64),
    scale: f64,
    which_dimensions: (bool, bool, bool),
) -> PyImage {
    let (buf, dims) = py.allow_threads(|| {
        core_grid_image_source(
            [size.0, size.1, size.2],
            [spacing.0, spacing.1, spacing.2],
            [origin.0, origin.1, origin.2],
            [sigma.0, sigma.1, sigma.2],
            [grid_spacing.0, grid_spacing.1, grid_spacing.2],
            [grid_offset.0, grid_offset.1, grid_offset.2],
            scale,
            [which_dimensions.0, which_dimensions.1, which_dimensions.2],
        )
    });
    let device = NdArrayDevice::default();
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(buf, Shape::new(dims)), &device);
    let image = Image::new(
        tensor,
        Point::new([origin.2, origin.1, origin.0]),
        Spacing::new([spacing.2, spacing.1, spacing.0]),
        Direction::identity(),
    );
    into_py_image(image)
}

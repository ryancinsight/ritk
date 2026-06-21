use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use pyo3::prelude::*;
use ritk_filter::PasteImageFilter;
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
                    out[i] = if (cx + cy + cz) % 2 == 0 {
                        da[i]
                    } else {
                        db[i]
                    };
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

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    ConstantPadImageFilter, FftPadBoundary, FftPadImageFilter, MirrorPadImageFilter,
    Padding, RegionOfInterestImageFilter, WrapPadImageFilter, ZeroFluxNeumannPadImageFilter,
};

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

/// Pad each axis to the next size whose greatest prime factor is `<= max_prime`
/// (default 5), for efficient FFT. `boundary` selects the fill: 0 = zero, 1 =
/// zero-flux Neumann (edge replicate, default), 2 = periodic (wrap). ITK Parity:
/// FFTPadImageFilter (`sitk.FFTPad`).
#[pyfunction]
#[pyo3(signature = (image, max_prime=5, boundary=1))]
pub fn fft_pad(
    py: Python<'_>,
    image: &PyImage,
    max_prime: usize,
    boundary: u8,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let bc = match boundary {
        0 => FftPadBoundary::Zero,
        1 => FftPadBoundary::ZeroFluxNeumann,
        2 => FftPadBoundary::Periodic,
        other => {
            return Err(RitkPyError::value(format!(
                "fft_pad: boundary must be 0 (zero), 1 (zero-flux Neumann), or 2 (periodic); got {other}"
            )))
        }
    };
    py.allow_threads(|| {
        FftPadImageFilter::new(max_prime, bc)
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

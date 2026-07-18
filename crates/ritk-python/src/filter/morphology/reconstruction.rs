use crate::errors::{RitkPyError, RitkResult};
use crate::image::{
    burn_into_py_image, native_into_py_image, py_image_to_burn, py_image_to_native, PyImage };
use coeus_core::SequentialBackend;
use pyo3::prelude::*;
use ritk_filter::{
    ClosingByReconstructionFilter, GrayscaleFillholeFilter, GrayscaleGrindPeakFilter,
    HConcaveFilter, HConvexFilter, HMaximaFilter, HMinimaFilter, MorphologicalReconstruction,
    OpeningByReconstructionFilter, ReconstructionMode, RegionalMaximaFilter, RegionalMinimaFilter,
    ValuedRegionalMaximaFilter, ValuedRegionalMinimaFilter };

use super::contour::connectivity_from;

/// Geodesic morphological reconstruction.
///
/// Args:
///     marker: seed image (marker â‰¤ mask for dilation, marker â‰¥ mask for erosion).
///     mask: constraint image.
///     mode: "dilation" or "erosion".
///     fully_connected: if False (default), use face connectivity (6-connected in
///         3-D, 4-connected in 2-D) to match ITK's default; if True, use full
///         connectivity (26/8-connected, including diagonals).
#[pyfunction]
#[pyo3(signature = (marker, mask, mode = "dilation", fully_connected = false))]
pub fn morphological_reconstruction(
    py: Python<'_>,
    marker: &PyImage,
    mask: &PyImage,
    mode: &str,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let recon_mode = match mode {
        "dilation" => ReconstructionMode::Dilation,
        "erosion" => ReconstructionMode::Erosion,
        other => {
            return Err(RitkPyError::value(format!(
                "Unknown reconstruction mode '{}'. Use 'dilation' or 'erosion'.",
                other
            )))
        }
    };
    let connectivity = connectivity_from(fully_connected);
    let marker_arc = py_image_to_burn(marker);
    let mask_arc = py_image_to_burn(mask);
    py.allow_threads(|| {
        MorphologicalReconstruction::new(recon_mode)
            .with_connectivity(connectivity)
            .apply(&marker_arc, &mask_arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// H-maxima transform: suppress bright regional maxima with contrast below
/// `height`. ITK Parity: HMaximaImageFilter (`sitk.HMaxima`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_maxima(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HMaximaFilter::new(height)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// H-minima transform: suppress dark regional minima with contrast below
/// `height`. ITK Parity: HMinimaImageFilter (`sitk.HMinima`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_minima(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let image = py_image_to_native(image)?;
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HMinimaFilter::new(height)
            .with_connectivity(conn)
            .apply_native(&image, &SequentialBackend)
            .map_err(|error| RitkPyError::value(error.to_string()))
    })
    .map(native_into_py_image)
}

/// H-convex transform: `f âˆ’ HMaxima_h(f)`, the bright dynamic suppressed by
/// the h-maxima transform. ITK Parity: HConvexImageFilter (`sitk.HConvex`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_convex(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HConvexFilter::new(height)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// H-concave transform: `HMinima_h(f) âˆ’ f`, the dark dynamic suppressed by the
/// h-minima transform. ITK Parity: HConcaveImageFilter (`sitk.HConcave`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_concave(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HConcaveFilter::new(height)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Binary regional maxima: `foreground` on regional maxima, `background`
/// elsewhere. ITK Parity: RegionalMaximaImageFilter (`sitk.RegionalMaxima`).
#[pyfunction]
#[pyo3(signature = (image, foreground = 1.0, background = 0.0, fully_connected = false))]
pub fn regional_maxima(
    py: Python<'_>,
    image: &PyImage,
    foreground: f32,
    background: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        RegionalMaximaFilter::new()
            .with_values(foreground, background)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Binary regional minima. ITK Parity: RegionalMinimaImageFilter
/// (`sitk.RegionalMinima`).
#[pyfunction]
#[pyo3(signature = (image, foreground = 1.0, background = 0.0, fully_connected = false))]
pub fn regional_minima(
    py: Python<'_>,
    image: &PyImage,
    foreground: f32,
    background: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let image = py_image_to_native(image)?;
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        RegionalMinimaFilter::new()
            .with_values(foreground, background)
            .with_connectivity(conn)
            .apply_native(&image, &SequentialBackend)
            .map_err(|error| RitkPyError::value(error.to_string()))
    })
    .map(native_into_py_image)
}

/// Valued regional maxima: keep the input value on regional maxima, set
/// non-maxima to âˆ’FLT_MAX. ITK Parity: ValuedRegionalMaximaImageFilter
/// (`sitk.ValuedRegionalMaxima`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false))]
pub fn valued_regional_maxima(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ValuedRegionalMaximaFilter::new()
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Valued regional minima: keep the input value on regional minima, set
/// non-minima to +FLT_MAX. ITK Parity: ValuedRegionalMinimaImageFilter
/// (`sitk.ValuedRegionalMinima`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false))]
pub fn valued_regional_minima(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ValuedRegionalMinimaFilter::new()
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Opening by reconstruction: erode with a box SE of half-width `radius`, then
/// reconstruct under the input by dilation. ITK Parity:
/// OpeningByReconstructionImageFilter (`sitk.OpeningByReconstruction`, box SE).
#[pyfunction]
#[pyo3(signature = (image, radius, fully_connected = false))]
pub fn opening_by_reconstruction(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        OpeningByReconstructionFilter::new(radius)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Closing by reconstruction: dilate with a box SE of half-width `radius`, then
/// reconstruct under the input by erosion. ITK Parity:
/// ClosingByReconstructionImageFilter (`sitk.ClosingByReconstruction`, box SE).
#[pyfunction]
#[pyo3(signature = (image, radius, fully_connected = false))]
pub fn closing_by_reconstruction(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ClosingByReconstructionFilter::new(radius)
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Grayscale fill-hole: raise dark regional minima not connected to the image
/// border. ITK Parity: GrayscaleFillholeImageFilter (`sitk.GrayscaleFillhole`).
#[pyfunction]
pub fn grayscale_fillhole(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        GrayscaleFillholeFilter::new()
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Grayscale grind-peak: grind down bright peaks not connected to the image
/// border (dual of fill-hole). ITK Parity: GrayscaleGrindPeakImageFilter
/// (`sitk.GrayscaleGrindPeak`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false))]
pub fn grayscale_grind_peak(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        GrayscaleGrindPeakFilter::new()
            .with_connectivity(conn)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

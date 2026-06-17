//! Morphological filters: grayscale erosion/dilation, label morphology, top-hat, hit-or-miss.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    BinaryContourImageFilter, BlackTopHatFilter, ClosingByReconstructionFilter, Connectivity,
    GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion, GrayscaleFillholeFilter,
    GrayscaleGrindPeakFilter, GrayscaleOpeningFilter, HConcaveFilter, HConvexFilter, HMaximaFilter,
    HMinimaFilter, HitOrMissTransform, LabelClosing, LabelContourImageFilter, LabelDilation,
    LabelErosion, LabelOpening, MorphologicalReconstruction, OpeningByReconstructionFilter,
    ReconstructionMode, RegionalMaximaFilter, RegionalMinimaFilter, ValuedRegionalMaximaFilter,
    ValuedRegionalMinimaFilter, VotingBinaryImageFilter, WhiteTopHatFilter,
};

/// Apply grayscale morphological erosion with a flat cubic structuring element.
///
/// Each output voxel is the minimum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Eroded PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GrayscaleErosion::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply grayscale morphological dilation with a flat cubic structuring element.
///
/// Each output voxel is the maximum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Dilated PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GrayscaleDilation::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        LabelErosion::new(radius)
            .apply(img.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        LabelOpening::new(radius)
            .apply(img.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        LabelClosing::new(radius)
            .apply(img.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Dilate labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        LabelDilation::new(radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply white top-hat transform (image minus morphological opening).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn white_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        WhiteTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply black top-hat transform (morphological closing minus image).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn black_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        BlackTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        HitOrMissTransform::new(fg_radius, bg_radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Geodesic morphological reconstruction.
///
/// Args:
///     marker: seed image (marker ≤ mask for dilation, marker ≥ mask for erosion).
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
    let marker_arc = std::sync::Arc::clone(&marker.inner);
    let mask_arc = std::sync::Arc::clone(&mask.inner);
    py.allow_threads(|| {
        MorphologicalReconstruction::new(recon_mode)
            .with_connectivity(connectivity)
            .apply(marker_arc.as_ref(), mask_arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HMaximaFilter::new(height)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HMinimaFilter::new(height)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// H-convex transform: `f − HMaxima_h(f)`, the bright dynamic suppressed by
/// the h-maxima transform. ITK Parity: HConvexImageFilter (`sitk.HConvex`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_convex(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HConvexFilter::new(height)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// H-concave transform: `HMinima_h(f) − f`, the dark dynamic suppressed by the
/// h-minima transform. ITK Parity: HConcaveImageFilter (`sitk.HConcave`).
#[pyfunction]
#[pyo3(signature = (image, height, fully_connected = false))]
pub fn h_concave(
    py: Python<'_>,
    image: &PyImage,
    height: f32,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        HConcaveFilter::new(height)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        RegionalMaximaFilter::new()
            .with_values(foreground, background)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        RegionalMinimaFilter::new()
            .with_values(foreground, background)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Valued regional maxima: keep the input value on regional maxima, set
/// non-maxima to −FLT_MAX. ITK Parity: ValuedRegionalMaximaImageFilter
/// (`sitk.ValuedRegionalMaxima`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false))]
pub fn valued_regional_maxima(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ValuedRegionalMaximaFilter::new()
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ValuedRegionalMinimaFilter::new()
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        OpeningByReconstructionFilter::new(radius)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        ClosingByReconstructionFilter::new(radius)
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Grayscale morphological closing with a flat cubic (box) SE of half-width
/// `radius`. ITK Parity: GrayscaleMorphologicalClosingImageFilter
/// (`sitk.GrayscaleMorphologicalClosing`, box SE).
#[pyfunction]
pub fn grayscale_closing(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        GrayscaleClosingFilter::new(radius)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Grayscale morphological opening with a flat cubic (box) SE of half-width
/// `radius`. ITK Parity: GrayscaleMorphologicalOpeningImageFilter
/// (`sitk.GrayscaleMorphologicalOpening`, box SE).
#[pyfunction]
pub fn grayscale_opening(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        GrayscaleOpeningFilter::new(radius)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Grayscale fill-hole: raise dark regional minima not connected to the image
/// border. ITK Parity: GrayscaleFillholeImageFilter (`sitk.GrayscaleFillhole`).
#[pyfunction]
pub fn grayscale_fillhole(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        GrayscaleFillholeFilter::new()
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        GrayscaleGrindPeakFilter::new()
            .with_connectivity(conn)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Binary contour: keep only foreground voxels that touch the background
/// (the object boundary). ITK Parity: BinaryContourImageFilter
/// (`sitk.BinaryContour`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false, foreground_value = 1.0))]
pub fn binary_contour(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
    foreground_value: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        BinaryContourImageFilter::new(conn, foreground_value)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Label contour: mark the boundary voxels of every labelled region (voxels
/// adjacent to a different label or the background). ITK Parity:
/// LabelContourImageFilter (`sitk.LabelContour`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false, background_value = 0.0))]
pub fn label_contour(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
    background_value: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        LabelContourImageFilter::new(conn, background_value)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// One voting (cellular-automaton) step on a binary image: a background voxel
/// becomes foreground if ≥ `birth_threshold` of its `(2·radius+1)³` neighbours
/// are foreground; a foreground voxel survives if ≥ `survival_threshold` are.
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
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        VotingBinaryImageFilter::new(
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value,
            background_value,
        )
        .apply(arc.as_ref())
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Map the `fully_connected` flag to the structuring-element adjacency
/// (ITK's `FullyConnectedOff` → face, `On` → full).
fn connectivity_from(fully_connected: bool) -> Connectivity {
    if fully_connected {
        Connectivity::Vertex26
    } else {
        Connectivity::Face6
    }
}

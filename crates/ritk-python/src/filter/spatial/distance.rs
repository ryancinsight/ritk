use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::SignedDistanceTransformImageFilter;
use ritk_segmentation::DistanceTransform;

/// Distance metric variant for distance transform, replacing `squared: bool`.
///
/// Eliminates boolean blindness: `metric="euclidean"` vs `metric="squared"` is
/// self-documenting compared to `squared=True/False`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDistanceMetric {
    /// Euclidean distance (sqrt of sum of squares).
    Euclidean,
    /// Squared Euclidean distance (no sqrt; faster, preserves differentiability).
    Squared,
}

impl<'py> FromPyObject<'py> for PyDistanceMetric {
    fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "euclidean" => Ok(Self::Euclidean),
            "squared" => Ok(Self::Squared),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown distance metric '{}'. Choices: euclidean, squared",
                other
            ))),
        }
    }
}

/// Compute the Euclidean (or squared Euclidean) distance transform of a binary image.
///
/// For each background voxel the output is the distance to the nearest foreground
/// voxel **in physical units** (image spacing applied per axis); foreground
/// voxels receive 0.0. Implements the exact O(N) Meijster et al. (2000) algorithm
/// and is float-exact to scipy `distance_transform_edt(sampling=spacing)` on the
/// inverted mask. (SimpleITK's `DanielssonDistanceMap` / `SignedMaurerDistanceMap`
/// also use physical units but the opposite foreground sense, and Danielsson is an
/// approximate vector transform.)
///
/// Args:
/// image: Input binary image (foreground > foreground_threshold).
/// foreground_threshold: Threshold above which a voxel is foreground (default 0.5).
/// metric: Distance metric: "euclidean" (default) or "squared".
///     "euclidean" returns true Euclidean distances (with sqrt).
///     "squared" returns squared distances (no sqrt; faster, preserves differentiability).
///
/// Returns:
/// Distance image with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, foreground_threshold=0.5_f32, metric=PyDistanceMetric::Euclidean))]
pub fn distance_transform(
    py: Python<'_>,
    image: &PyImage,
    foreground_threshold: f32,
    metric: PyDistanceMetric,
) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| match metric {
        PyDistanceMetric::Euclidean => {
            DistanceTransform::transform(arc.as_ref(), foreground_threshold)
        }
        PyDistanceMetric::Squared => DistanceTransform::squared(arc.as_ref(), foreground_threshold),
    });
    into_py_image(result)
}

/// Signed Euclidean distance map of a binary image (physical units).
///
/// Foreground voxels (value > `foreground_threshold`) receive the **negative**
/// distance to the nearest background voxel (inside the object); background
/// voxels receive the **positive** distance to the nearest foreground voxel.
///
/// Float-exact to `scipy.ndimage.distance_transform_edt` (signed, voxel-centre
/// convention). NOTE: this is distance to the nearest opposite-class voxel
/// **centre** — it does NOT match `sitk.SignedMaurerDistanceMap`, which measures
/// distance to the object boundary/interface (differs by up to √2 voxel).
#[pyfunction]
#[pyo3(signature = (image, foreground_threshold=0.5_f32))]
pub fn signed_distance_map(
    py: Python<'_>,
    image: &PyImage,
    foreground_threshold: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        SignedDistanceTransformImageFilter::new()
            .with_threshold(foreground_threshold)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

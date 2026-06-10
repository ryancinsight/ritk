//! Spatial filters: resample image to new spacing, Euclidean distance transform.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use pyo3::prelude::*;
use ritk_core::filter::ResampleImageFilter;
use ritk_core::interpolation::LinearInterpolator;
use ritk_core::interpolation::{
    BSplineInterpolator, Lanczos4Interpolator, NearestNeighborInterpolator,
};
use ritk_core::segmentation::DistanceTransform;
use ritk_core::spatial::Spacing as CoreSpacing;
use ritk_core::transform::affine::translation::TranslationTransform;

/// Resample a 3-D image to new voxel spacing.
///
/// Output size N_d_prime = max(1, round(N_d * S_d / S_d_prime)).
///
/// Args:
///     image:     Input PyImage.
///     spacing_z: Output voxel spacing along Z axis (default 1.0 mm).
///     spacing_y: Output voxel spacing along Y axis (default 1.0 mm).
///     spacing_x: Output voxel spacing along X axis (default 1.0 mm).
///     mode:      Interpolation mode: "nearest", "linear" (default), "bspline", "lanczos4".
///
/// Returns:
///     Resampled PyImage with updated spacing and shape.
///
/// Raises:
///     ValueError: if any spacing is ≤ 0 or mode is unrecognized.
#[pyfunction]
#[pyo3(signature = (image, spacing_z=1.0_f64, spacing_y=1.0_f64, spacing_x=1.0_f64, mode="linear"))]
pub fn resample_image(
    py: Python<'_>,
    image: &PyImage,
    spacing_z: f64,
    spacing_y: f64,
    spacing_x: f64,
    mode: &str,
) -> RitkResult<PyImage> {
    if spacing_z <= 0.0 || spacing_y <= 0.0 || spacing_x <= 0.0 {
        return Err(RitkPyError::value(format!(
            "spacing values must be positive, got ({spacing_z},{spacing_y},{spacing_x})"
        )));
    }
    let mode = mode.to_string();
    let inner = std::sync::Arc::clone(&image.inner);

    py.allow_threads(move || -> Result<_, String> {
        let orig_dims = inner.shape();
        let orig_sp = *inner.spacing();
        let orig_orig = *inner.origin();
        let orig_dir = *inner.direction();

        let new_nz = ((orig_dims[0] as f64 * orig_sp[0]) / spacing_z)
            .round()
            .max(1.0) as usize;
        let new_ny = ((orig_dims[1] as f64 * orig_sp[1]) / spacing_y)
            .round()
            .max(1.0) as usize;
        let new_nx = ((orig_dims[2] as f64 * orig_sp[2]) / spacing_x)
            .round()
            .max(1.0) as usize;

        let new_sp = CoreSpacing::new([spacing_z, spacing_y, spacing_x]);
        let device: <Backend as BurnBackend>::Device = Default::default();
        let zero_t = Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![0.0f32; 3], Shape::new([3])),
            &device,
        );

        match mode.as_str() {
            "nearest" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx],
                orig_orig,
                new_sp,
                orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                NearestNeighborInterpolator::new(),
            )
            .apply(inner.as_ref())),
            "linear" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx],
                orig_orig,
                new_sp,
                orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                LinearInterpolator::new(),
            )
            .apply(inner.as_ref())),
            "bspline" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx],
                orig_orig,
                new_sp,
                orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                BSplineInterpolator::new(),
            )
            .apply(inner.as_ref())),
            "lanczos4" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx],
                orig_orig,
                new_sp,
                orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                Lanczos4Interpolator::new(),
            )
            .apply(inner.as_ref())),
            other => Err(format!(
                "Unknown interpolation mode '{}'. Use: nearest, linear, bspline, lanczos4",
                other
            )),
        }
    })
    .map_err(RitkPyError::value)
    .map(into_py_image)
}

/// Compute the Euclidean (or squared Euclidean) distance transform of a binary image.
///
/// For each background voxel the output is the distance to the nearest foreground
/// voxel (in physical units, respecting image spacing).  Foreground voxels receive 0.0.
/// Implements the exact O(N) Meijster et al. (2000) algorithm.
///
/// Args:
///     image:                Input binary image (foreground > foreground_threshold).
///     foreground_threshold: Threshold above which a voxel is foreground (default 0.5).
///     squared:              If True, return squared distances (no sqrt; default False).
///
/// Returns:
///     Distance image with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, foreground_threshold=0.5_f32, squared=false))]
pub fn distance_transform(
    py: Python<'_>,
    image: &PyImage,
    foreground_threshold: f32,
    squared: bool,
) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        if squared {
            DistanceTransform::squared(arc.as_ref(), foreground_threshold)
        } else {
            DistanceTransform::transform(arc.as_ref(), foreground_threshold)
        }
    });
    into_py_image(result)
}

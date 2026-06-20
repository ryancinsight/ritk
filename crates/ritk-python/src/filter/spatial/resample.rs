use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use pyo3::prelude::*;
use ritk_filter::ResampleImageFilter;
use ritk_interpolation::LinearInterpolator;
use ritk_interpolation::{BSplineInterpolator, Lanczos5Interpolator, NearestNeighborInterpolator};
use ritk_spatial::Spacing as CoreSpacing;
use ritk_transform::affine::translation::TranslationTransform;

/// Resample a 3-D image to new voxel spacing.
///
/// Output size N_d_prime = max(1, round(N_d * S_d / S_d_prime)).
///
/// Args:
///     image:     Input PyImage.
///     spacing_z: Output voxel spacing along Z axis (default 1.0 mm).
///     spacing_y: Output voxel spacing along Y axis (default 1.0 mm).
///     spacing_x: Output voxel spacing along X axis (default 1.0 mm).
///     mode:      Interpolation mode: "nearest", "linear" (default), "bspline", "lanczos".
///                "lanczos" is the radius-5 windowed-sinc matching
///                SimpleITK's `sitkLanczosWindowedSinc`.
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
            "lanczos" => Ok(ResampleImageFilter::new(
                [new_nz, new_ny, new_nx],
                orig_orig,
                new_sp,
                orig_dir,
                TranslationTransform::<Backend, 3>::new(zero_t),
                Lanczos5Interpolator::new(),
            )
            .apply(inner.as_ref())),
            other => Err(format!(
                "Unknown interpolation mode '{}'. Use: nearest, linear, bspline, lanczos",
                other
            )),
        }
    })
    .map_err(RitkPyError::value)
    .map(into_py_image)
}

/// Zoom a 3-D image by a scale factor (scipy.ndimage.zoom parity).
///
/// Equivalent to `resample_image` with `new_spacing = old_spacing / zoom_factor`
/// and output size `round(old_size * zoom_factor)`.
///
/// A zoom > 1.0 upsamples (finer grid); < 1.0 downsamples (coarser grid).
/// Per-axis zoom factors allow anisotropic rescaling.
///
/// Args:
///     image:    Input PyImage.
///     zoom_z:   Zoom factor along Z axis (default 1.0).
///     zoom_y:   Zoom factor along Y axis (default 1.0).
///     zoom_x:   Zoom factor along X axis (default 1.0).
///     mode:     Interpolation mode — "linear" (default), "nearest",
///               "bspline", "lanczos".
///
/// Returns:
///     Zoomed PyImage with new shape and updated spacing.
///
/// Raises:
///     ValueError: if any zoom factor is ≤ 0 or `mode` is unrecognised.
#[pyfunction]
#[pyo3(signature = (image, zoom_z=1.0_f64, zoom_y=1.0_f64, zoom_x=1.0_f64, mode="linear"))]
pub fn zoom_image(
    py: Python<'_>,
    image: &PyImage,
    zoom_z: f64,
    zoom_y: f64,
    zoom_x: f64,
    mode: &str,
) -> RitkResult<PyImage> {
    if zoom_z <= 0.0 || zoom_y <= 0.0 || zoom_x <= 0.0 {
        return Err(RitkPyError::value(format!(
            "zoom factors must be positive, got ({zoom_z},{zoom_y},{zoom_x})"
        )));
    }
    let inner = std::sync::Arc::clone(&image.inner);
    let sp = *inner.spacing();
    // new_spacing = old_spacing / zoom_factor
    let new_sz = sp[0] / zoom_z;
    let new_sy = sp[1] / zoom_y;
    let new_sx = sp[2] / zoom_x;
    resample_image(py, image, new_sz, new_sy, new_sx, mode)
}

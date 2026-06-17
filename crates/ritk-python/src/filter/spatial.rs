//! Spatial filters: resample image to new spacing, rotate, shift, zoom, and Euclidean distance transform.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use pyo3::prelude::*;
use ritk_filter::{ResampleImageFilter, SignedDistanceTransformImageFilter};
use ritk_interpolation::LinearInterpolator;
use ritk_interpolation::{BSplineInterpolator, Lanczos5Interpolator, NearestNeighborInterpolator};
use ritk_segmentation::DistanceTransform;
use ritk_spatial::Spacing as CoreSpacing;
use ritk_transform::affine::affine::AffineTransform;
use ritk_transform::affine::translation::TranslationTransform;

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

// ── GAP-SCI-01: rotate_image ─────────────────────────────────────────────────

/// Rotate a 3-D image about its geometric centre.
///
/// The result matches SimpleITK's
/// `Euler3DTransform.SetRotation(angle_x, angle_y, angle_z)` (with its default
/// `ComputeZYX = false`, i.e. `R = R_z·R_x·R_y`) about the image centre, for
/// arbitrary single- or multi-axis rotations, spacings, and origins. Each
/// `angle_<axis>` rotates about the corresponding physical axis.
///
/// The output grid is identical to the input grid (same shape, spacing, and
/// origin).  Out-of-bounds voxels receive `default_pixel_value`.
///
/// Args:
///     image:               Input PyImage.
///     angle_x:             Rotation about physical X axis in radians (default 0.0).
///     angle_y:             Rotation about physical Y axis in radians (default 0.0).
///     angle_z:             Rotation about physical Z axis in radians (default 0.0).
///     mode:                Interpolation mode — "linear" (default), "nearest",
///                          "bspline", "lanczos".
///     default_pixel_value: Fill value for voxels outside the field of view
///                          (default 0.0).
///
/// Returns:
///     Rotated PyImage with identical shape, spacing, origin, and direction.
///
/// Raises:
///     ValueError: if `mode` is not one of the recognised modes.
#[pyfunction]
#[pyo3(signature = (image, angle_x=0.0_f64, angle_y=0.0_f64, angle_z=0.0_f64, mode="linear", default_pixel_value=0.0_f64))]
#[allow(clippy::too_many_arguments)]
pub fn rotate_image(
    py: Python<'_>,
    image: &PyImage,
    angle_x: f64,
    angle_y: f64,
    angle_z: f64,
    mode: &str,
    default_pixel_value: f64,
) -> RitkResult<PyImage> {
    let mode = mode.to_string();
    let inner = std::sync::Arc::clone(&image.inner);
    py.allow_threads(move || -> Result<_, String> {
        let shape = inner.shape();
        let sp = *inner.spacing();
        let orig = *inner.origin();
        let dir = *inner.direction();
        let device: <Backend as BurnBackend>::Device = Default::default();

        // Centre of rotation in physical coordinates:
        //   c_d = origin_d + spacing_d * (shape_d - 1) / 2
        // Note: stored in ZYX order (shape[0]=Z, shape[1]=Y, shape[2]=X)
        let centre: Vec<f32> = (0..3)
            .map(|d| orig[d] as f32 + sp[d] as f32 * (shape[d] as f32 - 1.0) / 2.0)
            .collect();
        let centre_t = Tensor::<Backend, 1>::from_data(
            TensorData::new(centre, Shape::new([3])),
            &device,
        );
        // Zero translation (pure rotation about centre).
        let translation = Tensor::<Backend, 1>::zeros([3], &device);
        // Build the rotation to match SimpleITK's `Euler3DTransform`
        // (`ComputeZYX = false`, the default), whose matrix in physical [x, y, z]
        // space is `M = R_z(angle_z) · R_x(angle_x) · R_y(angle_y)`. ritk's
        // resample operates in tensor-axis [z, y, x] space — the reverse of the
        // physical axes — so the applied matrix is `P · M · Pᵀ` with `P` the
        // axis-reversal permutation, i.e. `A[i][j] = M[2-i][2-j]`. Using the
        // explicit matrix (instead of `RigidTransform`'s `R_z·R_y·R_x` Euler
        // builder) reproduces SimpleITK's composition for simultaneous
        // multi-axis rotations, not just one axis at a time.
        let m = euler_zxy_matrix(angle_x, angle_y, angle_z);
        let a: Vec<f32> = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| m[2 - i][2 - j] as f32)
            .collect();
        let matrix =
            Tensor::<Backend, 2>::from_data(TensorData::new(a, Shape::new([3, 3])), &device);
        let transform = AffineTransform::<Backend, 3>::new(matrix, translation, centre_t);

        match mode.as_str() {
            "nearest" => Ok(ResampleImageFilter::new(
                shape, orig, sp, dir, transform.clone(), NearestNeighborInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "linear" => Ok(ResampleImageFilter::new(
                shape, orig, sp, dir, transform.clone(), LinearInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "bspline" => Ok(ResampleImageFilter::new(
                shape, orig, sp, dir, transform.clone(), BSplineInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "lanczos" => Ok(ResampleImageFilter::new(
                shape, orig, sp, dir, transform, Lanczos5Interpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            other => Err(format!(
                "rotate_image: unknown interpolation mode '{}'. Use: nearest, linear, bspline, lanczos",
                other
            )),
        }
    })
    .map_err(RitkPyError::value)
    .map(into_py_image)
}

// ── GAP-SCI-02: shift_image ──────────────────────────────────────────────────

/// Translate (shift) a 3-D image by a physical offset.
///
/// Implements `scipy.ndimage.shift` / SimpleITK TranslationTransform parity.
/// The output grid is identical to the input grid (same shape, spacing, origin,
/// and direction).  Out-of-bounds voxels receive `default_pixel_value`.
///
/// Args:
///     image:               Input PyImage.
///     shift_z:             Translation along Z axis in physical units (mm, default 0.0).
///     shift_y:             Translation along Y axis in physical units (mm, default 0.0).
///     shift_x:             Translation along X axis in physical units (mm, default 0.0).
///     mode:                Interpolation mode — "linear" (default), "nearest",
///                          "bspline", "lanczos".
///     default_pixel_value: Fill value for voxels outside the field of view
///                          (default 0.0).
///
/// Returns:
///     Shifted PyImage with identical shape, spacing, origin, and direction.
///
/// Raises:
///     ValueError: if `mode` is not one of the recognised modes.
#[pyfunction]
#[pyo3(signature = (image, shift_z=0.0_f64, shift_y=0.0_f64, shift_x=0.0_f64, mode="linear", default_pixel_value=0.0_f64))]
#[allow(clippy::too_many_arguments)]
pub fn shift_image(
    py: Python<'_>,
    image: &PyImage,
    shift_z: f64,
    shift_y: f64,
    shift_x: f64,
    mode: &str,
    default_pixel_value: f64,
) -> RitkResult<PyImage> {
    let mode = mode.to_string();
    let inner = std::sync::Arc::clone(&image.inner);
    py.allow_threads(move || -> Result<_, String> {
        let shape = inner.shape();
        let sp = *inner.spacing();
        let orig = *inner.origin();
        let dir = *inner.direction();
        let device: <Backend as BurnBackend>::Device = Default::default();

        // TranslationTransform shifts the OUTPUT→INPUT mapping, so we negate:
        // to shift the image by (dz, dy, dx), the transform must map
        // out_point → out_point - [dz, dy, dx] in physical space.
        let translation = Tensor::<Backend, 1>::from_data(
            TensorData::new(
                vec![-shift_z as f32, -shift_y as f32, -shift_x as f32],
                Shape::new([3]),
            ),
            &device,
        );
        let transform = TranslationTransform::<Backend, 3>::new(translation);

        match mode.as_str() {
            "nearest" => Ok(ResampleImageFilter::new(
                shape,
                orig,
                sp,
                dir,
                transform.clone(),
                NearestNeighborInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "linear" => Ok(ResampleImageFilter::new(
                shape,
                orig,
                sp,
                dir,
                transform.clone(),
                LinearInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "bspline" => Ok(ResampleImageFilter::new(
                shape,
                orig,
                sp,
                dir,
                transform.clone(),
                BSplineInterpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            "lanczos" => Ok(ResampleImageFilter::new(
                shape,
                orig,
                sp,
                dir,
                transform,
                Lanczos5Interpolator::new(),
            )
            .with_default_pixel_value(default_pixel_value)
            .apply(inner.as_ref())),
            other => Err(format!(
                "shift_image: unknown mode '{}'. Use: nearest, linear, bspline, lanczos",
                other
            )),
        }
    })
    .map_err(RitkPyError::value)
    .map(into_py_image)
}

// ── GAP-SCI-15: zoom_image ───────────────────────────────────────────────────

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

// ── Euler rotation matrix (SimpleITK Euler3DTransform, ComputeZYX = false) ──────

/// 3×3 product of two row-major matrices.
fn matmul3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for (i, ci) in c.iter_mut().enumerate() {
        for (j, cij) in ci.iter_mut().enumerate() {
            *cij = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// Rotation matrix `R_z(angle_z) · R_x(angle_x) · R_y(angle_y)` in physical
/// [x, y, z] coordinates — the composition SimpleITK's `Euler3DTransform` uses
/// with its default `ComputeZYX = false`.
fn euler_zxy_matrix(angle_x: f64, angle_y: f64, angle_z: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = angle_x.sin_cos();
    let (sb, cb) = angle_y.sin_cos();
    let (sg, cg) = angle_z.sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]];
    let ry = [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]];
    let rz = [[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]];
    matmul3(rz, matmul3(rx, ry))
}

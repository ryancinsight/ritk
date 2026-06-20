use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use pyo3::prelude::*;
use ritk_filter::ResampleImageFilter;
use ritk_interpolation::LinearInterpolator;
use ritk_interpolation::{BSplineInterpolator, Lanczos5Interpolator, NearestNeighborInterpolator};
use ritk_transform::affine::affine::AffineTransform;
use ritk_transform::affine::translation::TranslationTransform;

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

/// Sample an affine transform onto a reference grid as a dense displacement
/// field, matching `SimpleITK.TransformToDisplacementField` for an
/// `AffineTransform`.
///
/// Returns the field `D(p) = T(p) − p`, `T(p) = M·(p − c) + c + t`, as three
/// scalar component images `(disp_z, disp_y, disp_x)` on the reference grid —
/// the order `filter.warp` consumes. `matrix` is row-major 3×3, `translation`
/// and `center` are length-3, all in the physical `(x, y, z)` frame (SimpleITK's
/// `AffineTransform` convention).
///
/// The reference should carry its loaded geometry (e.g. via `ritk.io.read_image`)
/// so its Direction matches the file; physical points are taken from the
/// canonical index→world transform.
#[pyfunction]
#[pyo3(signature = (reference, matrix, translation, center=[0.0, 0.0, 0.0]))]
pub fn transform_to_displacement_field(
    py: Python<'_>,
    reference: &PyImage,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
) -> RitkResult<(PyImage, PyImage, PyImage)> {
    let arc = std::sync::Arc::clone(&reference.inner);
    let (dz, dy, dx) = py.allow_threads(|| {
        ritk_filter::transform_to_displacement_field(arc.as_ref(), matrix, translation, center)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    Ok((into_py_image(dz), into_py_image(dy), into_py_image(dx)))
}

/// Apply an affine transform to an image's geometry (origin + direction),
/// leaving the voxel data and spacing unchanged. Matches
/// `SimpleITK.TransformGeometry` for an `AffineTransform`.
///
/// ITK applies the inverse linear map: `origin' = A⁻¹·(origin − c − t) + c`,
/// `direction' = A⁻¹·direction`. `matrix` is row-major 3×3, `translation` and
/// `center` are length-3, all in the physical `(x, y, z)` frame.
///
/// The image should carry its loaded geometry (e.g. via `ritk.io.read_image`) so
/// its Direction matches the file.
#[pyfunction]
#[pyo3(signature = (image, matrix, translation, center=[0.0, 0.0, 0.0]))]
pub fn transform_geometry(
    py: Python<'_>,
    image: &PyImage,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ritk_filter::transform_geometry(arc.as_ref(), matrix, translation, center)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

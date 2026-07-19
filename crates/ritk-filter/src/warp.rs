//! Warp filter: resample a moving image through a dense displacement field.
//!
//! # Mathematical Specification
//!
//! Matches `itk::WarpImageFilter`. For every voxel of the output grid (the
//! displacement-field grid) at physical point `p`, the output samples the moving
//! image at the displaced physical point:
//!
//! ```text
//! out(p) = moving( p + D(p) )
//! ```
//!
//! Coordinate mapping is delegated entirely to the image's canonical
//! [`Image::index_to_world_native`] / [`Image::world_to_index_native`] â€” the same
//! transforms `ResampleImageFilter` uses and which are verified float-exact to
//! SimpleITK on loaded anisotropic data. Operating on the tensor directly (rather
//! than a hand-rolled flat-index loop) makes the filter agnostic to image
//! construction and honours spacing, origin, and direction (including the
//! axis-reversal Direction that `ritk.io` assigns NRRD-loaded images). Trilinear
//! interpolation samples the moving image; samples whose continuous index leaves
//! the moving buffer (`c_a âˆ‰ [âˆ’0.5, N_a âˆ’ 0.5)` on any axis) take the edge-padding
//! value 0 (ITK's `IsInsideBuffer`).
//!
//! Displacement components are supplied as three scalar images `(D_z, D_y, D_x)`
//! on the field grid, assembled into the displacement tensor in the
//! **innermost-first** `(x, y, z)` column order that `index_to_world_tensor`
//! produces so the displacement adds to the world points column-wise.

use anyhow::{anyhow, Result};
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_image::Image;
use ritk_spatial::Point;
use ritk_tensor_ops::extract_vec_infallible;

use crate::resample::native::{fixed_world_points, sample_moving_at_world};

/// Warp a moving image through a dense displacement field.
///
/// `moving` is the image to sample; `disp_z`, `disp_y`, `disp_x` are the
/// physical-space displacement components defined on the output (field) grid.
/// All three field components must share the field's shape; the output adopts the
/// field's geometry.
pub fn warp_image<B: Backend>(
    moving: &Image<f32, B, 3>,
    disp_z: &Image<f32, B, 3>,
    disp_y: &Image<f32, B, 3>,
    disp_x: &Image<f32, B, 3>,
) -> Result<Image<f32, B, 3>> {
    let field_dims = disp_z.shape();
    if disp_y.shape() != field_dims || disp_x.shape() != field_dims {
        return Err(anyhow!(
            "warp: displacement components must share the field shape {field_dims:?}"
        ));
    }
    let n: usize = field_dims.iter().product();
    let (dz, _) = extract_vec_infallible(disp_z);
    let (dy, _) = extract_vec_infallible(disp_y);
    let (dx, _) = extract_vec_infallible(disp_x);
    let (moving_values, moving_shape) = extract_vec_infallible(moving);
    anyhow::ensure!(
        moving_shape.into_iter().all(|extent| extent > 0),
        "warp: moving image dimensions must all be nonzero, got {moving_shape:?}"
    );
    let plane = field_dims[1] * field_dims[2];
    let output = (0..n)
        .map(|linear| {
            let field_index = Point::new([
                (linear / plane) as f64,
                ((linear % plane) / field_dims[2]) as f64,
                (linear % field_dims[2]) as f64,
            ]);
            let mut world = disp_z.transform_continuous_index_to_physical_point(&field_index);
            world[0] += dz[linear] as f64;
            world[1] += dy[linear] as f64;
            world[2] += dx[linear] as f64;
            let moving_index = moving.transform_physical_point_to_continuous_index(&world);
            trilinear_sample(&moving_values, moving_shape, moving_index)
        })
        .collect::<Vec<_>>();

    Image::new(
        Tensor::<f32, B>::from_slice(field_dims, &output),
        *disp_z.origin(),
        *disp_z.spacing(),
        *disp_z.direction(),
    )
}

fn trilinear_sample(values: &[f32], shape: [usize; 3], index: Point<3>) -> f32 {
    if (0..3).any(|axis| index[axis] < -0.5 || index[axis] >= shape[axis] as f64 - 0.5) {
        return 0.0;
    }

    let mut lower = [0usize; 3];
    let mut upper = [0usize; 3];
    let mut fraction = [0.0f32; 3];
    for axis in 0..3 {
        let floor = index[axis].floor();
        lower[axis] = floor.clamp(0.0, (shape[axis] - 1) as f64) as usize;
        upper[axis] = (lower[axis] + 1).min(shape[axis] - 1);
        fraction[axis] = (index[axis] - floor) as f32;
    }

    let offset = |z: usize, y: usize, x: usize| (z * shape[1] + y) * shape[2] + x;
    let mut result = 0.0;
    for z_bit in 0..=1 {
        for y_bit in 0..=1 {
            for x_bit in 0..=1 {
                let z = [lower[0], upper[0]][z_bit];
                let y = [lower[1], upper[1]][y_bit];
                let x = [lower[2], upper[2]][x_bit];
                let weight = [1.0 - fraction[0], fraction[0]][z_bit]
                    * [1.0 - fraction[1], fraction[1]][y_bit]
                    * [1.0 - fraction[2], fraction[2]][x_bit];
                result += values[offset(z, y, x)] * weight;
            }
        }
    }
    result
}

/// Warp a native image through a dense physical displacement field.
///
/// For each field-grid world point `p`, the output samples `moving` at
/// `p + (D_z(p), D_y(p), D_x(p))`. The displacement components and output use
/// axis-major physical order `[z, y, x]`; samples outside the moving image's
/// half-voxel buffer are zero-filled by the shared native resampler.
///
/// # Errors
///
/// Returns an error when field component shapes or spatial metadata differ, or
/// when the shared native sampler rejects the moving image or point batch.
pub fn warp_image_native<B>(
    moving: &ritk_image::Image<f32, B, 3>,
    disp_z: &ritk_image::Image<f32, B, 3>,
    disp_y: &ritk_image::Image<f32, B, 3>,
    disp_x: &ritk_image::Image<f32, B, 3>,
) -> Result<ritk_image::Image<f32, B, 3>>
where
    B: coeus_core::Backend + coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let shape = disp_z.shape();
    for (axis, field) in [("y", disp_y), ("x", disp_x)] {
        anyhow::ensure!(
            field.shape() == shape,
            "warp: displacement {axis} component shape {:?} differs from z component shape {shape:?}",
            field.shape()
        );
        anyhow::ensure!(
            field.origin() == disp_z.origin()
                && field.spacing() == disp_z.spacing()
                && field.direction() == disp_z.direction(),
            "warp: displacement {axis} component geometry differs from z component geometry"
        );
    }

    let dz = disp_z.data_cow();
    let dy = disp_y.data_cow();
    let dx = disp_x.data_cow();
    let mut moving_world = fixed_world_points(disp_z);
    for (point, ((&z, &y), &x)) in moving_world
        .chunks_exact_mut(3)
        .zip(dz.iter().zip(dy.iter()).zip(dx.iter()))
    {
        point[0] += z;
        point[1] += y;
        point[2] += x;
    }

    ritk_image::Image::from_flat(
        sample_moving_at_world(moving, &moving_world)?,
        shape,
        *disp_z.origin(),
        *disp_z.spacing(),
        *disp_z.direction(),
    )
}

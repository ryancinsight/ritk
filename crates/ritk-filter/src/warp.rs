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
//! [`ritk_image::Image::index_to_world_native`] /
//! [`ritk_image::Image::world_to_index_native`] — the same transforms
//! `ResampleImageFilter` uses and which are verified float-exact to SimpleITK on
//! loaded anisotropic data. Operating on the tensor directly (rather than a
//! hand-rolled flat-index loop) makes the filter agnostic to image construction
//! and honours spacing, origin, and direction (including the axis-reversal
//! Direction that `ritk.io` assigns NRRD-loaded images). Trilinear interpolation
//! samples the moving image; samples whose continuous index leaves the moving
//! buffer (`c_a ∉ [−0.5, N_a − 0.5)` on any axis) take the edge-padding value 0
//! (ITK's `IsInsideBuffer`).
//!
//! Displacement components are supplied as three scalar images `(D_z, D_y, D_x)`
//! on the field grid, assembled into the displacement tensor in the
//! **innermost-first** `(x, y, z)` column order that `index_to_world_tensor`
//! produces so the displacement adds to the world points column-wise.

use crate::resample::native::{fixed_world_points, sample_moving_at_world};
use anyhow::Result;

/// Warp a native image through a dense physical displacement field.
///
/// For each field-grid world point `p`, the output samples `moving` at
/// `p + (D_x(p), D_y(p), D_z(p))`. Arguments retain the public
/// `(disp_z, disp_y, disp_x)` image order, while physical points use world
/// `[x, y, z]` column order. Samples outside the moving image's half-voxel
/// buffer are zero-filled by the shared native resampler.
///
/// # Errors
///
/// Returns an error when field component shapes or spatial metadata differ, or
/// when the shared native sampler rejects the moving image or point batch.
pub fn warp_image<B>(
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
        point[0] += x;
        point[1] += y;
        point[2] += z;
    }

    ritk_image::Image::from_flat(
        sample_moving_at_world(moving, &moving_world)?,
        shape,
        *disp_z.origin(),
        *disp_z.spacing(),
        *disp_z.direction(),
    )
}

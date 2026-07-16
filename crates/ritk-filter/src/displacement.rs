//! Transform-to-displacement-field: sample an affine transform onto a grid.
//!
//! # Mathematical Specification
//!
//! Matches `itk::TransformToDisplacementFieldFilter` for an affine transform.
//! For every voxel of a reference grid at physical point `p`, the displacement
//! is the transform applied minus the point itself:
//!
//! ```text
//! D(p) = T(p) − p,    T(p) = M·(p − c) + c + t
//! ```
//!
//! where `M` is the 3×3 matrix, `t` the translation, and `c` the centre, all in
//! the physical `(x, y, z)` frame (SimpleITK's `AffineTransform` convention).
//! Physical points come from the image's canonical
//! [`Image::index_to_world_native_on`], whose innermost-first columns are exactly
//! `(x, y, z)`, so the result is float-exact to `sitk.TransformToDisplacementField`.
//!
//! The field is returned as three scalar component images `(D_z, D_y, D_x)` on
//! the reference grid — the same `(disp_z, disp_y, disp_x)` order
//! [`crate::warp::warp_image`] consumes, so `warp(moving, …) ≡ resample(moving,
//! transform)`.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use ritk_image::native::Image;
use ritk_image::generate_grid;

/// Sample an affine transform `T(p) = M·(p − c) + c + t` onto the reference
/// grid, returning the dense displacement field `D(p) = T(p) − p` as
/// `(disp_z, disp_y, disp_x)` scalar component images.
///
/// `matrix`, `translation`, and `center` are in the physical `(x, y, z)` frame
/// (row-major matrix), matching SimpleITK's `AffineTransform`.
pub fn transform_to_displacement_field<B>(
    reference: &Image<f32, B, 3>,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
    backend: &B,
) -> Result<(Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>)>
where
    B: ComputeBackend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let dims = reference.shape();
    let n: usize = dims.iter().product();

    // Reference-grid physical points. `index_to_world_native_on` returns world
    // columns in [x, y, z] order (output column c = world component c, paired
    // with origin[c] and direction[(c, axis)]), matching SimpleITK's frame.
    let indices = generate_grid::<f32, B, 3>(dims, backend);
    let world = reference.index_to_world_native_on(&indices, backend); // [N, 3], cols [x, y, z]

    // T(p) = M·(p − c) + c + t, computed row-wise as
    // (p − c) · Mᵀ + (c + t). Build Mᵀ so `world_centered.matmul(mt)` applies M
    // to each row in [x, y, z] order.
    let mt_data: Vec<f32> = (0..3)
        .flat_map(|i| (0..3).map(move |j| matrix[j][i] as f32))
        .collect();
    let mt = Tensor::<f32, B>::from_slice_on([3, 3], &mt_data, backend);
    let c_row = Tensor::<f32, B>::from_slice_on(
        [1, 3],
        &center.map(|v| v as f32),
        backend,
    );
    let ct_row = Tensor::<f32, B>::from_slice_on(
        [1, 3],
        &[
            (center[0] + translation[0]) as f32,
            (center[1] + translation[1]) as f32,
            (center[2] + translation[2]) as f32,
        ],
        backend,
    );

    let centered_xyz = coeus_ops::sub(&world, &c_row, backend);
    let transformed_xyz = coeus_ops::add(&coeus_ops::matmul(&centered_xyz, &mt, backend), &ct_row, backend); // T(p) in [x, y, z]
    let disp_xyz = coeus_ops::sub(&transformed_xyz, &world, backend); // D(p) = T(p) − p, columns [Dx, Dy, Dz]

    let extract_component = |col: usize| -> Result<Image<f32, B, 3>> {
        let col_tensor = disp_xyz.slice(&[(0, n), (col, col + 1)]).reshape(dims);
        Image::new(
            col_tensor,
            *reference.origin(),
            *reference.spacing(),
            *reference.direction(),
        )
    };

    Ok((extract_component(2)?, extract_component(1)?, extract_component(0)?))
}

#[cfg(test)]
#[path = "tests_displacement.rs"]
mod tests_displacement;

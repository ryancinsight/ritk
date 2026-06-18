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
//! [`Image::index_to_world_tensor`], whose innermost-first columns are exactly
//! `(x, y, z)`, so the result is float-exact to `sitk.TransformToDisplacementField`.
//!
//! The field is returned as three scalar component images `(D_z, D_y, D_x)` on
//! the reference grid — the same `(disp_z, disp_y, disp_x)` order
//! [`crate::warp::warp_image`] consumes, so `warp(moving, …) ≡ resample(moving,
//! transform)`.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::{generate_grid, Image};

/// Sample an affine transform `T(p) = M·(p − c) + c + t` onto the reference
/// grid, returning the dense displacement field `D(p) = T(p) − p` as
/// `(disp_z, disp_y, disp_x)` scalar component images.
///
/// `matrix`, `translation`, and `center` are in the physical `(x, y, z)` frame
/// (row-major matrix), matching SimpleITK's `AffineTransform`.
pub fn transform_to_displacement_field<B: Backend>(
    reference: &Image<B, 3>,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
) -> Result<(Image<B, 3>, Image<B, 3>, Image<B, 3>)> {
    let dims = reference.shape();
    let n: usize = dims.iter().product();
    let device = reference.data().device();

    // Reference-grid physical points (innermost-first columns [x, y, z]).
    let indices = generate_grid::<B, 3>(dims, &device);
    let world = reference.index_to_world_tensor(indices); // [N, 3]

    // T(p) = M·(p − c) + c + t, computed row-wise as
    // (p − c) · Mᵀ + (c + t). Build Mᵀ so `world_centered.matmul(mt)` applies M
    // to each row.
    let mt_data: Vec<f32> = (0..3)
        .flat_map(|i| (0..3).map(move |j| matrix[j][i] as f32))
        .collect();
    let mt = Tensor::<B, 2>::from_data(TensorData::new(mt_data, Shape::new([3, 3])), &device);
    let c_row = Tensor::<B, 2>::from_data(
        TensorData::new(center.map(|v| v as f32).to_vec(), Shape::new([1, 3])),
        &device,
    );
    let ct_row = Tensor::<B, 2>::from_data(
        TensorData::new(
            [
                (center[0] + translation[0]) as f32,
                (center[1] + translation[1]) as f32,
                (center[2] + translation[2]) as f32,
            ]
            .to_vec(),
            Shape::new([1, 3]),
        ),
        &device,
    );

    let centered = world.clone() - c_row;
    let transformed = centered.matmul(mt) + ct_row; // T(p), [N, 3]
    let disp = transformed - world; // D(p) = T(p) − p, columns [Dx, Dy, Dz]

    // Split columns into (D_z, D_y, D_x): column 0 = Dx, 1 = Dy, 2 = Dz.
    let component = |col: usize| -> Image<B, 3> {
        let c = disp
            .clone()
            .slice([0..n, col..col + 1])
            .reshape(Shape::new(dims));
        Image::new(
            c,
            *reference.origin(),
            *reference.spacing(),
            *reference.direction(),
        )
    };
    Ok((component(2), component(1), component(0)))
}

#[cfg(test)]
#[path = "tests_displacement.rs"]
mod tests_displacement;

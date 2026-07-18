//! Transform-to-displacement-field: sample an affine transform onto a grid.
//!
//! # Mathematical Specification
//!
//! Matches `itk::TransformToDisplacementFieldFilter` for an affine transform.
//! For every voxel of a reference grid at physical point `p`, the displacement
//! is the transform applied minus the point itself:
//!
//! ```text
//! D(p) = T(p) âˆ’ p,    T(p) = MÂ·(p âˆ’ c) + c + t
//! ```
//!
//! where `M` is the 3Ã—3 matrix, `t` the translation, and `c` the centre, all in
//! the physical `(x, y, z)` frame (SimpleITK's `AffineTransform` convention).
//! Physical points come from the image's canonical
//! [`Image::index_to_world_tensor`], whose innermost-first columns are exactly
//! `(x, y, z)`, so the result is float-exact to `sitk.TransformToDisplacementField`.
//!
//! The field is returned as three scalar component images `(D_z, D_y, D_x)` on
//! the reference grid â€” the same `(disp_z, disp_y, disp_x)` order
//! [`crate::warp::warp_image`] consumes, so `warp(moving, â€¦) â‰¡ resample(moving,
//! transform)`.

use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_image::Image;
use ritk_spatial::Point;

type DisplacementImages<B> = (Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>);

/// Sample an affine transform `T(p) = MÂ·(p âˆ’ c) + c + t` onto the reference
/// grid, returning the dense displacement field `D(p) = T(p) âˆ’ p` as
/// `(disp_z, disp_y, disp_x)` scalar component images.
///
/// `matrix`, `translation`, and `center` are in the physical `(x, y, z)` frame
/// (row-major matrix), matching SimpleITK's `AffineTransform`.
pub fn transform_to_displacement_field<B: Backend>(
    reference: &Image<f32, B, 3>,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
) -> Result<DisplacementImages<B>> {
    let dims = reference.shape();
    let n: usize = dims.iter().product();
    let mut components = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let plane = dims[1] * dims[2];
    for linear in 0..n {
        let index = Point::new([
            (linear / plane) as f64,
            ((linear % plane) / dims[2]) as f64,
            (linear % dims[2]) as f64,
        ]);
        let point = reference.transform_continuous_index_to_physical_point(&index);
        for (row, component) in components.iter_mut().enumerate() {
            let transformed = (0..3)
                .map(|column| matrix[row][column] * (point[column] - center[column]))
                .sum::<f64>()
                + center[row]
                + translation[row];
            component.push((transformed - point[row]) as f32);
        }
    }

    let extract_component = |values: Vec<f32>| -> Image<f32, B, 3> {
        Image::new(
            Tensor::<f32, B>::from_slice(dims, &values),
            *reference.origin(),
            *reference.spacing(),
            *reference.direction(),
        )
    };

    let [dx, dy, dz] = components.map(extract_component);

    Ok((dz, dy, dx))
}

#[cfg(test)]
#[path = "tests_displacement.rs"]
mod tests_displacement;

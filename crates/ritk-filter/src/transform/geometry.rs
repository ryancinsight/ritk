//! Affine geometry transform (`itk::TransformGeometryImageFilter` /
//! `sitk.TransformGeometry`).
//!
//! # Mathematical Specification
//!
//! Moves an image's physical coordinate system by an affine transform `T`
//! without resampling — the voxel data and spacing are unchanged; only the
//! origin and direction matrix are updated. ITK applies the **inverse** linear
//! map so that a subsequent resample reproduces `T`:
//!
//! ```text
//! T(p) = A·(p − c) + c + t          (affine, world (x, y, z) frame)
//! origin' = A⁻¹·(origin − c − t) + c
//! D'.col  = A⁻¹·D.col               (each direction column, a world vector)
//! ```
//!
//! `matrix` (`A`), `translation` (`t`), and `center` (`c`) are in the physical
//! `(x, y, z)` frame, matching SimpleITK's `AffineTransform`. The core direction
//! matrix already stores its columns as world vectors, so `A⁻¹` left-multiplies
//! it directly. Float-exact to `sitk.TransformGeometry`.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use nalgebra::{Matrix3, Vector3};
use ritk_image::Image;
use ritk_spatial::{Direction, Point};

/// Apply an affine transform to an image's geometry (origin + direction),
/// leaving the voxel data and spacing unchanged.
///
/// `matrix` is row-major in the physical `(x, y, z)` frame. Returns `Err` if the
/// matrix is singular (no inverse).
pub fn transform_geometry<B: Backend>(
    image: &Image<B, 3>,
    matrix: [[f64; 3]; 3],
    translation: [f64; 3],
    center: [f64; 3],
) -> Result<Image<B, 3>> {
    let a = Matrix3::new(
        matrix[0][0], matrix[0][1], matrix[0][2], //
        matrix[1][0], matrix[1][1], matrix[1][2], //
        matrix[2][0], matrix[2][1], matrix[2][2],
    );
    let Some(a_inv) = a.try_inverse() else {
        bail!("transform_geometry: matrix is singular (no inverse)");
    };
    let t = Vector3::new(translation[0], translation[1], translation[2]);
    let c = Vector3::new(center[0], center[1], center[2]);

    let o = image.origin();
    let o_vec = Vector3::new(o[0], o[1], o[2]);
    let new_o = a_inv * (o_vec - c - t) + c;

    // Each direction column is a world vector; A⁻¹ left-multiplies the matrix.
    let new_dir = a_inv * image.direction().0;

    Ok(Image::new(
        image.data().clone(),
        Point::new([new_o[0], new_o[1], new_o[2]]),
        *image.spacing(),
        Direction(new_dir),
    ))
}

#[cfg(test)]
#[path = "tests_geometry.rs"]
mod tests_geometry;

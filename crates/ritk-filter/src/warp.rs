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
//! [`Image::index_to_world_tensor`] / [`Image::world_to_index_tensor`] — the same
//! transforms `ResampleImageFilter` uses and which are verified float-exact to
//! SimpleITK on loaded anisotropic data. Operating on the tensor directly (rather
//! than a hand-rolled flat-index loop) makes the filter agnostic to image
//! construction and honours spacing, origin, and direction (including the
//! axis-reversal Direction that `ritk.io` assigns NRRD-loaded images). Trilinear
//! interpolation samples the moving image; samples whose continuous index leaves
//! the moving buffer (`c_a ∉ [−0.5, N_a − 0.5)` on any axis) take the edge-padding
//! value 0 (ITK's `IsInsideBuffer`).
//!
//! Displacement components are supplied as three scalar images `(D_z, D_y, D_x)`
//! on the field grid, assembled into the displacement tensor in the
//! **innermost-first** `(x, y, z)` column order that `index_to_world_tensor`
//! produces so the displacement adds to the world points column-wise.

use anyhow::{anyhow, Result};
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::{generate_grid_burn as generate_grid, Image};
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_tensor_ops::extract_vec_infallible;

/// Replace interpolated values with 0 wherever the continuous moving index falls
/// outside the buffer, reproducing ITK's `IsInsideBuffer` (inside iff every axis
/// lies in `[-0.5, N - 0.5)`). Mirrors `ResampleImageFilter` so warp and resample
/// share identical edge semantics. `indices` columns are innermost-first
/// (`column c` ↔ spatial axis `D-1-c`).
fn apply_inside_buffer_mask<B: Backend>(
    values: Tensor<B, 1>,
    indices: Tensor<B, 2>,
    shape: [usize; 3],
    device: &<B as Backend>::Device,
) -> Tensor<B, 1> {
    let n = indices.dims()[0];
    let mut mask = Tensor::<B, 1>::ones([n], device);
    for c in 0..3 {
        let axis = 3 - 1 - c;
        let upper = shape[axis] as f64 - 0.5;
        let col = indices.clone().slice([0..n, c..c + 1]).flatten::<1>(0, 1);
        let ge_low = col.clone().greater_equal_elem(-0.5).float();
        let lt_high = col.lower_elem(upper).float();
        mask = mask * ge_low * lt_high;
    }
    values * mask
}

/// Warp a moving image through a dense displacement field.
///
/// `moving` is the image to sample; `disp_z`, `disp_y`, `disp_x` are the
/// physical-space displacement components defined on the output (field) grid.
/// All three field components must share the field's shape; the output adopts the
/// field's geometry.
pub fn warp_image<B: Backend>(
    moving: &Image<B, 3>,
    disp_z: &Image<B, 3>,
    disp_y: &Image<B, 3>,
    disp_x: &Image<B, 3>,
) -> Result<Image<B, 3>> {
    let field_dims = disp_z.shape();
    if disp_y.shape() != field_dims || disp_x.shape() != field_dims {
        return Err(anyhow!(
            "warp: displacement components must share the field shape {field_dims:?}"
        ));
    }
    let n: usize = field_dims.iter().product();
    let device = moving.data().device();

    // Output-grid indices (innermost-first columns, row-major batch order), then
    // their physical points on the field grid via the canonical transform.
    let indices = generate_grid::<B, 3>(field_dims, &device);
    let world = disp_z.index_to_world_tensor(indices);

    // Displacement tensor [N, 3], innermost-first columns [dx, dy, dz] to match
    // the column order of `index_to_world_tensor`, in the same row-major batch
    // order as the grid.
    let (dz, _) = extract_vec_infallible(disp_z);
    let (dy, _) = extract_vec_infallible(disp_y);
    let (dx, _) = extract_vec_infallible(disp_x);
    let mut disp_flat = vec![0.0f32; n * 3];
    let dx_ref = &dx;
    let dy_ref = &dy;
    let dz_ref = &dz;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut disp_flat,
        3000,
        |chunk_idx, slice| {
            let start_triplet = chunk_idx * 1000;
            for (offset, triplet) in slice.chunks_exact_mut(3).enumerate() {
                let i = start_triplet + offset;
                triplet[0] = dx_ref[i];
                triplet[1] = dy_ref[i];
                triplet[2] = dz_ref[i];
            }
        },
    );
    let disp = Tensor::<B, 2>::from_data(TensorData::new(disp_flat, Shape::new([n, 3])), &device);

    // Displaced world points → moving continuous indices (innermost-first).
    let mov_idx = moving.world_to_index_tensor(world + disp);

    // Trilinear sample + ITK IsInsideBuffer gate (out-of-buffer → 0).
    let values = LinearInterpolator::new().interpolate(moving.data(), mov_idx.clone());
    let masked = apply_inside_buffer_mask::<B>(values, mov_idx, moving.shape(), &device);

    let out = masked.reshape(Shape::new(field_dims));
    Ok(Image::new(
        out,
        *disp_z.origin(),
        *disp_z.spacing(),
        *disp_z.direction(),
    ))
}

#[cfg(test)]
#[path = "tests_warp.rs"]
mod tests_warp;

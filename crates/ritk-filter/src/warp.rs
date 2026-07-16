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
//! [`Image::index_to_world_native_on`] / [`Image::world_to_index_native_on`] — the same
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
//! **innermost-first** `(x, y, z)` column order that `index_to_world_native_on`
//! produces so the displacement adds to the world points column-wise.

use anyhow::{anyhow, Result};
use coeus_core::{Backend, ComputeBackend, CpuAddressableStorage};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use ritk_image::native::Image;
use ritk_image::{generate_grid, native};
use ritk_interpolation::{Interpolator, LinearInterpolator};

/// Replace interpolated values with 0 wherever the continuous moving index falls
/// outside the buffer, reproducing ITK's `IsInsideBuffer` (inside iff every axis
/// lies in `[-0.5, N - 0.5)`). Mirrors `ResampleImageFilter` so warp and resample
/// share identical edge semantics. `indices` columns are innermost-first
/// (`column c` ↔ spatial axis `D-1-c`).
fn apply_inside_buffer_mask<B: ComputeBackend + BackendOps<f32>>(
    values: Tensor<f32, B>,
    indices: &Tensor<f32, B>,
    shape: [usize; 3],
    backend: &B,
) -> Tensor<f32, B>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let n = indices.shape()[0];
    let mut mask = Tensor::<f32, B>::ones_on([n], backend);
    for c in 0..3 {
        let axis = 2 - c;
        let upper = shape[axis] as f32 - 0.5;
        let col = indices.slice(&[(0, n), (c, c + 1)]).reshape([n]);
        let lower_bound = Tensor::<f32, B>::full_on([n], -0.5, backend);
        let upper_bound = Tensor::<f32, B>::full_on([n], upper, backend);
        let ge_low = coeus_ops::ge(&col, &lower_bound, backend);
        let lt_high = coeus_ops::lt(&col, &upper_bound, backend);
        let partial = coeus_ops::mul(&ge_low, &lt_high, backend);
        mask = coeus_ops::mul(&mask, &partial, backend);
    }
    coeus_ops::mul(&values, &mask, backend)
}

/// Warp a moving image through a dense displacement field.
///
/// `moving` is the image to sample; `disp_z`, `disp_y`, `disp_x` are the
/// physical-space displacement components defined on the output (field) grid.
/// All three field components must share the field's shape; the output adopts the
/// field's geometry.
pub fn warp_image<B>(
    moving: &Image<f32, B, 3>,
    disp_z: &Image<f32, B, 3>,
    disp_y: &Image<f32, B, 3>,
    disp_x: &Image<f32, B, 3>,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: Backend + ComputeBackend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let field_dims = disp_z.shape();
    if disp_y.shape() != field_dims || disp_x.shape() != field_dims {
        return Err(anyhow!(
            "warp: displacement components must share the field shape {field_dims:?}"
        ));
    }
    let n: usize = field_dims.iter().product();

    // Output-grid indices (innermost-first columns, row-major batch order), then
    // their physical points on the field grid via the canonical transform.
    let indices = generate_grid::<f32, B, 3>(field_dims, backend);
    let world = disp_z.index_to_world_native_on(&indices, backend);

    // Displacement tensor [N, 3], innermost-first columns [dx, dy, dz] to match
    // the column order of `index_to_world_native_on`, in the same row-major batch
    // order as the grid.
    let dz = disp_z.data().as_slice().to_vec();
    let dy = disp_y.data().as_slice().to_vec();
    let dx = disp_x.data().as_slice().to_vec();
    let mut disp_flat = vec![0.0f32; n * 3];
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut disp_flat,
        3000,
        |chunk_idx, slice| {
            let start_triplet = chunk_idx * 1000;
            for (offset, triplet) in slice.chunks_exact_mut(3).enumerate() {
                let i = start_triplet + offset;
                triplet[0] = dx[i];
                triplet[1] = dy[i];
                triplet[2] = dz[i];
            }
        },
    );
    let disp = Tensor::<f32, B>::from_slice_on([n, 3], &disp_flat, backend);

    // Displaced world points → moving continuous indices (innermost-first).
    let mov_idx = moving.world_to_index_native_on(
        &coeus_ops::add(&world, &disp, backend),
        backend,
    );

    // Trilinear sample + ITK IsInsideBuffer gate (out-of-bounds → 0).
    let values = LinearInterpolator::new().interpolate(moving.data(), mov_idx.clone());
    let masked = apply_inside_buffer_mask(values, &mov_idx, moving.shape(), backend);

    let out = masked.reshape(field_dims);
    native::Image::from_flat_on(
        out.as_slice().to_vec(),
        field_dims,
        *disp_z.origin(),
        *disp_z.spacing(),
        *disp_z.direction(),
        backend,
    )
}

#[cfg(test)]
#[path = "tests_warp.rs"]
mod tests_warp;

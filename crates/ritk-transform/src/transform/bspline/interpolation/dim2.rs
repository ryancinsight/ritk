use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_wgpu_compat::apply_row_chunks;

/// 2D B-spline transform — chunked over rows for WGPU-friendly memory
/// pressure. Called from `super::sealed` via const-generic `match D = 2`.
///
/// The dimension `2` is named only here (the body references
/// `self.grid_size[0]`, `self.grid_size[1]`, stride `nx`, etc.). Per-D
/// bodies are `pub(super)` so they are visible to the dispatch site in
/// `mod.rs` but hidden from the rest of the crate.
#[inline]
pub(super) fn transform_2d<B: Backend, const D: usize>(
    t: &BSplineTransform<B, D>,
    points: Tensor<B, 2>,
) -> Tensor<B, 2> {
    apply_row_chunks(points, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk| {
        transform_2d_chunk(t, chunk)
    })
}

#[inline]
fn transform_2d_chunk<B: Backend, const D: usize>(
    t: &BSplineTransform<B, D>,
    points: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let device = points.device();
    let batch_size = points.shape().dims[0];

    // Convert physical points to grid indices
    let grid_coords = t.world_to_grid_tensor(points.clone()); // [Batch, 2]

    // Compute Mask for Out-of-Bounds
    let zero_tensor = Tensor::<B, 1>::zeros([2], &device).reshape([1, 2]);
    let size_tensor = Tensor::<B, 1>::from_floats(
        [t.grid_size[0] as f32 - 1.0, t.grid_size[1] as f32 - 1.0],
        &device,
    )
    .reshape([1, 2]);

    let mask_ge_zero = grid_coords.clone().greater_equal(zero_tensor);
    let mask_le_size = grid_coords.clone().lower_equal(size_tensor);
    let mask = mask_ge_zero.equal(mask_le_size);

    // Reduce mask along dim 1
    let mask_float = mask.float();
    let valid_mask = mask_float.sum_dim(1).equal_elem(2.0).float();

    // Interpolation Logic
    let grid_indices_float = grid_coords.clone().floor();
    let u_vec = grid_coords - grid_indices_float.clone();

    let base_index = grid_indices_float.int() - 1;

    let ux = u_vec
        .clone()
        .slice([0..batch_size, 0..1])
        .flatten::<1>(0, 1);
    let uy = u_vec
        .clone()
        .slice([0..batch_size, 1..2])
        .flatten::<1>(0, 1);

    let bx = BSplineTransform::<B, D>::compute_basis_tensor(ux);
    let by = BSplineTransform::<B, D>::compute_basis_tensor(uy);

    // Weights: [Batch, 4, 4] -> Flatten to [Batch, 16]
    let weights = bx.unsqueeze_dim::<3>(2) * by.unsqueeze_dim::<3>(1);
    let weights = weights.reshape([batch_size, 16, 1]);

    let nx = t.grid_size[0] as i32;
    let ny = t.grid_size[1] as i32;

    let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);

    let i_idx = range.clone().reshape([1, 4, 1]);
    let j_idx = range.clone().reshape([1, 1, 4]);

    let base_x = base_index
        .clone()
        .slice([0..batch_size, 0..1])
        .unsqueeze_dim::<3>(2);
    let base_y = base_index
        .clone()
        .slice([0..batch_size, 1..2])
        .unsqueeze_dim::<3>(2);

    let idx_x = base_x + i_idx;
    let idx_y = base_y + j_idx;

    let zeros = Tensor::<B, 3, burn::tensor::Int>::zeros([1, 4, 4], &device);

    let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 16]);
    let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 16]);

    let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
    let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);

    let flat_indices = idx_y_clamped * nx + idx_x_clamped;

    let gather_indices = flat_indices.reshape([batch_size * 16]);
    let coeffs = t.coefficients.val().clone().select(0, gather_indices); // [Batch*16, 2]
    let coeffs = coeffs.reshape([batch_size, 16, 2]);

    let displacement = (coeffs * weights).sum_dim(1).flatten::<2>(1, 2);

    let masked_displacement = displacement * valid_mask;

    points + masked_displacement
}

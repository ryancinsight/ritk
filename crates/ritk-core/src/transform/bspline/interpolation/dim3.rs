use crate::transform::bspline::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    pub(crate) fn transform_3d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = points.shape().dims[0];
        const CHUNK_SIZE: usize = 32768;

        if batch_size <= CHUNK_SIZE {
            self.transform_3d_chunk(points)
        } else {
            let num_chunks = (batch_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, batch_size);

                let chunk_points = points.clone().slice([start..end]);
                let chunk_result = self.transform_3d_chunk(chunk_points);
                chunks.push(chunk_result);
            }
            Tensor::cat(chunks, 0)
        }
    }

    pub(crate) fn transform_3d_chunk(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];

        // Convert physical points to grid indices
        let grid_coords = self.world_to_grid_tensor(points.clone()); // [Batch, 3]

        // Compute Mask for Out-of-Bounds
        let zero_tensor = Tensor::<B, 1>::zeros([3], &device).reshape([1, 3]);
        let size_tensor = Tensor::<B, 1>::from_floats(
            [
                self.grid_size[0] as f32 - 1.0,
                self.grid_size[1] as f32 - 1.0,
                self.grid_size[2] as f32 - 1.0,
            ],
            &device,
        )
        .reshape([1, 3]);

        let mask_ge_zero = grid_coords.clone().greater_equal(zero_tensor);
        let mask_le_size = grid_coords.clone().lower_equal(size_tensor);
        let mask = mask_ge_zero.equal(mask_le_size);

        let mask_float = mask.float();
        let valid_mask = mask_float.sum_dim(1).equal_elem(3.0).float();

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
        let uz = u_vec
            .clone()
            .slice([0..batch_size, 2..3])
            .flatten::<1>(0, 1);

        let bx = Self::compute_basis_tensor(ux);
        let by = Self::compute_basis_tensor(uy);
        let bz = Self::compute_basis_tensor(uz);

        // Weights: [Batch, 4, 4, 4] -> Flatten to [Batch, 64]
        let weights = bx.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3)
            * by.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3)
            * bz.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);

        let weights = weights.reshape([batch_size, 64, 1]);

        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        let nz = self.grid_size[2] as i32;

        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);

        let i_idx = range.clone().reshape([1, 4, 1, 1]);
        let j_idx = range.clone().reshape([1, 1, 4, 1]);
        let k_idx = range.clone().reshape([1, 1, 1, 4]);

        let base_x = base_index
            .clone()
            .slice([0..batch_size, 0..1])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);
        let base_y = base_index
            .clone()
            .slice([0..batch_size, 1..2])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);
        let base_z = base_index
            .clone()
            .slice([0..batch_size, 2..3])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);

        let idx_x = base_x + i_idx;
        let idx_y = base_y + j_idx;
        let idx_z = base_z + k_idx;

        let zeros = Tensor::<B, 4, burn::tensor::Int>::zeros([1, 4, 4, 4], &device);

        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 64]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 64]);
        let idx_z_flat = (idx_z + zeros.clone()).reshape([batch_size, 64]);

        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        let idx_z_clamped = idx_z_flat.clamp(0, nz - 1);

        let stride_z = nx * ny;
        let stride_y = nx;

        let flat_indices = idx_z_clamped * stride_z + idx_y_clamped * stride_y + idx_x_clamped;

        let gather_indices = flat_indices.reshape([batch_size * 64]);
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*64, 3]
        let coeffs = coeffs.reshape([batch_size, 64, 3]);

        let displacement = (coeffs * weights).sum_dim(1).flatten::<2>(1, 2);

        let masked_displacement = displacement * valid_mask;

        points + masked_displacement
    }
}

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::transform::bspline::BSplineTransform;

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    pub(crate) fn transform_4d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = points.shape().dims[0];
        const CHUNK_SIZE: usize = 16384; // Smaller chunk for 4D

        if batch_size <= CHUNK_SIZE {
            self.transform_4d_chunk(points)
        } else {
            let num_chunks = (batch_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, batch_size);

                let chunk_points = points.clone().slice([start..end]);
                let chunk_result = self.transform_4d_chunk(chunk_points);
                chunks.push(chunk_result);
            }
            Tensor::cat(chunks, 0)
        }
    }

    pub(crate) fn transform_4d_chunk(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];

        let grid_coords = self.world_to_grid_tensor(points.clone());

        // Mask
        let zero_tensor = Tensor::<B, 1>::zeros([4], &device).reshape([1, 4]);
        let size_tensor = Tensor::<B, 1>::from_floats(
            [
                self.grid_size[0] as f32 - 1.0,
                self.grid_size[1] as f32 - 1.0,
                self.grid_size[2] as f32 - 1.0,
                self.grid_size[3] as f32 - 1.0,
            ],
            &device,
        )
        .reshape([1, 4]);

        let mask = grid_coords
            .clone()
            .greater_equal(zero_tensor)
            .equal(grid_coords.clone().lower_equal(size_tensor));
        let valid_mask = mask.float().sum_dim(1).equal_elem(4.0).float();

        // Interpolation
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
        let uw = u_vec
            .clone()
            .slice([0..batch_size, 3..4])
            .flatten::<1>(0, 1);

        let bx = Self::compute_basis_tensor(ux);
        let by = Self::compute_basis_tensor(uy);
        let bz = Self::compute_basis_tensor(uz);
        let bw = Self::compute_basis_tensor(uw);

        // Weights: [Batch, 4, 4, 4, 4] -> [Batch, 256]
        let weights = bx
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4)
            * by.unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(3)
                .unsqueeze_dim::<5>(4)
            * bz.unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(1)
                .unsqueeze_dim::<5>(4)
            * bw.unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(1)
                .unsqueeze_dim::<5>(1);
        let weights = weights.reshape([batch_size, 256, 1]);

        let nx = self.grid_size[0] as i32;
        let ny = self.grid_size[1] as i32;
        let nz = self.grid_size[2] as i32;
        let nw = self.grid_size[3] as i32;

        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);

        let i_idx = range.clone().reshape([1, 4, 1, 1, 1]);
        let j_idx = range.clone().reshape([1, 1, 4, 1, 1]);
        let k_idx = range.clone().reshape([1, 1, 1, 4, 1]);
        let l_idx = range.clone().reshape([1, 1, 1, 1, 4]);

        let base_x = base_index
            .clone()
            .slice([0..batch_size, 0..1])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let base_y = base_index
            .clone()
            .slice([0..batch_size, 1..2])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let base_z = base_index
            .clone()
            .slice([0..batch_size, 2..3])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let base_w = base_index
            .clone()
            .slice([0..batch_size, 3..4])
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);

        let idx_x = base_x + i_idx;
        let idx_y = base_y + j_idx;
        let idx_z = base_z + k_idx;
        let idx_w = base_w + l_idx;

        let zeros = Tensor::<B, 5, burn::tensor::Int>::zeros([1, 4, 4, 4, 4], &device);

        let idx_x_flat = (idx_x + zeros.clone()).reshape([batch_size, 256]);
        let idx_y_flat = (idx_y + zeros.clone()).reshape([batch_size, 256]);
        let idx_z_flat = (idx_z + zeros.clone()).reshape([batch_size, 256]);
        let idx_w_flat = (idx_w + zeros.clone()).reshape([batch_size, 256]);

        let idx_x_clamped = idx_x_flat.clamp(0, nx - 1);
        let idx_y_clamped = idx_y_flat.clamp(0, ny - 1);
        let idx_z_clamped = idx_z_flat.clamp(0, nz - 1);
        let idx_w_clamped = idx_w_flat.clamp(0, nw - 1);

        let stride_w = nx * ny * nz;
        let stride_z = nx * ny;
        let stride_y = nx;

        let flat_indices = idx_w_clamped * stride_w
            + idx_z_clamped * stride_z
            + idx_y_clamped * stride_y
            + idx_x_clamped;

        let gather_indices = flat_indices.reshape([batch_size * 256]);
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*256, 4]
        let coeffs = coeffs.reshape([batch_size, 256, 4]);

        let displacement = (coeffs * weights).sum_dim(1).flatten::<2>(1, 2);
        points + displacement * valid_mask
    }
}

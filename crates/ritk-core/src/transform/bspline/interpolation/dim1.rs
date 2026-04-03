use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::transform::bspline::BSplineTransform;

impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    pub(crate) fn transform_1d(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        let batch_size = points.shape().dims[0];

        let grid_coords = self.world_to_grid_tensor(points.clone()); // [Batch, 1]

        let zero_tensor = Tensor::<B, 1>::zeros([1], &device).reshape([1, 1]);
        let size_tensor =
            Tensor::<B, 1>::from_floats([self.grid_size[0] as f32 - 1.0], &device).reshape([1, 1]);

        let mask = grid_coords
            .clone()
            .greater_equal(zero_tensor)
            .equal(grid_coords.clone().lower_equal(size_tensor));
        let valid_mask = mask.float(); // [Batch, 1]

        let grid_indices_float = grid_coords.clone().floor();
        let u_vec = grid_coords - grid_indices_float.clone();
        let base_index = grid_indices_float.int() - 1;

        let ux = u_vec.flatten::<1>(0, 1);
        let bx = Self::compute_basis_tensor(ux); // [Batch, 4]
        let weights = bx.unsqueeze_dim::<3>(2); // [Batch, 4, 1]

        let nx = self.grid_size[0] as i32;
        let range = Tensor::<B, 1, burn::tensor::Int>::from_ints([0, 1, 2, 3], &device);
        let i_idx = range.reshape([1, 4]);

        let base_x = base_index.clone(); // [Batch, 1]
        let idx_x = base_x + i_idx; // [Batch, 4]

        let idx_x_clamped = idx_x.clamp(0, nx - 1); // [Batch, 4]
        let flat_indices = idx_x_clamped;

        let gather_indices = flat_indices.reshape([batch_size * 4]);
        let coeffs = self.coefficients.val().clone().select(0, gather_indices); // [Batch*4, 1]
        let coeffs = coeffs.reshape([batch_size, 4, 1]);

        let displacement = (coeffs * weights).sum_dim(1).flatten::<2>(1, 2); // [Batch, 1]
        points + displacement * valid_mask
    }
}

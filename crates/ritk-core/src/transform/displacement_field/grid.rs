use super::core::DisplacementField;
use crate::wgpu_compat::apply_row_chunks;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

impl<B: Backend, const D: usize> DisplacementField<B, D> {
    /// Convert physical continuous points iteratively to indices isolated across chunked dimensional grids.
    ///
    /// Follows the mathematically verified mapping $ v^T = (w - O) T $.
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        apply_row_chunks(
            points,
            crate::wgpu_compat::WGPU_CHUNK_SIZE,
            |chunk_points| {
                let diff = chunk_points - self.origin_tensor.clone();
                diff.matmul(self.world_to_index_matrix.clone())
            },
        )
    }
}

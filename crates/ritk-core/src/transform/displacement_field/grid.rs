use super::core::DisplacementField;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

impl<B: Backend, const D: usize> DisplacementField<B, D> {
    /// Convert physical continuous points iteratively to indices isolated across chunked dimensional grids.
    ///
    /// Follows the mathematically verified mapping $ v^T = (w - O) T $.
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n_points, _] = points.dims();

        const CHUNK_SIZE: usize = 32768;

        if n_points <= CHUNK_SIZE {
            let diff = points - self.origin_tensor.clone();
            diff.matmul(self.world_to_index_matrix.clone())
        } else {
            let mut chunks = Vec::new();
            let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n_points);
                let chunk_points = points.clone().slice([start..end]);

                let diff = chunk_points - self.origin_tensor.clone();
                let result = diff.matmul(self.world_to_index_matrix.clone());
                chunks.push(result);
            }

            Tensor::cat(chunks, 0)
        }
    }
}

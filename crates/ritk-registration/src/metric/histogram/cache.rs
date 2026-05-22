use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct HistogramCache<B: Backend> {
    /// World-space coordinates of all fixed-image voxels [N, D].
    pub points: Tensor<B, 2>,
    /// Precomputed Parzen weight matrix for the fixed image, transposed: [num_bins, N].
    /// Constant across all registration iterations because the fixed image never changes.
    /// Reusing this avoids O(N × num_bins) kernel computation and removes the fixed-image
    /// Parzen path from the autodiff graph on every iteration after the first.
    pub w_fixed_transposed: Option<Tensor<B, 2>>,
    pub shape: Vec<usize>,
    pub origin: Vec<f64>,
    pub spacing: Vec<f64>,
    pub direction: Vec<f64>,
}

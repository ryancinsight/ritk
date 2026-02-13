use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::image::grid;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::filter::gaussian::GaussianFilter;
use super::trait_::Metric;

/// Local Normalized Cross Correlation (LNCC) Metric.
///
/// Computes the normalized cross correlation within a local window around each pixel.
/// Robust to local intensity variations and bias fields.
///
/// LNCC = < (I - mean_I) * (J - mean_J) > / ( std_I * std_J )
#[derive(Clone)]
pub struct LocalNormalizedCrossCorrelation<B: Backend> {
    interpolator: LinearInterpolator,
    kernel_sigma: f64,
    epsilon: f64,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> LocalNormalizedCrossCorrelation<B> {
    /// Create a new LNCC metric.
    ///
    /// # Arguments
    /// * `kernel_sigma` - Standard deviation of the Gaussian kernel (mm) defining the local window size.
    pub fn new(kernel_sigma: f64) -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
            kernel_sigma,
            epsilon: 1e-5,
            _b: std::marker::PhantomData,
        }
    }

    fn compute_local_stats<const D: usize>(
        &self,
        img: Tensor<B, D>,
        filter: &GaussianFilter<B>,
        spacing: &ritk_core::spatial::Spacing<D>
    ) -> (Tensor<B, D>, Tensor<B, D>) {
        // Mean = I * K
        let mean = filter.apply_tensor(img.clone(), spacing);
        
        // MeanSq = I^2 * K
        let sq = img.powf_scalar(2.0);
        let mean_sq = filter.apply_tensor(sq, spacing);
        
        // Var = MeanSq - Mean^2
        let var = mean_sq - mean.clone().powf_scalar(2.0);
        
        // Clamp variance to avoid sqrt of negative due to float errors
        let var = var.clamp_min(0.0);
        
        (mean, var)
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for LocalNormalizedCrossCorrelation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Generate grid (Full, as we need the full spatial structure for convolution)
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();
        let fixed_indices = grid::generate_grid(fixed_shape, &device); // [N, D]
        let [n, _] = fixed_indices.dims();

        // 2. Resample moving image with chunking to avoid WGPU dispatch limits
        const CHUNK_SIZE: usize = 32768;
        
        let moving_values_flat = if n <= CHUNK_SIZE {
            let fixed_points = fixed.index_to_world_tensor(fixed_indices);
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);
            self.interpolator.interpolate(moving.data(), moving_indices)
        } else {
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                
                let chunk_indices = fixed_indices.clone().slice([start..end]);
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_values = self.interpolator.interpolate(moving.data(), chunk_moving_indices);
                chunks.push(chunk_values);
            }
            Tensor::cat(chunks, 0)
        };
        
        // 3. Reshape back to spatial dimensions for convolution
        // fixed.shape() gives [D, H, W] etc.
        // We need to cast shape dims to array
        let shape_dims: [usize; D] = fixed.data().shape().dims(); // [usize; D]
        let moving_values = moving_values_flat.reshape(burn::tensor::Shape::new(shape_dims));
        
        let fixed_values = fixed.data().clone(); // Already spatial [D, H, W]

        // 4. Setup filter
        let filter = GaussianFilter::new(vec![self.kernel_sigma; D]);
        
        // 5. Compute local stats
        let (mean_f, var_f) = self.compute_local_stats(fixed_values.clone(), &filter, fixed.spacing());
        let (mean_m, var_m) = self.compute_local_stats(moving_values.clone(), &filter, fixed.spacing());
        
        // 6. Compute Cross Term
        // Cross = (F * M) * K
        let fm = fixed_values * moving_values;
        let mean_fm = filter.apply_tensor(fm, fixed.spacing());
        
        // Covariance = MeanFM - MeanF * MeanM
        let cov = mean_fm - (mean_f * mean_m);
        
        // 7. Compute LNCC
        // Denom = sqrt(VarF * VarM)
        let denom = (var_f * var_m).sqrt() + self.epsilon;
        
        let lncc = cov / denom;
        
        // 8. Return negative mean (to minimize)
        lncc.mean().neg()
    }

    fn name(&self) -> &'static str {
        "LocalNormalizedCrossCorrelation"
    }
}

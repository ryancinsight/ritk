//! Normalized Cross Correlation (NCC) metric implementation.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use super::trait_::{Metric, utils};

/// Normalized Cross Correlation Metric.
///
/// Computes the zero-normalized cross correlation between pixel intensities:
/// NCC = sum((F - mean(F)) * (M - mean(M))) / sqrt(sum((F - mean(F))^2) * sum((M - mean(M))^2))
///
/// Returns negative NCC as loss (to be minimized).
/// Range: [-1, 1], where -1 is perfect correlation (minimized loss).
pub struct NormalizedCrossCorrelation {
    interpolator: LinearInterpolator,
}

impl NormalizedCrossCorrelation {
    /// Create a new NCC metric.
    pub fn new() -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
        }
    }
}

impl Default for NormalizedCrossCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedCrossCorrelation {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let device = fixed.data().device();
        let fixed_shape = fixed.shape();
        
        // 1. Generate grid
        let fixed_indices = grid::generate_grid(fixed_shape, &device);
        let [n, _] = fixed_indices.dims();

        // 2. Process in chunks to avoid WGPU limits
        const CHUNK_SIZE: usize = 32768;
        
        // We need to compute sums for the numerator and denominators
        // Sum( (F-meanF)*(M-meanM) ), Sum( (F-meanF)^2 ), Sum( (M-meanM)^2 )
        // But to do that we first need the means.
        // Means can also be computed in chunks or by accumulating sums.
        
        // Strategy:
        // Pass 1: Compute sums of F and M to get means.
        // Pass 2: Compute correlation sums using means.
        
        // However, this requires two passes over the data (or at least the transform).
        // Since we can't easily cache the transformed moving image (memory limits), 
        // we might have to re-compute the transform or store the values if memory allows.
        // Given we are chunking for dispatch limits, not necessarily memory limits (though related),
        // let's try to compute means first.
        
        // Actually, we can use the online algorithm or just simple accumulation for means.
        
        let (sum_f, sum_m) = if n <= CHUNK_SIZE {
            let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone());
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);
            let moving_values = self.interpolator.interpolate(moving.data(), moving_indices);
            let fixed_values = fixed.data().clone().reshape([n]);
            
            (fixed_values.sum(), moving_values.sum())
        } else {
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut s_f = Tensor::zeros([1], &device);
            let mut s_m = Tensor::zeros([1], &device);
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                
                let chunk_indices = fixed_indices.clone().slice([start..end]);
                
                // Fixed values
                // Flatten fixed data once or slice it? Slicing 1D view is better if possible.
                // But fixed.data() is [D, H, W].
                // We can use index_select or just reshape and slice.
                // Reshaping the whole fixed image is cheap (metadata).
                let fixed_values_flat = fixed.data().clone().reshape([n]);
                let chunk_fixed_values = fixed_values_flat.slice([start..end]);
                
                // Moving values
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_moving_values = self.interpolator.interpolate(moving.data(), chunk_moving_indices);
                
                s_f = s_f + chunk_fixed_values.sum();
                s_m = s_m + chunk_moving_values.sum();
            }
            (s_f, s_m)
        };
        
        let mean_f = sum_f / (n as f32);
        let mean_m = sum_m / (n as f32);
        
        // Pass 2: Compute NCC components
        // Numerator: sum((F - meanF) * (M - meanM))
        // DenomF: sum((F - meanF)^2)
        // DenomM: sum((M - meanM)^2)
        
        let (numerator, denom_f, denom_m) = if n <= CHUNK_SIZE {
             let fixed_points = fixed.index_to_world_tensor(fixed_indices);
             let moving_points = transform.transform_points(fixed_points);
             let moving_indices = moving.world_to_index_tensor(moving_points);
             let moving_values = self.interpolator.interpolate(moving.data(), moving_indices);
             let fixed_values = fixed.data().clone().reshape([n]);
             
             let f_centered = fixed_values - mean_f.clone();
             let m_centered = moving_values - mean_m.clone();
             
             let num = (f_centered.clone() * m_centered.clone()).sum();
             let d_f = f_centered.powf_scalar(2.0).sum();
             let d_m = m_centered.powf_scalar(2.0).sum();
             
             (num, d_f, d_m)
        } else {
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut acc_num = Tensor::zeros([1], &device);
            let mut acc_d_f = Tensor::zeros([1], &device);
            let mut acc_d_m = Tensor::zeros([1], &device);
            
             for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                
                let chunk_indices = fixed_indices.clone().slice([start..end]);
                
                let fixed_values_flat = fixed.data().clone().reshape([n]);
                let chunk_fixed_values = fixed_values_flat.slice([start..end]);
                
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_moving_values = self.interpolator.interpolate(moving.data(), chunk_moving_indices);
                
                let f_centered = chunk_fixed_values - mean_f.clone();
                let m_centered = chunk_moving_values - mean_m.clone();
                
                acc_num = acc_num + (f_centered.clone() * m_centered.clone()).sum();
                acc_d_f = acc_d_f + f_centered.powf_scalar(2.0).sum();
                acc_d_m = acc_d_m + m_centered.powf_scalar(2.0).sum();
            }
            (acc_num, acc_d_f, acc_d_m)
        };

        // Add epsilon to avoid division by zero
        let epsilon = 1e-10;
        let denominator = (denom_f * denom_m).sqrt() + epsilon;

        let ncc = numerator / denominator;

        // 5. Return Loss (-NCC)
        ncc.neg()
    }

    fn name(&self) -> &'static str {
        "NormalizedCrossCorrelation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Point, Spacing, Direction};
    use ritk_core::transform::TranslationTransform;
    use burn::tensor::{Shape, TensorData};

    type B = NdArray<f32>;

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
        let spacing = Spacing::new([1.0, 1.0, 1.0]);
        let origin = Point::new([0.0, 0.0, 0.0]);
        let direction = Direction::identity();
        Image::new(tensor, origin, spacing, direction)
    }

    #[test]
    fn test_ncc_identical() {
        let size = 10;
        let count = size * size * size;
        let data: Vec<f32> = (0..count).map(|x| (x as f32)).collect(); // Gradient
        let image = create_test_image(data.clone(), [size, size, size]);
        
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&image, &image, &transform);
        
        let loss_val = loss.into_scalar();
        // Identical images should have NCC = 1.0, so Loss = -1.0
        assert!((loss_val + 1.0).abs() < 1e-5, "NCC for identical images should be 1.0 (loss -1.0), got {}", loss_val);
    }

    #[test]
    fn test_ncc_linear_relationship() {
        let size = 10;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|x| (x as f32)).collect();
        // data2 = 2 * data1 + 10. Linear relationship should still give NCC = 1.0
        let data2: Vec<f32> = data1.iter().map(|&x| 2.0 * x + 10.0).collect();
        
        let fixed = create_test_image(data1, [size, size, size]);
        let moving = create_test_image(data2, [size, size, size]);
        
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);
        
        let loss_val = loss.into_scalar();
        assert!((loss_val + 1.0).abs() < 1e-4, "NCC for linear relationship should be 1.0 (loss -1.0), got {}", loss_val);
    }

    #[test]
    fn test_ncc_inverse_relationship() {
        let size = 10;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|x| (x as f32)).collect();
        // data2 = -data1. Inverse relationship should give NCC = -1.0, Loss = 1.0
        let data2: Vec<f32> = data1.iter().map(|&x| -x).collect();
        
        let fixed = create_test_image(data1, [size, size, size]);
        let moving = create_test_image(data2, [size, size, size]);
        
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);
        
        let loss_val = loss.into_scalar();
        assert!((loss_val - 1.0).abs() < 1e-4, "NCC for inverse relationship should be -1.0 (loss 1.0), got {}", loss_val);
    }

    #[test]
    fn test_ncc_uncorrelated() {
        let count = 100;
        let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
        // Alternating pattern should have low correlation with linear ramp
        let data2: Vec<f32> = (0..count).map(|x| if x % 2 == 0 { 10.0 } else { -10.0 }).collect();
        
        let size = 100; // 1D like
        let fixed = create_test_image(data1, [size, 1, 1]);
        let moving = create_test_image(data2, [size, 1, 1]);
        
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);
        
        let loss_val = loss.into_scalar();
        // NCC should be close to 0, so loss = -NCC should be close to 0
        assert!(loss_val.abs() < 0.5, "NCC for uncorrelated data should be low (close to 0), got loss {}", loss_val);
    }
}

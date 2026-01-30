//! Mutual Information metric implementation.

use burn::tensor::{Tensor, Int};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use super::trait_::{Metric, utils};

/// Mutual Information Metric using Parzen Window estimation.
///
/// Computes the mutual information between two images:
/// MI(A, B) = H(A) + H(B) - H(A, B)
/// where H is the Shannon entropy.
///
/// Uses differentiable histogram estimation (Parzen Window) to allow
/// gradient-based optimization.
pub struct MutualInformation {
    interpolator: LinearInterpolator,
    num_bins: usize,
    sigma: f64,
}

impl MutualInformation {
    /// Create a new Mutual Information metric.
    ///
    /// # Arguments
    /// * `num_bins` - Number of histogram bins (default: 32)
    /// * `sigma` - Standard deviation for the Parzen window kernel (default: 0.1)
    pub fn new(num_bins: usize, sigma: f64) -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
            num_bins,
            sigma,
        }
    }

    /// Compute soft histogram.
    ///
    /// # Arguments
    /// * `values` - Input values [N]
    /// * `min_val` - Minimum value for histogram range
    /// * `max_val` - Maximum value for histogram range
    /// * `bins` - Tensor of bin centers [num_bins]
    /// * `sigma` - Kernel width
    fn compute_histogram<B: Backend>(
        values: Tensor<B, 1>,
        bins: Tensor<B, 1>,
        sigma: f64,
    ) -> Tensor<B, 1> {
        let n = values.dims()[0];
        let num_bins = bins.dims()[0];

        // Reshape for broadcasting:
        // values: [N, 1]
        // bins: [1, num_bins]
        let values_exp = values.reshape([n, 1]);
        let bins_exp = bins.reshape([1, num_bins]);

        // Compute difference: [N, num_bins]
        let diff = values_exp - bins_exp;

        // Gaussian kernel: exp(-0.5 * (x/sigma)^2)
        // We omit normalization constant as it cancels out in entropy or scales uniformly
        let exponent = diff.powf_scalar(2.0) * (-0.5 / (sigma * sigma));
        let weights = exponent.exp();

        // Sum over samples (dim 0) -> [1, num_bins] -> [num_bins]
        let histogram = weights.sum_dim(0).reshape([num_bins]);

        // Normalize to sum to 1 (probability distribution)
        let sum = histogram.clone().sum() + 1e-10;
        histogram / sum
    }

    /// Compute joint soft histogram.
    fn compute_joint_histogram<B: Backend>(
        val_a: Tensor<B, 1>,
        val_b: Tensor<B, 1>,
        bins_a: Tensor<B, 1>,
        bins_b: Tensor<B, 1>,
        sigma: f64,
    ) -> Tensor<B, 2> {
        let n = val_a.dims()[0];
        let num_bins_a = bins_a.dims()[0];
        let num_bins_b = bins_b.dims()[0];

        // Reshape:
        // val_a: [N, 1]
        // bins_a: [1, num_bins_a]
        let val_a_exp = val_a.reshape([n, 1]);
        let bins_a_exp = bins_a.reshape([1, num_bins_a]);
        
        // Kernel A: [N, num_bins_a]
        let diff_a = val_a_exp - bins_a_exp;
        let weights_a = (diff_a.powf_scalar(2.0) * (-0.5 / (sigma * sigma))).exp();

        // val_b: [N, 1]
        // bins_b: [1, num_bins_b]
        let val_b_exp = val_b.reshape([n, 1]);
        let bins_b_exp = bins_b.reshape([1, num_bins_b]);

        // Kernel B: [N, num_bins_b]
        let diff_b = val_b_exp - bins_b_exp;
        let weights_b = (diff_b.powf_scalar(2.0) * (-0.5 / (sigma * sigma))).exp();

        // Compute outer product for each sample and sum?
        // Actually, Joint histogram entry (i, j) is sum_k (w_a(k, i) * w_b(k, j))
        // This is matrix multiplication: weights_a^T * weights_b
        // weights_a: [N, Na] -> Transpose -> [Na, N]
        // weights_b: [N, Nb]
        // Result: [Na, Nb]
        
        let weights_a_t = weights_a.transpose();
        let joint_hist = weights_a_t.matmul(weights_b); // [Na, Nb]

        // Normalize
        let sum = joint_hist.clone().sum();
        joint_hist / sum.reshape([1, 1])
    }

    /// Compute entropy of a probability distribution.
    fn compute_entropy<B: Backend, const D: usize>(probs: Tensor<B, D>) -> Tensor<B, 1> {
        let epsilon = 1e-10;
        let log_probs = (probs.clone() + epsilon).log();
        (probs * log_probs).sum().neg()
    }
}

impl Default for MutualInformation {
    fn default() -> Self {
        Self::new(32, 0.1)
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for MutualInformation {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let device = fixed.data().device();
        let fixed_shape = fixed.shape();
        
        // 1. Sampling (same as MSE)
        let fixed_indices = utils::generate_grid(fixed_shape, &device);
        let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone());
        let moving_points = transform.transform_points(fixed_points);
        let moving_indices = moving.world_to_index_tensor(moving_points);
        let moving_values = self.interpolator.interpolate(moving.data(), moving_indices);
        let fixed_values = fixed.data().clone().reshape([fixed_indices.dims()[0]]);

        // 2. Setup Bins
        // We determine the range dynamically using min/max of the current batch.
        
        // Construct bins: linear space from min to max
        fn create_bins<B: Backend>(min: Tensor<B, 1>, max: Tensor<B, 1>, count: usize) -> Tensor<B, 1> {
            let device = min.device();
            let step = (max.clone() - min.clone()) / ((count - 1) as f64);
            let indices = Tensor::<B, 1, Int>::arange(0..count as i64, &device).float();
            min + indices * step
        }

        let f_min_t = fixed_values.clone().min();
        let f_max_t = fixed_values.clone().max();
        let fixed_bins = create_bins(f_min_t, f_max_t, self.num_bins);

        let m_min_t = moving_values.clone().min();
        let m_max_t = moving_values.clone().max();
        let moving_bins = create_bins(m_min_t, m_max_t, self.num_bins);

        // 3. Compute Histograms
        // Marginal H(F) - constant w.r.t transform, but good to compute for consistency
        let p_f = Self::compute_histogram(fixed_values.clone(), fixed_bins.clone(), self.sigma);
        let h_f = Self::compute_entropy(p_f);

        // Marginal H(M)
        let p_m = Self::compute_histogram(moving_values.clone(), moving_bins.clone(), self.sigma);
        let h_m = Self::compute_entropy(p_m);

        // Joint H(F, M)
        let p_fm = Self::compute_joint_histogram(
            fixed_values, moving_values,
            fixed_bins, moving_bins,
            self.sigma
        );
        let h_fm = Self::compute_entropy(p_fm);

        // 4. MI = H(F) + H(M) - H(F, M)
        // We want to maximize MI, so we return negative MI as loss.
        // Loss = -MI = H(F, M) - H(F) - H(M)
        h_fm - h_f - h_m
    }

    fn name(&self) -> &'static str {
        "MutualInformation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Point, Spacing, Direction};
    use ritk_core::transform::TranslationTransform;
    use burn::tensor::Shape;
    use burn::tensor::TensorData;

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
    fn test_mutual_information_identical() {
        let size = 10;
        let count = size * size * size;
        let data: Vec<f32> = (0..count).map(|x| (x as f32) / (count as f32)).collect();
        let image = create_test_image(data.clone(), [size, size, size]);
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let mi_metric = MutualInformation::new(32, 0.1);
        let loss = mi_metric.forward(&image, &image, &transform);
        
        // MI(X, X) = H(X).
        // Loss = -MI = -H(X).
        // Since distribution is roughly uniform, H(X) ~ ln(N) or similar.
        // We just check it runs and gives a finite value.
        let loss_val = loss.into_scalar();
        assert!(loss_val.is_finite());
        // Since X=Y, H(X,Y) = H(X). So MI = H(X) + H(Y) - H(X,Y) = H(X).
        // Loss = -H(X).
        // H(X) should be positive, so loss should be negative.
        assert!(loss_val < 0.0);
    }

    #[test]
    fn test_mutual_information_different() {
        let size = 10;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|x| (x as f32) / (count as f32)).collect();
        // Shift data2
        let data2: Vec<f32> = (0..count).map(|x| ((x + count/2) % count) as f32 / (count as f32)).collect();
        
        let fixed = create_test_image(data1, [size, size, size]);
        let moving = create_test_image(data2, [size, size, size]);
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));
        
        let mi_metric = MutualInformation::new(32, 0.1);
        let loss = mi_metric.forward(&fixed, &moving, &transform);
        
        let loss_val = loss.into_scalar();
        assert!(loss_val.is_finite());
    }
}

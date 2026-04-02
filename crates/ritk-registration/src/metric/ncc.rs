//! Normalized Cross Correlation (NCC) metric implementation.
//!
//! # Theorem: Zero-Normalized Cross Correlation (Single-Pass Algebraic Moments)
//!
//! **Theorem** (Lewis 1995, *Vision Interface* 9:120–123):
//! The zero-normalized cross correlation between fixed image $F$ and moving image $M$
//! over $N$ voxels can be computed exactly in a single pass using raw statistical moments:
//!
//! ```text
//! S_F = ΣF,  S_M = ΣM,  S_{FF} = ΣF²,  S_{MM} = ΣM²,  S_{FM} = Σ(F·M)
//!
//! Numerator   = S_{FM} - (S_F · S_M) / N
//! Variance_F  = S_{FF} - (S_F)² / N
//! Variance_M  = S_{MM} - (S_M)² / N
//!
//! NCC(F, M)   = Numerator / √(Variance_F · Variance_M + ε)
//! ```
//! By accumulating these five moments simultaneously, we eliminate the need for a
//! two-pass interpolation loop, cutting scattered memory accesses and coordinate
//! transformations strictly in half (O(N) exact).

use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::grid;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

/// Normalized Cross Correlation Metric.
///
/// Computes the zero-normalized cross correlation between pixel intensities
/// using a single-pass algebraic moment accumulation to minimize memory bandwidth
/// and interpolation overhead.
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

        // 1. Generate dense coordinate grid
        let fixed_indices = grid::generate_grid(fixed_shape, &device);
        let [n, _] = fixed_indices.dims();

        // 2. Process in chunks to respect GPU dispatch limits while maintaining single-pass
        const CHUNK_SIZE: usize = 32768;

        // Compute the 5 raw statistical moments in a single pass:
        let (s_f, s_m, s_ff, s_mm, s_fm) = if n <= CHUNK_SIZE {
            let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone());
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);

            let m = self.interpolator.interpolate(moving.data(), moving_indices);
            let f = fixed.data().clone().reshape([n]);

            (
                f.clone().sum(),
                m.clone().sum(),
                f.clone().powf_scalar(2.0).sum(),
                m.clone().powf_scalar(2.0).sum(),
                (f * m).sum(),
            )
        } else {
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

            let mut acc_f = Tensor::zeros([1], &device);
            let mut acc_m = Tensor::zeros([1], &device);
            let mut acc_ff = Tensor::zeros([1], &device);
            let mut acc_mm = Tensor::zeros([1], &device);
            let mut acc_fm = Tensor::zeros([1], &device);

            let fixed_values_flat = fixed.data().clone().reshape([n]);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);

                let chunk_indices = fixed_indices.clone().slice([start..end]);
                let f = fixed_values_flat.clone().slice([start..end]);

                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let m = self
                    .interpolator
                    .interpolate(moving.data(), chunk_moving_indices);

                acc_f = acc_f + f.clone().sum();
                acc_m = acc_m + m.clone().sum();
                acc_ff = acc_ff + f.clone().powf_scalar(2.0).sum();
                acc_mm = acc_mm + m.clone().powf_scalar(2.0).sum();
                acc_fm = acc_fm + (f * m).sum();
            }
            (acc_f, acc_m, acc_ff, acc_mm, acc_fm)
        };

        // 3. Algebraic reduction to Central Moments
        let n_f32 = n as f32;
        let num = s_fm - (s_f.clone() * s_m.clone()) / n_f32;
        let d_f = s_ff - s_f.powf_scalar(2.0) / n_f32;
        let d_m = s_mm - s_m.powf_scalar(2.0) / n_f32;

        let epsilon = 1e-10_f32;
        // Clamp variances to epsilon to guarantee numerical stability when identical or constant
        let d_f_clamped = d_f.clamp_min(epsilon);
        let d_m_clamped = d_m.clamp_min(epsilon);

        let denominator = (d_f_clamped * d_m_clamped).sqrt();
        let ncc = num / denominator;

        // 4. Return negative NCC (to frame as minimization loss)
        ncc.neg()
    }

    fn name(&self) -> &'static str {
        "NormalizedCrossCorrelation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

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
        let data: Vec<f32> = (0..count).map(|x| x as f32).collect(); // Gradient
        let image = create_test_image(data.clone(), [size, size, size]);

        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));

        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&image, &image, &transform);

        let loss_val = loss.into_scalar();
        // Identical images should have NCC = 1.0, so Loss = -1.0
        assert!(
            (loss_val + 1.0).abs() < 1e-4,
            "NCC for identical images should be 1.0 (loss -1.0), got {}",
            loss_val
        );
    }

    #[test]
    fn test_ncc_linear_relationship() {
        let size = 10;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
        // data2 = 2 * data1 + 10. Linear relationship should still give NCC = 1.0
        let data2: Vec<f32> = data1.iter().map(|&x| 2.0 * x + 10.0).collect();

        let fixed = create_test_image(data1, [size, size, size]);
        let moving = create_test_image(data2, [size, size, size]);

        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));

        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);

        let loss_val = loss.into_scalar();
        assert!(
            (loss_val + 1.0).abs() < 1e-4,
            "NCC for linear relationship should be 1.0 (loss -1.0), got {}",
            loss_val
        );
    }

    #[test]
    fn test_ncc_inverse_relationship() {
        let size = 10;
        let count = size * size * size;
        let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
        // data2 = -data1. Inverse relationship should give NCC = -1.0, Loss = 1.0
        let data2: Vec<f32> = data1.iter().map(|&x| -x).collect();

        let fixed = create_test_image(data1, [size, size, size]);
        let moving = create_test_image(data2, [size, size, size]);

        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));

        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);

        let loss_val = loss.into_scalar();
        assert!(
            (loss_val - 1.0).abs() < 1e-4,
            "NCC for inverse relationship should be -1.0 (loss 1.0), got {}",
            loss_val
        );
    }

    #[test]
    fn test_ncc_uncorrelated() {
        let count = 100;
        let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
        // Alternating pattern should have low correlation with linear ramp
        let data2: Vec<f32> = (0..count)
            .map(|x| if x % 2 == 0 { 10.0 } else { -10.0 })
            .collect();

        let size = 100; // 1D like
        let fixed = create_test_image(data1, [size, 1, 1]);
        let moving = create_test_image(data2, [size, 1, 1]);

        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::zeros([3], &device));

        let ncc_metric = NormalizedCrossCorrelation::new();
        let loss = ncc_metric.forward(&fixed, &moving, &transform);

        let loss_val = loss.into_scalar();
        assert!(
            loss_val.abs() < 0.5,
            "NCC for uncorrelated data should be low (close to 0), got loss {}",
            loss_val
        );
    }
}

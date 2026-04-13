//! Z-score intensity normalization.
//!
//! # Mathematical Specification
//! Given an image X with population mean μ and population standard deviation σ:
//!
//!   Z(x) = (x − μ) / (σ + ε),   ε = 1e-8
//!
//! After normalization: E[Z] ≈ 0, Var[Z] ≈ 1.
//! The ε term prevents division by zero on constant images.
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the input.
//! - μ and σ are computed from the full image population (not a sample).

use crate::image::Image;
use crate::statistics::image_statistics::compute_statistics;
use burn::tensor::backend::Backend;

/// Z-score normalizer.
///
/// Transforms image intensities to zero mean and unit variance using
/// population statistics derived from the image itself.
pub struct ZScoreNormalizer;

impl ZScoreNormalizer {
    /// Create a new `ZScoreNormalizer`.
    pub fn new() -> Self {
        Self
    }

    /// Normalize `image` to zero mean, unit variance.
    ///
    /// # Formula
    /// `output = (input − mean) / (std + 1e-8)`
    ///
    /// Spatial metadata is preserved exactly.
    pub fn normalize<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let stats = compute_statistics(image);
        let mean = stats.mean;
        let std = stats.std;
        let denom = std + 1e-8_f32;

        let normalized = image.data().clone().sub_scalar(mean).div_scalar(denom);

        Image::new(
            normalized,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        )
    }
}

impl Default for ZScoreNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image(data: Vec<f32>, len: usize) -> Image<TestBackend, 1> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([len])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    // ── Positive tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_zscore_zero_mean() {
        // Values [1, 2, 3, 4, 5]: mean = 3.0
        // After normalization the mean of the output must be ≈ 0.
        let image = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let stats = crate::statistics::image_statistics::compute_statistics(&result);
        assert!(
            stats.mean.abs() < 1e-5,
            "output mean must be ≈ 0, got {}",
            stats.mean
        );
    }

    #[test]
    fn test_zscore_unit_variance() {
        // Values [1, 2, 3, 4, 5]:
        //   variance = Σ(xᵢ−3)²/5 = (4+1+0+1+4)/5 = 2  →  std = √2
        //   After division by (√2 + ε), output std ≈ 1.
        let image = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let stats = crate::statistics::image_statistics::compute_statistics(&result);
        assert!(
            (stats.std - 1.0).abs() < 1e-3,
            "output std must be ≈ 1, got {}",
            stats.std
        );
    }

    #[test]
    fn test_zscore_preserves_metadata() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        assert_eq!(result.origin(), &origin, "origin must be preserved");
        assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
        assert_eq!(
            result.direction(),
            &direction,
            "direction must be preserved"
        );
        assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
    }

    #[test]
    fn test_zscore_known_values() {
        // Values [1, 3]:
        //   mean = 2.0
        //   variance = ((1−2)² + (3−2)²) / 2 = 1.0 → std = 1.0
        //   z(1) = (1−2)/(1+1e-8) ≈ −1.0
        //   z(3) = (3−2)/(1+1e-8) ≈  1.0
        let image = make_image(vec![1.0, 3.0], 2);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let result_data = result.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert!(
            (slice[0] - (-1.0)).abs() < 1e-4,
            "z(1) ≈ -1.0, got {}",
            slice[0]
        );
        assert!(
            (slice[1] - 1.0).abs() < 1e-4,
            "z(3) ≈ 1.0, got {}",
            slice[1]
        );
    }

    // ── Boundary / edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_zscore_constant_image_does_not_panic() {
        // Constant image: std = 0. Division by (0 + 1e-8) must not panic.
        // All output values = (5 − 5) / 1e-8 = 0.
        let image = make_image(vec![5.0; 8], 8);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let result_data = result.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();
        for &v in slice.iter() {
            assert!(
                v.abs() < 1e-3,
                "constant image must normalize to ≈ 0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_zscore_single_voxel() {
        // Single voxel: mean = value, std = 0 → output = 0.
        let image = make_image(vec![100.0], 1);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let result_data = result.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();
        assert!(
            slice[0].abs() < 1e-3,
            "single voxel output = 0, got {}",
            slice[0]
        );
    }

    #[test]
    fn test_zscore_negative_values_preserved_sign() {
        // Values [−2, −1, 0, 1, 2]: mean = 0, std = √2 ≈ 1.4142
        // z(−2) < z(−1) < z(0) == 0 < z(1) < z(2): ordering preserved.
        let image = make_image(vec![-2.0, -1.0, 0.0, 1.0, 2.0], 5);
        let normalizer = ZScoreNormalizer::new();
        let result = normalizer.normalize(&image);

        let result_data = result.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        for i in 0..4 {
            assert!(
                slice[i] < slice[i + 1],
                "ordering violated at index {}: {} >= {}",
                i,
                slice[i],
                slice[i + 1]
            );
        }
        // Central value (0) maps to 0.
        assert!(slice[2].abs() < 1e-5, "z(0) must be 0, got {}", slice[2]);
    }
}

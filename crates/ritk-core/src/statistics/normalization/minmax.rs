//! Min-max intensity normalization.
//!
//! # Mathematical Specification
//! Given an image X with intensity range [xₘᵢₙ, xₘₐₓ]:
//!
//!   N(x) = (x − xₘᵢₙ) / (xₘₐₓ − xₘᵢₙ + ε),   ε = 1e-8
//!
//! This maps intensities to [0, 1].  An optional affine remap then applies:
//!
//!   R(x) = N(x) · (target_max − target_min) + target_min
//!
//! which maps to [target_min, target_max].
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the input.
//! - ε prevents division by zero on constant images.
//! - Default target range is [0.0, 1.0].

use crate::image::Image;
use crate::statistics::image_statistics::compute_statistics;
use burn::tensor::backend::Backend;

/// Min-max intensity normalizer.
///
/// Rescales image intensities to [0, 1] (default) or an arbitrary range
/// `[target_min, target_max]`.
pub struct MinMaxNormalizer {
    /// Lower bound of the output intensity range.
    pub target_min: f32,
    /// Upper bound of the output intensity range.
    pub target_max: f32,
}

impl MinMaxNormalizer {
    /// Create a normalizer that maps intensities to [0, 1].
    pub fn new() -> Self {
        Self {
            target_min: 0.0,
            target_max: 1.0,
        }
    }

    /// Create a normalizer that maps intensities to `[target_min, target_max]`.
    pub fn with_range(target_min: f32, target_max: f32) -> Self {
        Self {
            target_min,
            target_max,
        }
    }

    /// Normalize `image`.
    ///
    /// # Formula
    /// ```text
    /// normalized = (input − min) / (max − min + 1e-8)
    /// output     = normalized · (target_max − target_min) + target_min
    /// ```
    ///
    /// Spatial metadata is preserved exactly.
    pub fn normalize<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let stats = compute_statistics(image);
        let min = stats.min;
        let max = stats.max;
        let range = (max - min) as f32 + 1e-8_f32;

        // N(x) = (x − min) / (max − min + ε)
        let normalized = image.data().clone().sub_scalar(min).div_scalar(range);

        // R(x) = N(x) · (target_max − target_min) + target_min
        let output_range = self.target_max - self.target_min;
        let remapped = if (output_range - 1.0).abs() < 1e-9 && self.target_min.abs() < 1e-9 {
            // Default [0,1] case: skip the remap arithmetic entirely.
            normalized
        } else {
            normalized
                .mul_scalar(output_range)
                .add_scalar(self.target_min)
        };

        Image::new(
            remapped,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        )
    }
}

impl Default for MinMaxNormalizer {
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

    fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn get_values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Positive tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_minmax_known_values_unit_range() {
        // Values [0, 5, 10]: min=0, max=10, range=10
        //   N(0)  = 0.0
        //   N(5)  = 0.5
        //   N(10) = (10−0)/(10+1e-8) ≈ 1.0
        let image = make_image_1d(vec![0.0, 5.0, 10.0]);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        assert!(values[0].abs() < 1e-5, "N(0) ≈ 0.0, got {}", values[0]);
        assert!(
            (values[1] - 0.5).abs() < 1e-4,
            "N(5) ≈ 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 1.0).abs() < 1e-4,
            "N(10) ≈ 1.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_minmax_min_is_zero_after_normalization() {
        // Minimum value must map to 0 in [0,1] range.
        let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);

        let stats = crate::statistics::image_statistics::compute_statistics(&result);
        assert!(
            stats.min.abs() < 1e-5,
            "min after normalization must be ≈ 0.0, got {}",
            stats.min
        );
    }

    #[test]
    fn test_minmax_max_is_one_after_normalization() {
        // Maximum value must map to ≈ 1 in [0,1] range.
        let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);

        let stats = crate::statistics::image_statistics::compute_statistics(&result);
        assert!(
            (stats.max - 1.0).abs() < 1e-4,
            "max after normalization must be ≈ 1.0, got {}",
            stats.max
        );
    }

    #[test]
    fn test_minmax_ordering_preserved() {
        // Monotone input order must be preserved in the output.
        let data: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        for i in 0..7 {
            assert!(
                values[i] < values[i + 1],
                "ordering violated: values[{}]={} >= values[{}]={}",
                i,
                values[i],
                i + 1,
                values[i + 1]
            );
        }
    }

    #[test]
    fn test_minmax_with_custom_range() {
        // Values [0, 5, 10] normalized to [−1, 1]:
        //   N(0)  = 0.0  → −1.0 + 0   * 2 = −1.0
        //   N(5)  = 0.5  → −1.0 + 0.5 * 2 =  0.0
        //   N(10) ≈ 1.0  → −1.0 + 1   * 2 =  1.0
        let image = make_image_1d(vec![0.0, 5.0, 10.0]);
        let normalizer = MinMaxNormalizer::with_range(-1.0, 1.0);
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        assert!(
            (values[0] - (-1.0)).abs() < 1e-4,
            "R(0) ≈ -1.0, got {}",
            values[0]
        );
        assert!(values[1].abs() < 1e-4, "R(5) ≈ 0.0, got {}", values[1]);
        assert!(
            (values[2] - 1.0).abs() < 1e-4,
            "R(10) ≈ 1.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_minmax_with_positive_shift() {
        // Values [2, 4, 6] normalized to [100, 200]:
        //   N(2) = 0.0 → 100.0
        //   N(4) = 0.5 → 150.0
        //   N(6) ≈ 1.0 → 200.0
        let image = make_image_1d(vec![2.0, 4.0, 6.0]);
        let normalizer = MinMaxNormalizer::with_range(100.0, 200.0);
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        assert!(
            (values[0] - 100.0).abs() < 1e-3,
            "R(2) ≈ 100.0, got {}",
            values[0]
        );
        assert!(
            (values[1] - 150.0).abs() < 1e-3,
            "R(4) ≈ 150.0, got {}",
            values[1]
        );
        assert!(
            (values[2] - 200.0).abs() < 1e-3,
            "R(6) ≈ 200.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_minmax_negative_input_values() {
        // Values [−10, 0, 10]:
        //   range = 20,  N(−10)=0, N(0)=0.5, N(10)≈1
        let image = make_image_1d(vec![-10.0, 0.0, 10.0]);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        assert!(values[0].abs() < 1e-5, "N(−10) ≈ 0.0, got {}", values[0]);
        assert!(
            (values[1] - 0.5).abs() < 1e-4,
            "N(0) ≈ 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 1.0).abs() < 1e-4,
            "N(10) ≈ 1.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_minmax_preserves_metadata() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Shape::new([2, 2, 2]),
            ),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);

        assert_eq!(result.origin(), &origin, "origin must be preserved");
        assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
        assert_eq!(
            result.direction(),
            &direction,
            "direction must be preserved"
        );
        assert_eq!(result.shape(), [2, 2, 2], "shape must be preserved");
    }

    #[test]
    fn test_minmax_3d_shape_preserved() {
        let image = make_image_3d((0..27).map(|x| x as f32).collect(), [3, 3, 3]);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        assert_eq!(result.shape(), [3, 3, 3]);

        let values = get_values_3d(&result);
        // First element (min) maps to ≈ 0.
        assert!(
            values[0].abs() < 1e-5,
            "first element ≈ 0.0, got {}",
            values[0]
        );
        // Last element (max) maps to ≈ 1.
        let last = *values.last().unwrap();
        assert!(
            (last - 1.0).abs() < 1e-4,
            "last element ≈ 1.0, got {}",
            last
        );
    }

    // ── Boundary / edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_minmax_constant_image_does_not_panic() {
        // Constant image: range = 0. Division by (0 + 1e-8) must not panic.
        // N(c) = (c − c) / ε = 0.  Remapped: target_min + 0 * range = target_min.
        let image = make_image_1d(vec![7.0; 8]);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        for &v in &values {
            assert!(
                v.abs() < 1e-3,
                "constant image → all outputs ≈ 0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_minmax_constant_image_custom_range_maps_to_target_min() {
        // Constant image with custom range [5, 10]:
        //   N(c) = 0 → 5 + 0 * 5 = 5 = target_min.
        let image = make_image_1d(vec![7.0; 4]);
        let normalizer = MinMaxNormalizer::with_range(5.0, 10.0);
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        for &v in &values {
            assert!(
                (v - 5.0).abs() < 1e-3,
                "constant image with range [5,10] → all outputs ≈ 5.0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_minmax_single_voxel() {
        // Single voxel: range = 0, N = 0, output = target_min = 0.
        let image = make_image_1d(vec![42.0]);
        let normalizer = MinMaxNormalizer::new();
        let result = normalizer.normalize(&image);
        let values = get_values(&result);

        assert!(
            values[0].abs() < 1e-3,
            "single voxel → output ≈ 0.0, got {}",
            values[0]
        );
    }

    #[test]
    fn test_default_is_unit_range() {
        let n1 = MinMaxNormalizer::default();
        assert_eq!(n1.target_min, 0.0);
        assert_eq!(n1.target_max, 1.0);
    }
}

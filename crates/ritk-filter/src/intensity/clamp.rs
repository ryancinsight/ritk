//! Intensity clamp filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! output(x) = clamp(I(x), lower, upper)
//!           = max(lower, min(upper, I(x)))
//!
//! This is the pointwise projection of I onto the interval [lower, upper].
//!
//! ## Proof of idempotence
//!
//! Applying the clamp twice yields identical output because every voxel
//! already lies in [lower, upper] after the first application:
//!
//!   clamp(clamp(v, lo, hi), lo, hi) = clamp(v, lo, hi)
//!
//! ## Invariants
//!
//! - All output values satisfy `lower ≤ out(x) ≤ upper`.
//! - If all input values already lie in `[lower, upper]`, the output equals
//!   the input exactly.
//! - `lower > upper` is a logic error; the constructor panics.
//!
//! # ITK / SimpleITK Parity
//!
//! Corresponds to `itk::ClampImageFilter<TInputImage, TOutputImage>` with
//! `SetBounds(lower, upper)`.
//!
//! # Complexity
//!
//! O(N) time, O(N) space (output allocation), O(1) auxiliary.

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Clamp image intensity to the closed interval `[lower, upper]`.
///
/// Every voxel whose value lies below `lower` is raised to `lower`; every
/// voxel whose value exceeds `upper` is lowered to `upper`; all other voxels
/// are preserved unchanged.
///
/// # Invariants
///
/// - `lower ≤ upper` (enforced by constructor panic).
/// - All output values lie in `[lower, upper]`.
#[derive(Debug, Clone)]
pub struct ClampImageFilter {
    /// Inclusive lower bound for output intensity.
    pub lower: f32,
    /// Inclusive upper bound for output intensity.
    pub upper: f32,
}

impl ClampImageFilter {
    /// Create a `ClampImageFilter` with the given bounds.
    ///
    /// # Panics
    ///
    /// Panics if `lower > upper`.
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "ClampImageFilter: lower bound {lower} must be ≤ upper bound {upper}"
        );
        Self { lower, upper }
    }

    /// Apply the clamp to `image`.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let lo = self.lower;
        let hi = self.upper;
        let out: Vec<f32> = vals.iter().map(|&v| v.clamp(lo, hi)).collect();

        let out_td = TensorData::new(out, Shape::new(dims));
        let device = image.data().device();
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_tensor_ops::extract_vec_infallible;

    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    /// Constant field within bounds — output equals input.
    ///
    /// # Analytical basis
    /// All voxels = 50.0 lie in [0.0, 100.0] → clamp is identity.
    #[test]
    fn constant_within_bounds_is_identity() {
        let img = make_image(vec![50.0_f32; 27], [3, 3, 3]);
        let out = ClampImageFilter::new(0.0, 100.0).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        assert!(
            vals.iter().all(|&v| v == 50.0),
            "all voxels must remain 50.0; got {:?}",
            &vals[..5]
        );
    }

    /// All voxels below lower bound — output is all lower.
    ///
    /// # Analytical basis
    /// All voxels = -10.0 < 0.0 = lower → all outputs = 0.0.
    #[test]
    fn all_below_lower_clamped_to_lower() {
        let img = make_image(vec![-10.0_f32; 8], [2, 2, 2]);
        let out = ClampImageFilter::new(0.0, 255.0).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        assert!(
            vals.iter().all(|&v| v == 0.0),
            "all voxels below lower must be clamped to 0.0; got {:?}",
            vals
        );
    }

    /// All voxels above upper bound — output is all upper.
    ///
    /// # Analytical basis
    /// All voxels = 300.0 > 255.0 = upper → all outputs = 255.0.
    #[test]
    fn all_above_upper_clamped_to_upper() {
        let img = make_image(vec![300.0_f32; 8], [2, 2, 2]);
        let out = ClampImageFilter::new(0.0, 255.0).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        assert!(
            vals.iter().all(|&v| v == 255.0),
            "all voxels above upper must be clamped to 255.0; got {:?}",
            vals
        );
    }

    /// Mixed values — below, within, above.
    ///
    /// # Analytical basis
    /// Input: [-5, 50, 300], bounds [0, 100].
    /// Expected output: [0, 50, 100].
    #[test]
    fn mixed_values_clamped_correctly() {
        let img = make_image(vec![-5.0_f32, 50.0, 300.0], [1, 1, 3]);
        let out = ClampImageFilter::new(0.0, 100.0).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        assert_eq!(vals[0], 0.0, "below lower must be clamped to 0");
        assert_eq!(vals[1], 50.0, "within bounds must be unchanged");
        assert_eq!(vals[2], 100.0, "above upper must be clamped to 100");
    }

    /// Lower == upper — all voxels map to that single value.
    ///
    /// # Analytical basis
    /// Bounds [42.0, 42.0]: every voxel maps to 42.0 regardless of input.
    #[test]
    fn lower_equals_upper_all_same() {
        let img = make_image(vec![0.0, 42.0, 100.0, -100.0], [1, 2, 2]);
        let out = ClampImageFilter::new(42.0, 42.0).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        assert!(
            vals.iter().all(|&v| v == 42.0),
            "lower==upper: all outputs must be 42.0; got {:?}",
            vals
        );
    }

    /// Spatial metadata is preserved.
    ///
    /// # Analytical basis
    /// Clamp is a pointwise filter — origin, spacing, direction invariant.
    #[test]
    fn spatial_metadata_preserved() {
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let origin = Point::new([3.0_f64, 5.0, 7.0]);
        let spacing = Spacing::new([0.5_f64, 0.5, 1.0]);
        let direction = Direction::identity();
        let img = Image::new(tensor, origin, spacing, direction);
        let out = ClampImageFilter::new(0.0, 10.0).apply(&img).unwrap();
        assert_eq!(out.origin(), &origin);
        assert_eq!(out.spacing()[0], 0.5_f64, "spacing[0]");
        assert_eq!(out.spacing()[1], 0.5_f64, "spacing[1]");
        assert_eq!(out.spacing()[2], 1.0_f64, "spacing[2]");
    }

    /// Output shape matches input shape.
    #[test]
    fn output_shape_matches_input() {
        let img = make_image(vec![0.0_f32; 60], [3, 4, 5]);
        let out = ClampImageFilter::new(-1.0, 1.0).apply(&img).unwrap();
        assert_eq!(out.shape(), [3, 4, 5]);
    }

    /// All outputs satisfy lower ≤ v ≤ upper for random-like input.
    ///
    /// # Analytical basis
    /// Input range [-1000, 2000]; bounds [-500, 500].
    /// Every output must lie in [-500, 500].
    #[test]
    fn output_always_in_bounds() {
        let data: Vec<f32> = (-10..=10).map(|i| (i * 100) as f32).collect();
        let n = data.len();
        let img = make_image(data, [1, 1, n]);
        let lo = -500.0_f32;
        let hi = 500.0_f32;
        let out = ClampImageFilter::new(lo, hi).apply(&img).unwrap();
        let (vals, _) = extract_vec_infallible(&out);
        for &v in &vals {
            assert!(v >= lo, "voxel {v} < lower bound {lo}");
            assert!(v <= hi, "voxel {v} > upper bound {hi}");
        }
    }
}

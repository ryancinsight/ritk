//! Binary threshold segmentation with user-specified intensity bounds.
//!
//! # Mathematical Specification
//!
//! Unlike auto-threshold methods (Otsu, Li, Yen, etc.), `BinaryThreshold` applies
//! a user-specified closed interval \[lower, upper\] to classify voxels:
//!
//!   O(x) = inside_value   if lower ≤ I(x) ≤ upper
//!   O(x) = outside_value  otherwise
//!
//! This is the direct Rust equivalent of ITK's `BinaryThresholdImageFilter`.
//!
//! ## Special cases
//! - lower = f32::NEG_INFINITY: any value ≤ upper → inside.
//! - upper = f32::INFINITY:     any value ≥ lower → inside.
//! - lower = f32::NEG_INFINITY, upper = f32::INFINITY: all voxels → inside.
//!
//! ## Invariants
//! - lower ≤ upper (panics otherwise).
//! - inside_value and outside_value must be finite (panics otherwise).
//! - Spatial metadata preserved exactly.
//!
//! # References
//! - ITK `BinaryThresholdImageFilter` (www.itk.org/Doxygen/html/classitk_1_1BinaryThresholdImageFilter.html)

use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

// ── Public API ─────────────────────────────────────────────────────────────────

/// User-specified binary threshold segmentation.
///
/// Maps voxels in \[lower, upper\] to `inside_value` and all others to `outside_value`.
///
/// # Defaults
/// - `inside_value`  = 1.0
/// - `outside_value` = 0.0
/// - `lower`         = `f32::NEG_INFINITY` (no lower bound)
/// - `upper`         = `f32::INFINITY`     (no upper bound)
#[derive(Debug, Clone)]
pub struct BinaryThreshold {
    /// Inclusive lower intensity bound. Default `f32::NEG_INFINITY`.
    pub lower: f32,
    /// Inclusive upper intensity bound. Default `f32::INFINITY`.
    pub upper: f32,
    /// Output value for voxels inside \[lower, upper\]. Default 1.0.
    pub inside_value: f32,
    /// Output value for voxels outside \[lower, upper\]. Default 0.0.
    pub outside_value: f32,
}

impl BinaryThreshold {
    /// Create a `BinaryThreshold` with explicit bounds and default inside/outside values (1.0 / 0.0).
    ///
    /// # Panics
    /// Panics if `lower > upper`.
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "lower bound {lower} must be ≤ upper bound {upper}"
        );
        Self {
            lower,
            upper,
            inside_value: 1.0,
            outside_value: 0.0,
        }
    }

    /// Builder: set custom inside and outside values.
    ///
    /// # Panics
    /// Panics if either value is not finite.
    pub fn with_values(mut self, inside_value: f32, outside_value: f32) -> Self {
        assert!(
            inside_value.is_finite(),
            "inside_value must be finite, got {}",
            inside_value
        );
        assert!(
            outside_value.is_finite(),
            "outside_value must be finite, got {}",
            outside_value
        );
        self.inside_value = inside_value;
        self.outside_value = outside_value;
        self
    }

    /// Apply the binary threshold to `image`.
    ///
    /// Returns an image with the same shape and spatial metadata as `image`.
    /// Each voxel is set to `inside_value` or `outside_value` according to the
    /// threshold interval \[lower, upper\].
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        binary_threshold(
            image,
            self.lower,
            self.upper,
            self.inside_value,
            self.outside_value,
        )
    }
}

impl Default for BinaryThreshold {
    fn default() -> Self {
        Self {
            lower: f32::NEG_INFINITY,
            upper: f32::INFINITY,
            inside_value: 1.0,
            outside_value: 0.0,
        }
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Apply a user-specified binary threshold to `image`.
///
/// Returns an image with the same shape and spatial metadata as `image`.
/// Voxels in \[lower, upper\] → `inside_value`; all others → `outside_value`.
///
/// # Panics
/// Panics if `lower > upper` or if either value is not finite.
pub fn binary_threshold<B: Backend, const D: usize>(
    image: &Image<B, D>,
    lower: f32,
    upper: f32,
    inside_value: f32,
    outside_value: f32,
) -> Image<B, D> {
    assert!(
        lower <= upper,
        "lower bound {lower} must be ≤ upper bound {upper}"
    );
    assert!(
        inside_value.is_finite(),
        "inside_value must be finite, got {inside_value}"
    );
    assert!(
        outside_value.is_finite(),
        "outside_value must be finite, got {outside_value}"
    );

    let device = image.data().device();
    let shape: [usize; D] = image.shape();
    let (img_vals, _shape) = extract_vec_infallible(image);
    let slice: &[f32] = &img_vals;

    let output: Vec<f32> =
        apply_binary_threshold_to_slice(slice, lower, upper, inside_value, outside_value);

    let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
}

// ── Core implementation ───────────────────────────────────────────────────────

/// Apply binary threshold directly to a flat `&[f32]` slice.
///
/// Zero-copy variant: accepts pre-extracted slice.
pub fn apply_binary_threshold_to_slice(
    slice: &[f32],
    lower: f32,
    upper: f32,
    inside_value: f32,
    outside_value: f32,
) -> Vec<f32> {
    slice
        .iter()
        .map(|&v| {
            if v >= lower && v <= upper {
                inside_value
            } else {
                outside_value
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_threshold.rs"]
mod tests_binary_threshold;

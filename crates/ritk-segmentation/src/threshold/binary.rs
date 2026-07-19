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
//! - lower = f32::NEG_INFINITY, upper = f32::INFINITY: all non-NaN voxels → inside.
//! - NaN input voxels never satisfy the closed interval and map to `outside_value`.
//!
//! ## Invariants
//! - lower and upper are not NaN, and lower ≤ upper (panics otherwise).
//! - inside_value and outside_value must be finite (panics otherwise).
//! - Spatial metadata preserved exactly.
//!
//! # References
//! - ITK `BinaryThresholdImageFilter` (www.itk.org/Doxygen/html/classitk_1_1BinaryThresholdImageFilter.html)

use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    lower: f32,
    /// Inclusive upper intensity bound. Default `f32::INFINITY`.
    upper: f32,
    /// Output value for voxels inside \[lower, upper\]. Default 1.0.
    inside_value: f32,
    /// Output value for voxels outside \[lower, upper\]. Default 0.0.
    outside_value: f32,
}

impl BinaryThreshold {
    /// Create a `BinaryThreshold` with explicit bounds and default inside/outside values (1.0 / 0.0).
    ///
    /// # Panics
    /// Panics if either bound is NaN or if `lower > upper`.
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

    /// Return the inclusive lower intensity bound.
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Return the inclusive upper intensity bound.
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Return the output value assigned inside the threshold interval.
    pub fn inside_value(&self) -> f32 {
        self.inside_value
    }

    /// Return the output value assigned outside the threshold interval.
    pub fn outside_value(&self) -> f32 {
        self.outside_value
    }

    /// Apply the binary threshold to `image`.
    ///
    /// Returns an image with the same shape and spatial metadata as `image`.
    /// Each voxel is set to `inside_value` or `outside_value` according to the
    /// threshold interval \[lower, upper\].
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        binary_threshold(
            image,
            self.lower,
            self.upper,
            self.inside_value,
            self.outside_value,
        )
    }

    /// Apply the threshold to a Coeus-native CPU-addressable image.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend storage is not available as a
    /// contiguous host slice or when the output image cannot be constructed
    /// from the computed flat volume and input geometry.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        ritk_image::Image::from_flat_on(
            apply_binary_threshold_to_slice(
                image.data_slice()?,
                self.lower,
                self.upper,
                self.inside_value,
                self.outside_value,
            ),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
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

// â”€â”€ Public function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Apply a user-specified binary threshold to `image`.
///
/// Returns an image with the same shape and spatial metadata as `image`.
/// Voxels in \[lower, upper\] → `inside_value`; all others → `outside_value`.
///
/// # Panics
/// Panics if `lower > upper` or if either value is not finite.
pub fn binary_threshold<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    lower: f32,
    upper: f32,
    inside_value: f32,
    outside_value: f32,
) -> Image<f32, B, D> {
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

    let device = B::default();
    let shape: [usize; D] = image.shape();
    let (img_vals, _shape) = extract_vec_infallible(image);
    let slice: &[f32] = &img_vals;

    let output: Vec<f32> =
        apply_binary_threshold_to_slice(slice, lower, upper, inside_value, outside_value);

    let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);

    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
    .expect("invariant: segmentation output tensor preserves the image rank")
}

// â”€â”€ Core implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_binary_threshold.rs"]
mod tests_binary_threshold;

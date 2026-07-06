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
//! - All output values satisfy `lower â‰¤ out(x) â‰¤ upper`.
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

use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Clamp image intensity to the closed interval `[lower, upper]`.
///
/// Every voxel whose value lies below `lower` is raised to `lower`; every
/// voxel whose value exceeds `upper` is lowered to `upper`; all other voxels
/// are preserved unchanged.
///
/// # Invariants
///
/// - `lower â‰¤ upper` (enforced by constructor panic).
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
            "ClampImageFilter: lower bound {lower} must be â‰¤ upper bound {upper}"
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
#[path = "tests_clamp.rs"]
mod tests;

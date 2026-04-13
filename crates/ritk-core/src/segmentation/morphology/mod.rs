//! Binary morphological operations.
//!
//! Provides four standard binary morphological filters operating on foreground
//! masks (0.0 = background, 1.0 = foreground):
//!
//! - [`BinaryErosion`]:  output\[x\] = 1 iff ALL neighbors within radius are 1.
//! - [`BinaryDilation`]: output\[x\] = 1 iff ANY neighbor within radius is 1.
//! - [`BinaryOpening`]:  erosion followed by dilation (removes small foreground regions).
//! - [`BinaryClosing`]:  dilation followed by erosion (fills small background holes).
//!
//! All operations are implemented CPU-side via `.into_data()` and manual neighborhood
//! enumeration, supporting D=1, D=2, and D=3 images.

pub mod binary_closing;
pub mod binary_dilation;
pub mod binary_erosion;
pub mod binary_opening;

pub use binary_closing::BinaryClosing;
pub use binary_dilation::BinaryDilation;
pub use binary_erosion::BinaryErosion;
pub use binary_opening::BinaryOpening;

/// Trait for binary morphological operations.
///
/// Implementors transform a binary mask image (0.0/1.0) into a new binary
/// mask image of the same shape, preserving all spatial metadata.
pub trait MorphologicalOperation<B: burn::tensor::backend::Backend, const D: usize> {
    /// Apply the morphological operation to `mask`.
    ///
    /// # Arguments
    /// * `mask` – Binary mask image (values 0.0 for background, 1.0 for foreground).
    ///
    /// # Returns
    /// A new `Image<B, D>` with the same shape and spatial metadata as `mask`,
    /// containing the morphologically transformed binary mask.
    fn apply(&self, mask: &crate::image::Image<B, D>) -> crate::image::Image<B, D>;
}

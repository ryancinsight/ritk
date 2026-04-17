//! Binary morphological operations.
//!
//! Provides six binary morphological filters operating on foreground
//! masks (0.0 = background, 1.0 = foreground):
//!
//! - []:          output\[x\] = 1 iff ALL neighbors within radius are 1.
//! - []:         output\[x\] = 1 iff ANY neighbor within radius is 1.
//! - []:          erosion followed by dilation (removes small foreground regions).
//! - []:          dilation followed by erosion (fills small background holes).
//! - []:        fills all enclosed background holes via border flood-fill.
//! - []:  boundary extraction = dilation AND NOT erosion.
//!
//! All operations are implemented CPU-side via  and manual neighborhood
//! enumeration, supporting D=1, D=2, and D=3 images (BinaryFillHoles and
//! MorphologicalGradient support D=3 only).

pub mod binary_closing;
pub mod binary_dilation;
pub mod binary_erosion;
pub mod binary_opening;
pub mod fill_holes;
pub mod morphological_gradient;
pub mod skeletonization;

pub use binary_closing::BinaryClosing;
pub use binary_dilation::BinaryDilation;
pub use binary_erosion::BinaryErosion;
pub use binary_opening::BinaryOpening;
pub use fill_holes::BinaryFillHoles;
pub use morphological_gradient::MorphologicalGradient;
pub use skeletonization::Skeletonization;

/// Trait for binary morphological operations.
///
/// Implementors transform a binary mask image (0.0/1.0) into a new binary
/// mask image of the same shape, preserving all spatial metadata.
pub trait MorphologicalOperation<B: burn::tensor::backend::Backend, const D: usize> {
    /// Apply the morphological operation to .
    ///
    /// # Arguments
    /// *  - Binary mask image (values 0.0 for background, 1.0 for foreground).
    ///
    /// # Returns
    /// A new  with the same shape and spatial metadata as ,
    /// containing the morphologically transformed binary mask.
    fn apply(&self, mask: &crate::image::Image<B, D>) -> crate::image::Image<B, D>;
}

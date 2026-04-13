//! Binary morphological opening: erosion followed by dilation.
//!
//! # Mathematical Specification
//!
//! Opening removes small foreground objects and smooths boundaries.
//! For a binary mask M and structuring element B with radius r:
//!
//!   Opening(M) = Dilation(Erosion(M, B), B)
//!
//! Invariant: Opening(M) ⊆ M (opening can only remove foreground, never add).

use super::MorphologicalOperation;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Binary morphological opening filter.
///
/// Applies erosion followed by dilation with a hypercubic structuring
/// element of the given radius, removing small foreground regions and
/// smoothing object boundaries.
pub struct BinaryOpening {
    /// Half-width of the hypercubic structuring element.
    pub radius: usize,
}

impl BinaryOpening {
    /// Create a `BinaryOpening` with the given structuring element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl Default for BinaryOpening {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl<B: Backend, const D: usize> MorphologicalOperation<B, D> for BinaryOpening {
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        use super::binary_dilation::BinaryDilation;
        use super::binary_erosion::BinaryErosion;
        let eroded = BinaryErosion {
            radius: self.radius,
        }
        .apply(mask);
        BinaryDilation {
            radius: self.radius,
        }
        .apply(&eroded)
    }
}

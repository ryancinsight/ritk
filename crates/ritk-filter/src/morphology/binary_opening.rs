//! Binary morphological opening filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary opening with structuring element B:
//!
//!   O_B(f) = D_B(E_B(f))
//!
//! i.e. erosion followed by dilation with the same structuring element.
//!
//! # Properties
//!
//! - **Anti-extensivity**: `O_B(f) ≤ f` — opening does not add foreground voxels.
//! - **Spike removal**: removes bright blobs / protrusions smaller than B.
//! - **Idempotence**: `O_B(O_B(f)) = O_B(f)`.
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryMorphologicalOpeningImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetBackgroundValue(0.0)`
//! - Flat ball structuring element of radius r.
//!
//! # Complexity
//!
//! O(2 · N · (2r + 1)³): one erosion pass + one dilation pass.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::binary_dilate::dilate_binary_3d;
use super::binary_erode::erode_binary_3d;
use super::types::ForegroundValue;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary morphological opening filter for 3-D images.
///
/// Applies erosion then dilation with a flat cubic structuring element of
/// half-width `radius`.  Removes small foreground protrusions / noise blobs
/// without significantly altering larger connected regions.
#[derive(Debug, Clone)]
pub struct BinaryMorphologicalOpening {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryMorphologicalOpening {
    /// Create an opening filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary opening to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        // Opening = dilate(erode(f))
        let eroded = erode_binary_3d(&vals, dims, self.radius, self.foreground_value);
        let result = dilate_binary_3d(&eroded, dims, self.radius, self.foreground_value);

        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for BinaryMorphologicalOpening {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
#[path = "tests_binary_opening.rs"]
mod tests_binary_opening;

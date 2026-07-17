//! Binary morphological closing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary closing with structuring element B:
//!
//!   C_B(f) = E_B(D_B(f))
//!
//! i.e. dilation followed by erosion with the same structuring element.
//!
//! # Properties
//!
//! - **Extensivity**: `C_B(f) ≥ f` — closing does not remove foreground voxels.
//! - **Hole filling**: removes dark cavities / holes smaller than B.
//! - **Idempotence**: `C_B(C_B(f)) = C_B(f)`.
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryMorphologicalClosingImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetSafeBorder(false)` (no extra safe border added)
//! - Flat ball structuring element of radius r.
//!
//! # Complexity
//!
//! O(2 · N · (2r + 1)³): one dilation pass + one erosion pass.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::binary_dilate::dilate_binary_3d;
use super::binary_erode::erode_binary_3d;
use super::types::ForegroundValue;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary morphological closing filter for 3-D images.
///
/// Applies dilation then erosion with a flat cubic structuring element of
/// half-width `radius`.  Removes dark holes / cavities smaller than the SE.
#[derive(Debug, Clone)]
pub struct BinaryMorphologicalClosing {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryMorphologicalClosing {
    /// Create a closing filter with `radius` and default `foreground_value = 1.0`.
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

    /// Apply binary closing to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        // Closing = erode(dilate(f))
        let dilated = dilate_binary_3d(&vals, dims, self.radius, self.foreground_value);
        let result = erode_binary_3d(&dilated, dims, self.radius, self.foreground_value);

        Ok(rebuild(result, dims, image))
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;

        // Closing = erode(dilate(f))
        let dilated = dilate_binary_3d(&vals, dims, self.radius, self.foreground_value);
        let result = erode_binary_3d(&dilated, dims, self.radius, self.foreground_value);

        crate::native_support::rebuild_image(result, dims, image, backend)
    }
}

impl Default for BinaryMorphologicalClosing {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
#[path = "tests_binary_closing.rs"]
mod tests_binary_closing;

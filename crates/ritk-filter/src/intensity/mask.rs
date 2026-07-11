//! Mask image filters for selective voxel zeroing.
//!
//! # Mathematical Specification
//!
//! Let `I : â„¤Â³ â†’ â„` be the input image and `M : â„¤Â³ â†’ â„` the mask image.
//!
//! **`MaskImageFilter`** (inside mask):
//! `out(x) = I(x)` if `M(x) > threshold`, else `outside_value`
//!
//! **`MaskNegatedImageFilter`** (outside mask):
//! `out(x) = I(x)` if `M(x) â‰¤ threshold`, else `outside_value`
//!
//! Default `threshold = 0.5`, `outside_value = 0.0`.
//! Spatial metadata (origin, spacing, direction) is taken from the input image `I`.
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                      | ITK class                    |
//! |-----------------------------|------------------------------|
//! | `MaskImageFilter`           | `MaskImageFilter`            |
//! | `MaskNegatedImageFilter`    | `MaskNegatedImageFilter`     |

use crate::distance::types::BinarizationThreshold;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

fn check_shapes(a: [usize; 3], b: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        a == b,
        "mask filter: shape mismatch image {:?} vs mask {:?}",
        a,
        b
    );
    Ok(())
}

// â”€â”€ MaskImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Retain image values where the mask is active (> threshold); replace elsewhere.
///
/// `out(x) = image(x)` if `mask(x) > threshold`, else `outside_value`
///
/// # ITK Parity: `MaskImageFilter`
#[derive(Debug, Clone)]
pub struct MaskImageFilter {
    /// Foreground threshold for the mask image. Default: 0.5.
    pub threshold: BinarizationThreshold,
    /// Value written where the mask is inactive. Default: 0.0.
    pub outside_value: f32,
}

impl Default for MaskImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            outside_value: 0.0,
        }
    }
}

impl MaskImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: impl Into<BinarizationThreshold>) -> Self {
        self.threshold = t.into();
        self
    }

    pub fn with_outside_value(mut self, v: f32) -> Self {
        self.outside_value = v;
        self
    }

    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        check_shapes(dims, mask.shape())?;
        let (iv, _) = extract_vec(image)?;
        let (mv, _) = extract_vec(mask)?;
        let outside = self.outside_value;
        let thr = f32::from(self.threshold);
        let out: Vec<f32> = iv
            .iter()
            .zip(mv.iter())
            .map(|(&img_val, &mask_val)| if mask_val > thr { img_val } else { outside })
            .collect();
        Ok(rebuild(out, dims, image))
    }

    /// Apply threshold-derived masking to a Coeus-native image.
    ///
    /// Voxels greater than `threshold` are retained; all others become zero.
    pub fn apply_threshold_native<B>(
        image: &ritk_image::native::Image<f32, B, 3>,
        threshold: BinarizationThreshold,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        let threshold = f32::from(threshold);
        let output = values
            .iter()
            .map(|&value| if value > threshold { value } else { 0.0 })
            .collect();
        ritk_image::native::Image::from_flat_on(
            output,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

// â”€â”€ MaskNegatedImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Retain image values where the mask is **inactive** (â‰¤ threshold); replace elsewhere.
///
/// `out(x) = image(x)` if `mask(x) â‰¤ threshold`, else `outside_value`
///
/// # ITK Parity: `MaskNegatedImageFilter`
#[derive(Debug, Clone)]
pub struct MaskNegatedImageFilter {
    /// Foreground threshold for the mask image. Default: 0.5.
    pub threshold: BinarizationThreshold,
    /// Value written where the mask is active. Default: 0.0.
    pub outside_value: f32,
}

impl Default for MaskNegatedImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            outside_value: 0.0,
        }
    }
}

impl MaskNegatedImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: impl Into<BinarizationThreshold>) -> Self {
        self.threshold = t.into();
        self
    }

    pub fn with_outside_value(mut self, v: f32) -> Self {
        self.outside_value = v;
        self
    }

    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        check_shapes(dims, mask.shape())?;
        let (iv, _) = extract_vec(image)?;
        let (mv, _) = extract_vec(mask)?;
        let outside = self.outside_value;
        let thr = f32::from(self.threshold);
        let out: Vec<f32> = iv
            .iter()
            .zip(mv.iter())
            .map(|(&img_val, &mask_val)| if mask_val <= thr { img_val } else { outside })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Assign `assign_value` where the mask is **active** (> threshold); keep the
/// image elsewhere — the role-inverse of [`MaskImageFilter`].
///
/// `out(x) = assign_value` if `mask(x) > threshold`, else `image(x)`
///
/// # ITK Parity: `MaskedAssignImageFilter` (`sitk.MaskedAssign`, constant form)
#[derive(Debug, Clone)]
pub struct MaskedAssignImageFilter {
    /// Foreground threshold for the mask image. Default: 0.5.
    pub threshold: BinarizationThreshold,
    /// Value written where the mask is active. Default: 0.0.
    pub assign_value: f32,
}

impl Default for MaskedAssignImageFilter {
    fn default() -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            assign_value: 0.0,
        }
    }
}

impl MaskedAssignImageFilter {
    /// Construct with the assign value (default threshold 0.5).
    pub fn new(assign_value: f32) -> Self {
        Self {
            threshold: BinarizationThreshold::DEFAULT,
            assign_value,
        }
    }

    /// Apply: write `assign_value` where `mask > threshold`, else keep `image`.
    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        check_shapes(dims, mask.shape())?;
        let (iv, _) = extract_vec(image)?;
        let (mv, _) = extract_vec(mask)?;
        let assign = self.assign_value;
        let thr = f32::from(self.threshold);
        let out: Vec<f32> = iv
            .iter()
            .zip(mv.iter())
            .map(|(&img_val, &mask_val)| if mask_val > thr { assign } else { img_val })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_mask.rs"]
mod tests;

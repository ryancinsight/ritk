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

/// Output source for one branch of a mask decision.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MaskFill {
    /// Copy the corresponding image voxel through unchanged.
    Keep,
    /// Write a constant value.
    Constant(f32),
}

/// Substrate-agnostic host core shared by all three mask filters.
///
/// For each voxel, selects `on_active` when `mask(x) > threshold`, else
/// `on_inactive`. Encoding every family as a single predicate (`mask > thr`)
/// with a per-branch [`MaskFill`] collapses the three previously-identical zips
/// to one entry point:
/// - [`MaskImageFilter`]: active → `Keep`, inactive → `Constant(outside)`.
/// - [`MaskNegatedImageFilter`]: active → `Constant(outside)`, inactive → `Keep`.
/// - [`MaskedAssignImageFilter`]: active → `Constant(assign)`, inactive → `Keep`.
pub(crate) fn mask_combine(
    image: &[f32],
    mask: &[f32],
    threshold: f32,
    on_active: MaskFill,
    on_inactive: MaskFill,
) -> Vec<f32> {
    image
        .iter()
        .zip(mask.iter())
        .map(|(&img_val, &mask_val)| {
            let fill = if mask_val > threshold {
                on_active
            } else {
                on_inactive
            };
            match fill {
                MaskFill::Keep => img_val,
                MaskFill::Constant(c) => c,
            }
        })
        .collect()
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
        let out = mask_combine(
            &iv,
            &mv,
            f32::from(self.threshold),
            MaskFill::Keep,
            MaskFill::Constant(self.outside_value),
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`MaskImageFilter::apply`].
    ///
    /// Runs the identical mask selection via the shared `mask_combine` host
    /// core on both images' contiguous host buffers, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    ///
    /// # Errors
    /// Returns an error on shape mismatch, non-contiguous buffers, or failed
    /// shape validation of the rebuilt image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        mask: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let thr = f32::from(self.threshold);
        let outside = self.outside_value;
        crate::native_support::map_flat_pair(image, mask, backend, |iv, mv, _dims| {
            mask_combine(iv, mv, thr, MaskFill::Keep, MaskFill::Constant(outside))
        })
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
        let out = mask_combine(
            &iv,
            &mv,
            f32::from(self.threshold),
            MaskFill::Constant(self.outside_value),
            MaskFill::Keep,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`MaskNegatedImageFilter::apply`].
    ///
    /// Runs the identical negated mask selection via the shared `mask_combine`
    /// host core, bitwise-identical to the Burn path. No Burn tensor is
    /// constructed.
    ///
    /// # Errors
    /// Returns an error on shape mismatch, non-contiguous buffers, or failed
    /// shape validation of the rebuilt image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        mask: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let thr = f32::from(self.threshold);
        let outside = self.outside_value;
        crate::native_support::map_flat_pair(image, mask, backend, |iv, mv, _dims| {
            mask_combine(iv, mv, thr, MaskFill::Constant(outside), MaskFill::Keep)
        })
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
        let out = mask_combine(
            &iv,
            &mv,
            f32::from(self.threshold),
            MaskFill::Constant(self.assign_value),
            MaskFill::Keep,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`MaskedAssignImageFilter::apply`].
    ///
    /// Runs the identical masked-assign via the shared `mask_combine` host
    /// core, bitwise-identical to the Burn path. No Burn tensor is constructed.
    ///
    /// # Errors
    /// Returns an error on shape mismatch, non-contiguous buffers, or failed
    /// shape validation of the rebuilt image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        mask: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let thr = f32::from(self.threshold);
        let assign = self.assign_value;
        crate::native_support::map_flat_pair(image, mask, backend, |iv, mv, _dims| {
            mask_combine(iv, mv, thr, MaskFill::Constant(assign), MaskFill::Keep)
        })
    }
}

#[cfg(test)]
#[path = "tests_mask.rs"]
mod tests;

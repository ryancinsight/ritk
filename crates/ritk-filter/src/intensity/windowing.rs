//! Intensity windowing filter (clamp-then-rescale).
//!
//! # Mathematical Specification
//!
//! Let f(x) = clamp(I(x), window_min, window_max).
//! If window_min == window_max: output(x) = out_min.
//! Else: output(x) = (f(x) - window_min) / (window_max - window_min) * (out_max - out_min) + out_min
//!
//! Pixels below window_min map to out_min; pixels above window_max map to out_max.
//! Interior pixels are mapped linearly.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Clamp input to [window_min, window_max], then rescale to [out_min, out_max].
#[derive(Debug, Clone)]
pub struct IntensityWindowingFilter {
    /// Lower bound of the intensity window.
    pub window_min: f32,
    /// Upper bound of the intensity window.
    pub window_max: f32,
    /// Minimum output value (maps from window_min).
    pub out_min: f32,
    /// Maximum output value (maps from window_max).
    pub out_max: f32,
}

impl IntensityWindowingFilter {
    /// Construct with explicit window and output ranges.
    pub fn new(window_min: f32, window_max: f32, out_min: f32, out_max: f32) -> Self {
        Self {
            window_min,
            window_max,
            out_min,
            out_max,
        }
    }

    /// Apply windowing to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = window_vec(
            &vals,
            self.window_min,
            self.window_max,
            self.out_min,
            self.out_max,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`IntensityWindowingFilter::apply`].
    ///
    /// Runs the identical clamp-then-rescale via the shared `window_vec` host
    /// core on the image's contiguous host buffer, so the result is
    /// bitwise-identical to the Coeus path. No Coeus tensor is constructed.
    /// Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            window_vec(
                vals,
                self.window_min,
                self.window_max,
                self.out_min,
                self.out_max,
            )
        })
    }
}

/// Substrate-agnostic host core for [`IntensityWindowingFilter`].
///
/// Clamps each voxel to `[window_min, window_max]`, then affinely remaps
/// `[window_min, window_max]` to `[out_min, out_max]`. A degenerate window
/// (`window_max == window_min` within `f32::EPSILON`) collapses every voxel to
/// `out_min`, matching the mathematical specification.
pub(crate) fn window_vec(
    vals: &[f32],
    window_min: f32,
    window_max: f32,
    out_min: f32,
    out_max: f32,
) -> Vec<f32> {
    if (window_max - window_min).abs() < f32::EPSILON {
        vec![out_min; vals.len()]
    } else {
        let scale = (out_max - out_min) / (window_max - window_min);
        vals.iter()
            .map(|&v| {
                let clamped = v.max(window_min).min(window_max);
                (clamped - window_min) * scale + out_min
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "tests_windowing.rs"]
mod tests;

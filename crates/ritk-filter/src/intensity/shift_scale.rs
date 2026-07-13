//! Shift-scale intensity filter.
//!
//! # Mathematical Specification
//!
//! For each voxel x in image I:
//!
//! `out(x) = (I(x) + shift) * scale`
//!
//! The operation is applied in f64 precision and cast back to f32.
//!
//! ## Invariants
//!
//! - Spatial metadata (shape, origin, spacing, direction) is preserved exactly.
//! - When `shift = 0` and `scale = 1`, `out = I` (identity).
//! - When `scale = 0`, `out = 0` everywhere regardless of shift.
//! - The transform is linear: `f(a + b) = f(a) + scale * b`.
//!
//! # ITK Parity
//!
//! `itk::ShiftScaleImageFilter` with `SetShift(s)` and `SetScale(k)`.
//! Output type defaults to f32 (matching ITK behaviour when input is float).

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Apply a linear shift-then-scale to every voxel.
///
/// `out(x) = (in(x) + shift) * scale`
///
/// # Example
///
/// ```no_run
/// # use ritk_filter::ShiftScaleImageFilter;
/// let filter = ShiftScaleImageFilter::new(-1024.0, 0.001);
/// // Converts Hounsfield units centred at â€“1024 to linear attenuation values
/// ```
#[derive(Debug, Clone)]
pub struct ShiftScaleImageFilter {
    /// Added to each voxel value before multiplication.
    pub shift: f32,
    /// Scale factor applied after shift.
    pub scale: f32,
}

impl Default for ShiftScaleImageFilter {
    fn default() -> Self {
        Self {
            shift: 0.0,
            scale: 1.0,
        }
    }
}

impl ShiftScaleImageFilter {
    /// Create a new filter with the given shift and scale.
    pub fn new(shift: f32, scale: f32) -> Self {
        Self { shift, scale }
    }

    /// Set shift value.
    pub fn with_shift(mut self, s: f32) -> Self {
        self.shift = s;
        self
    }

    /// Set scale value.
    pub fn with_scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }

    /// Apply the filter to a 3-D image.
    ///
    /// Returns a new `Image` with identical spatial metadata and
    /// voxel values transformed by `(v + shift) * scale`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let out_vals = shift_scale_vec(&vals_vec, self.shift, self.scale);
        Ok(rebuild(out_vals, dims, image))
    }

    /// Coeus-native sister of [`ShiftScaleImageFilter::apply`].
    ///
    /// Runs the identical `(v + shift) * scale` remap (computed in `f64`, cast to
    /// `f32`) via the shared [`shift_scale_vec`] host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            shift_scale_vec(vals, self.shift, self.scale)
        })
    }
}

/// Substrate-agnostic host core for [`ShiftScaleImageFilter`].
///
/// Applies `out = (v + shift) * scale` in `f64` precision then narrows to `f32`,
/// matching ITK's float-output behaviour.
pub(crate) fn shift_scale_vec(vals: &[f32], shift: f32, scale: f32) -> Vec<f32> {
    let shift = shift as f64;
    let scale = scale as f64;
    vals.iter()
        .map(|&v| ((v as f64 + shift) * scale) as f32)
        .collect()
}

#[cfg(test)]
#[path = "tests_shift_scale.rs"]
mod tests;

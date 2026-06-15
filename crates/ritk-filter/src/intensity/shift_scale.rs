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

use burn::tensor::backend::Backend;
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
        let vals = &vals_vec;

        let shift = self.shift as f64;
        let scale = self.scale as f64;

        let out_vals: Vec<f32> = vals
            .iter()
            .map(|&v| ((v as f64 + shift) * scale) as f32)
            .collect();

        Ok(rebuild(out_vals, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_shift_scale.rs"]
mod tests;

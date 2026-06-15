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

use burn::tensor::backend::Backend;
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let wmin = self.window_min;
        let wmax = self.window_max;
        let omin = self.out_min;
        let omax = self.out_max;

        let out: Vec<f32> = if (wmax - wmin).abs() < f32::EPSILON {
            vec![omin; vals.len()]
        } else {
            let scale = (omax - omin) / (wmax - wmin);
            vals.iter()
                .map(|&v| {
                    let clamped = v.max(wmin).min(wmax);
                    (clamped - wmin) * scale + omin
                })
                .collect()
        };

        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_windowing.rs"]
mod tests;

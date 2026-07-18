//! Sigmoid intensity transform filter.
//!
//! # Mathematical Specification
//!
//! output(x) = (max_output - min_output) / (1 + exp(-(I(x) - alpha) / beta)) + min_output
//!
//! Special case: if |beta| < 1e-12, use step function:
//!   output(x) = if I(x) >= alpha { max_output } else { min_output }
//!
//! At I(x) = alpha:        output = (max_output + min_output) / 2
//! At I(x) = alpha + beta: output = (max_output - min_output) / (1 + exp(-1)) + min_output
//!
//! Reference: Sethian (1996). The output is strictly bounded in (min_output, max_output)
//! for finite input and nonzero beta.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Pixel-wise sigmoid intensity transform.
///
/// Maps I(x) to (min_output, max_output) via the sigmoid function.
#[derive(Debug, Clone)]
pub struct SigmoidImageFilter {
    /// Inflection point (input value at which output = (max + min) / 2).
    pub alpha: f32,
    /// Width of the transition region.
    pub beta: f32,
    /// Minimum output intensity.
    pub min_output: f32,
    /// Maximum output intensity.
    pub max_output: f32,
}

impl SigmoidImageFilter {
    pub fn new(alpha: f32, beta: f32, min_output: f32, max_output: f32) -> Self {
        Self {
            alpha,
            beta,
            min_output,
            max_output,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = sigmoid_vec(
            &vals,
            self.alpha,
            self.beta,
            self.min_output,
            self.max_output,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`SigmoidImageFilter::apply`].
    ///
    /// Runs the identical per-voxel sigmoid transform via the shared
    /// `sigmoid_vec` host core on the image's contiguous host buffer, so the
    /// result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata (origin, spacing, direction) is preserved.
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
            sigmoid_vec(
                vals,
                self.alpha,
                self.beta,
                self.min_output,
                self.max_output,
            )
        })
    }
}

/// Substrate-agnostic host core for [`SigmoidImageFilter`].
///
/// Maps each voxel through the logistic transform bounded in
/// `(min_output, max_output)`. A degenerate width (`|beta| < 1e-12`) degrades
/// to the Heaviside step at `alpha`, matching the mathematical specification.
pub(crate) fn sigmoid_vec(
    vals: &[f32],
    alpha: f32,
    beta: f32,
    min_output: f32,
    max_output: f32,
) -> Vec<f32> {
    let range = max_output - min_output;
    if beta.abs() < 1e-12 {
        vals.iter()
            .map(|&v| if v >= alpha { max_output } else { min_output })
            .collect()
    } else {
        vals.iter()
            .map(|&v| range / (1.0 + (-(v - alpha) / beta).exp()) + min_output)
            .collect()
    }
}

#[cfg(test)]
#[path = "tests_sigmoid.rs"]
mod tests;

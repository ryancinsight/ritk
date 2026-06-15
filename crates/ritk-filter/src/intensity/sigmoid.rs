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

use burn::tensor::backend::Backend;
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

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let alpha = self.alpha;
        let beta = self.beta;
        let min_o = self.min_output;
        let max_o = self.max_output;
        let range = max_o - min_o;

        let out: Vec<f32> = if beta.abs() < 1e-12 {
            vals.iter()
                .map(|&v| if v >= alpha { max_o } else { min_o })
                .collect()
        } else {
            vals.iter()
                .map(|&v| range / (1.0 + (-(v - alpha) / beta).exp()) + min_o)
                .collect()
        };

        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_sigmoid.rs"]
mod tests;

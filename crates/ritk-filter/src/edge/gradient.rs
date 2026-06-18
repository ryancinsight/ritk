//! Image gradient filter (central differences → covariant vector field).
//!
//! Matches ITK `GradientImageFilter` / `sitk.Gradient`: emits a 3-component
//! vector image whose component `k` is the first central-difference derivative
//! along physical axis `k` in **sitk axis order** `(x, y, z)`. Each component is
//! exactly [`DerivativeImageFilter`] of order 1, so spacing handling
//! (`∂/∂x_k = (f[i+1] − f[i−1]) / (2·spacing_k)`) and zero-flux Neumann boundary
//! handling are inherited unchanged.
//!
//! Component → ritk tensor axis mapping (tensor order is `[z, y, x]`):
//! - component 0 (∂/∂x) ← axis 2
//! - component 1 (∂/∂y) ← axis 1
//! - component 2 (∂/∂z) ← axis 0

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::{ColorVolume, Image};
use ritk_tensor_ops::extract_vec;

use super::derivative::DerivativeImageFilter;

/// Image gradient filter producing a 3-component covariant vector field.
#[derive(Debug, Clone, Copy)]
pub struct GradientImageFilter {
    /// Divide each component by its axis spacing for a physical-unit gradient
    /// (ITK default `true`).
    pub use_image_spacing: bool,
}

impl GradientImageFilter {
    /// Construct a gradient filter.
    pub fn new(use_image_spacing: bool) -> Self {
        Self { use_image_spacing }
    }

    /// Apply the gradient, returning a 3-component vector image with components
    /// in sitk axis order `(∂/∂x, ∂/∂y, ∂/∂z)`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<ColorVolume<B, 3>> {
        // Component k (sitk axis k) is the derivative along ritk axis 2 − k.
        let dx = DerivativeImageFilter::new(2, 1, self.use_image_spacing).apply(image)?;
        let dy = DerivativeImageFilter::new(1, 1, self.use_image_spacing).apply(image)?;
        let dz = DerivativeImageFilter::new(0, 1, self.use_image_spacing).apply(image)?;

        let (bx, dims) = extract_vec(&dx)?;
        let (by, _) = extract_vec(&dy)?;
        let (bz, _) = extract_vec(&dz)?;

        ColorVolume::<B, 3>::from_component_buffers(
            &[bx, by, bz],
            dims,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            &image.data().device(),
        )
    }
}

#[cfg(test)]
#[path = "tests_gradient.rs"]
mod tests_gradient;

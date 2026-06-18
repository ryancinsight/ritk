//! Zero-crossing-based edge detection for 3-D images.
//!
//! # Mathematical Specification
//!
//! Ports `itk::ZeroCrossingBasedEdgeDetectionImageFilter`, a three-stage
//! mini-pipeline:
//!
//! 1. **Discrete Gaussian** smoothing with per-axis `variance` (physical units,
//!    `UseImageSpacing`) and `maximum_error` truncation.
//! 2. **Laplacian** `∇²` of the smoothed image (ZeroFluxNeumann boundary,
//!    divided by `spacing²` per axis).
//! 3. **Zero crossing** detection on the Laplacian, marking edge voxels with
//!    `foreground_value` and the rest with `background_value`.
//!
//! Each stage is the canonical ritk filter that is already float-exact to its
//! SimpleITK counterpart (`DiscreteGaussian`, `Laplacian`, `ZeroCrossing`), so
//! the composition matches `sitk.ZeroCrossingBasedEdgeDetection`.
//!
//! # ITK parity
//!
//! Defaults match ITK: `variance = 1.0`, `maximum_error = 0.01`,
//! `foreground_value = 1.0`, `background_value = 0.0`.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use crate::discrete_gaussian::DiscreteGaussianFilter;
use crate::edge::laplacian::LaplacianFilter;
use crate::intensity::zero_crossing::ZeroCrossingImageFilter;

/// Zero-crossing-based edge detector (`itk::ZeroCrossingBasedEdgeDetectionImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct ZeroCrossingBasedEdgeDetectionFilter {
    /// Isotropic Gaussian variance in physical units (ITK default `1.0`).
    pub variance: f64,
    /// Gaussian kernel truncation error (ITK default `0.01`).
    pub maximum_error: f64,
    /// Value marking zero-crossing (edge) voxels (ITK default `1.0`).
    pub foreground_value: f32,
    /// Value marking non-edge voxels (ITK default `0.0`).
    pub background_value: f32,
}

impl Default for ZeroCrossingBasedEdgeDetectionFilter {
    fn default() -> Self {
        Self {
            variance: 1.0,
            maximum_error: 0.01,
            foreground_value: 1.0,
            background_value: 0.0,
        }
    }
}

impl ZeroCrossingBasedEdgeDetectionFilter {
    /// Construct with explicit parameters.
    pub fn new(
        variance: f64,
        maximum_error: f64,
        foreground_value: f32,
        background_value: f32,
    ) -> Self {
        Self {
            variance,
            maximum_error,
            foreground_value,
            background_value,
        }
    }

    /// Run the Gaussian → Laplacian → zero-crossing mini-pipeline.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let smoothed = DiscreteGaussianFilter::<B>::new_isotropic(self.variance)
            .with_maximum_error(self.maximum_error)
            .apply(image);
        let laplacian = LaplacianFilter::new(*image.spacing()).apply(&smoothed)?;
        ZeroCrossingImageFilter::new()
            .with_foreground(self.foreground_value)
            .with_background(self.background_value)
            .apply(&laplacian)
    }
}

#[cfg(test)]
#[path = "tests_zero_crossing_based_edge.rs"]
mod tests_zero_crossing_based_edge;

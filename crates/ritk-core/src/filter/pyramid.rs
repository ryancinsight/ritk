use burn::tensor::backend::Backend;
use crate::image::Image;
use super::gaussian::GaussianFilter;
use super::downsample::DownsampleFilter;

/// Multi-resolution image pyramid.
///
/// Generates a sequence of images at different resolutions/scales.
/// Typically used for coarse-to-fine registration strategies.
pub struct MultiResolutionPyramid<B: Backend, const D: usize> {
    images: Vec<Image<B, D>>,
}

impl<B: Backend, const D: usize> MultiResolutionPyramid<B, D> {
    /// Create a pyramid from an input image and schedules.
    ///
    /// # Arguments
    /// * `input` - The original high-resolution image.
    /// * `shrink_factors` - Shrink factors for each level [level][dim].
    /// * `smoothing_sigmas` - Smoothing sigmas for each level [level][dim].
    ///
    /// # Panics
    /// Panics if schedules have different lengths.
    pub fn new(
        input: &Image<B, D>,
        shrink_factors: &[Vec<usize>],
        smoothing_sigmas: &[Vec<f64>],
    ) -> Self {
        assert_eq!(shrink_factors.len(), smoothing_sigmas.len(), "Schedule lengths must match");
        
        let mut images = Vec::with_capacity(shrink_factors.len());
        
        for (factors, sigmas) in shrink_factors.iter().zip(smoothing_sigmas.iter()) {
            // Optimization: if identity transform, just clone
            let is_identity_shrink = factors.iter().all(|&f| f == 1);
            let is_identity_smooth = sigmas.iter().all(|&s| s <= 1e-6);
            
            if is_identity_shrink && is_identity_smooth {
                images.push(input.clone());
                continue;
            }

            // 1. Smooth
            // Only smooth if sigmas are significant
            let smoothed = if !is_identity_smooth {
                let smoother = GaussianFilter::new(sigmas.clone());
                smoother.apply(input)
            } else {
                input.clone()
            };
            
            // 2. Downsample
            let result = if !is_identity_shrink {
                let downsampler = DownsampleFilter::new(factors.clone());
                downsampler.apply(&smoothed)
            } else {
                smoothed
            };
            
            images.push(result);
        }
        
        Self { images }
    }
    
    /// Get image at specific level.
    pub fn get_level(&self, level: usize) -> &Image<B, D> {
        &self.images[level]
    }
    
    /// Get number of levels.
    pub fn levels(&self) -> usize {
        self.images.len()
    }
    
    /// Create a default schedule for N levels with power-of-2 shrinking.
    ///
    /// Returns (shrink_factors, smoothing_sigmas)
    /// Levels are ordered from coarsest to finest.
    /// E.g. levels=3 -> factors [4, 2, 1], sigmas [2.0, 1.0, 0.0]
    pub fn default_schedule(levels: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let mut shrink_factors = Vec::with_capacity(levels);
        let mut smoothing_sigmas = Vec::with_capacity(levels);
        
        for i in 0..levels {
            let exponent = (levels - 1 - i) as u32;
            let factor = 2usize.pow(exponent);
            let sigma = if factor > 1 { 0.5 * (factor as f64) } else { 0.0 };
            
            shrink_factors.push(vec![factor; D]);
            smoothing_sigmas.push(vec![sigma; D]);
        }
        
        (shrink_factors, smoothing_sigmas)
    }
}

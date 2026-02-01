// Multi-resolution registration implementation
// Implements pyramid-based coarse-to-fine registration strategy

use crate::error::RegistrationError;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::progress::{ProgressCallback, ProgressTracker};
use crate::registration::Registration;
use crate::transform::Transform;
use ritk_core::image::Image;
use ritk_core::filter::pyramid::ImagePyramid;
use std::sync::Arc;

/// Configuration for multi-resolution registration
#[derive(Debug, Clone)]
pub struct MultiResolutionConfig {
    /// Number of pyramid levels to use
    pub levels: usize,
    /// Scale factor between levels (e.g., 2.0 for halving)
    pub scale_factor: f64,
    /// Whether to use Gaussian smoothing at each level
    pub use_smoothing: bool,
    /// Sigma for Gaussian smoothing
    pub smoothing_sigma: f64,
    /// Maximum iterations per level (can be scaled by level)
    pub max_iterations_per_level: Vec<usize>,
    /// Convergence threshold per level
    pub convergence_thresholds: Vec<f64>,
}

impl Default for MultiResolutionConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            scale_factor: 2.0,
            use_smoothing: true,
            smoothing_sigma: 1.0,
            max_iterations_per_level: vec![100, 200, 300],
            convergence_thresholds: vec![1e-2, 1e-3, 1e-4],
        }
    }
}

impl MultiResolutionConfig {
    /// Create a new multi-resolution configuration
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            ..Default::default()
        }
    }

    /// Set the scale factor between pyramid levels
    pub fn with_scale_factor(mut self, factor: f64) -> Self {
        self.scale_factor = factor;
        self
    }

    /// Enable/disable Gaussian smoothing
    pub fn with_smoothing(mut self, enabled: bool, sigma: f64) -> Self {
        self.use_smoothing = enabled;
        self.smoothing_sigma = sigma;
        self
    }

    /// Set custom iteration counts per level
    pub fn with_iterations(mut self, iterations: Vec<usize>) -> Self {
        self.max_iterations_per_level = iterations;
        self
    }

    /// Set custom convergence thresholds per level
    pub fn with_thresholds(mut self, thresholds: Vec<f64>) -> Self {
        self.convergence_thresholds = thresholds;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), RegistrationError> {
        if self.levels == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "Multi-resolution registration requires at least 1 level".to_string(),
            ));
        }
        if self.scale_factor <= 1.0 {
            return Err(RegistrationError::InvalidConfiguration(
                "Scale factor must be greater than 1.0".to_string(),
            ));
        }
        if self.max_iterations_per_level.len() != self.levels {
            return Err(RegistrationError::InvalidConfiguration(
                format!(
                    "Expected {} iteration counts, got {}",
                    self.levels,
                    self.max_iterations_per_level.len()
                ),
            ));
        }
        if self.convergence_thresholds.len() != self.levels {
            return Err(RegistrationError::InvalidConfiguration(
                format!(
                    "Expected {} convergence thresholds, got {}",
                    self.levels,
                    self.convergence_thresholds.len()
                ),
            ));
        }
        Ok(())
    }
}

/// Multi-resolution registration engine
pub struct MultiResolutionRegistration<T, M, O>
where
    T: Transform,
    M: Metric<T>,
    O: Optimizer<T, M>,
{
    config: MultiResolutionConfig,
    progress: Option<ProgressCallback>,
    _phantom: std::marker::PhantomData<(T, M, O)>,
}

impl<T, M, O> MultiResolutionRegistration<T, M, O>
where
    T: Transform,
    M: Metric<T>,
    O: Optimizer<T, M>,
{
    /// Create a new multi-resolution registration engine
    pub fn new(config: MultiResolutionConfig) -> Self {
        Self {
            config,
            progress: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set a progress callback
    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress = Some(callback);
        self
    }

    /// Execute multi-resolution registration
    pub fn execute(
        &self,
        fixed: &Image,
        moving: &Image,
        initial_transform: T,
        metric: &M,
        optimizer: &O,
    ) -> Result<(T, Vec<f64>), RegistrationError> {
        self.config.validate()?;

        // Build image pyramids
        let fixed_pyramid = ImagePyramid::build(
            fixed,
            self.config.levels,
            self.config.scale_factor,
            self.config.use_smoothing,
            self.config.smoothing_sigma,
        )?;

        let moving_pyramid = ImagePyramid::build(
            moving,
            self.config.levels,
            self.config.scale_factor,
            self.config.use_smoothing,
            self.config.smoothing_sigma,
        )?;

        let mut current_transform = initial_transform;
        let mut metric_history = Vec::new();

        // Track progress across all levels
        let total_iterations: usize = self.config.max_iterations_per_level.iter().sum();
        let mut completed_iterations = 0;
        let mut tracker = ProgressTracker::new(total_iterations);

        // Process from coarse to fine
        for level in (0..self.config.levels).rev() {
            let level_progress = format!("Level {} (scale: {:.2})", level, self.config.scale_factor.powi((self.config.levels - 1 - level) as i32));
            
            if let Some(ref callback) = self.progress {
                callback(0.0, &level_progress);
            }

            // Get images at this level
            let fixed_level = fixed_pyramid.get_level(level)?;
            let moving_level = moving_pyramid.get_level(level)?;

            // Scale transform parameters for this level
            let scale = self.config.scale_factor.powi((self.config.levels - 1 - level) as i32);
            let scaled_transform = self.scale_transform(&current_transform, scale)?;

            // Create registration for this level
            let mut registration = Registration::new(
                fixed_level.clone(),
                moving_level.clone(),
                scaled_transform,
            );

            registration.set_max_iterations(self.config.max_iterations_per_level[level]);
            registration.set_convergence_threshold(self.config.convergence_thresholds[level]);

            // Add progress tracking for this level
            let level_max_iter = self.config.max_iterations_per_level[level];
            let callback = self.progress.clone();
            let mut level_tracker = ProgressTracker::new(level_max_iter);
            
            registration.set_progress_callback(Box::new(move |progress, message| {
                let overall_progress = (completed_iterations as f64 + progress * level_max_iter as f64) 
                    / total_iterations as f64;
                if let Some(ref cb) = callback {
                    cb(overall_progress, &format!("{}: {}", level_progress, message));
                }
            }));

            // Execute registration at this level
            let (result_transform, level_metrics) = registration.execute(metric, optimizer)?;
            
            metric_history.extend(level_metrics);
            completed_iterations += level_max_iter;

            // Scale transform back to original resolution
            current_transform = self.scale_transform(&result_transform, 1.0 / scale)?;

            if let Some(ref callback) = self.progress {
                callback(
                    completed_iterations as f64 / total_iterations as f64,
                    &format!("{}: Complete", level_progress),
                );
            }
        }

        Ok((current_transform, metric_history))
    }

    /// Scale transform parameters based on resolution level
    fn scale_transform(&self, transform: &T, scale: f64) -> Result<T, RegistrationError> {
        // This is a simplified implementation
        // In practice, you'd need to handle different transform types appropriately
        // For now, we'll just return the transform as-is
        // A proper implementation would scale translation parameters, etc.
        Ok(transform.clone())
    }
}

/// Adaptive multi-resolution configuration
#[derive(Debug, Clone)]
pub struct AdaptiveMultiResolutionConfig {
    base: MultiResolutionConfig,
    /// Minimum image size for a level (in pixels)
    pub min_image_size: usize,
    /// Maximum number of levels to use
    pub max_levels: usize,
    /// Whether to automatically determine number of levels
    pub auto_levels: bool,
}

impl Default for AdaptiveMultiResolutionConfig {
    fn default() -> Self {
        Self {
            base: MultiResolutionConfig::default(),
            min_image_size: 64,
            max_levels: 5,
            auto_levels: true,
        }
    }
}

impl AdaptiveMultiResolutionConfig {
    /// Create adaptive configuration from image dimensions
    pub fn from_images(fixed: &Image, moving: &Image) -> Self {
        let min_dim = fixed.dimensions().iter().cloned()
            .chain(moving.dimensions().iter().cloned())
            .min()
            .unwrap_or(256);

        let mut levels = 1;
        let mut current_size = min_dim;
        while current_size >= 64 && levels < 5 {
            current_size = (current_size as f64 / 2.0) as usize;
            levels += 1;
        }

        let mut config = Self::default();
        config.base.levels = levels;
        config
    }

    /// Convert to standard MultiResolutionConfig
    pub fn into_config(self) -> MultiResolutionConfig {
        self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::MSE;
    use crate::optimizer::GradientDescent;
    use crate::transform::RigidTransform;
    use ritk_core::image::Image;

    #[test]
    fn test_multi_resolution_config_default() {
        let config = MultiResolutionConfig::default();
        assert_eq!(config.levels, 3);
        assert_eq!(config.scale_factor, 2.0);
        assert!(config.use_smoothing);
    }

    #[test]
    fn test_multi_resolution_config_builder() {
        let config = MultiResolutionConfig::new(4)
            .with_scale_factor(1.5)
            .with_smoothing(false, 0.0)
            .with_iterations(vec![50, 100, 150, 200])
            .with_thresholds(vec![1e-1, 1e-2, 1e-3, 1e-4]);

        assert_eq!(config.levels, 4);
        assert_eq!(config.scale_factor, 1.5);
        assert!(!config.use_smoothing);
        assert_eq!(config.max_iterations_per_level.len(), 4);
    }

    #[test]
    fn test_multi_resolution_config_validation() {
        // Valid config
        let config = MultiResolutionConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: zero levels
        let config = MultiResolutionConfig {
            levels: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: scale factor <= 1.0
        let config = MultiResolutionConfig {
            scale_factor: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_adaptive_config() {
        // Create a simple test image
        let dimensions = vec![128usize, 128];
        let spacing = vec![1.0f64, 1.0];
        let data = vec![0.0f32; 128 * 128];
        let image = Image::new(dimensions, spacing, data).unwrap();

        let config = AdaptiveMultiResolutionConfig::from_images(&image, &image);
        assert!(config.base.levels >= 2);
        assert!(config.base.levels <= 5);
    }
}

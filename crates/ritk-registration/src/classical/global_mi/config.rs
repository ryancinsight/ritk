//! Configuration types for global MI registration.

use crate::optimizer::RegularStepGdConfig;
use ritk_core::filter::GaussianSigma;

// ─── Transform Type ───────────────────────────────────────────────────────────

/// Transform type selection for global MI registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalMiTransformType {
    /// D DOF translation only.
    Translation,
    /// 6 DOF rigid (3 rotation + 3 translation) with center of rotation.
    Rigid,
    /// 12 DOF affine (9 matrix + 3 translation) with center.
    Affine,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for multi-resolution Mattes MI + RSGD registration.
///
/// Default values are calibrated to match ITK's `ImageRegistrationMethod`
/// workflow for typical brain MRI registration (≈128³, 1mm isotropic).
#[derive(Debug, Clone)]
pub struct GlobalMiConfig {
    /// Number of multi-resolution levels (default: 3).
    pub num_levels: usize,
    /// Shrink factors per level (default: [4, 2, 1]).
    /// Length must equal `num_levels`.
    pub shrink_factors: Vec<usize>,
    /// Gaussian smoothing sigma per level in physical units. `None` disables
    /// smoothing at that level (replaces the former `0.0` sentinel).
    /// Length must equal `num_levels`. Default: `[Some(4.0), Some(2.0), None]`.
    pub smoothing_sigmas: Vec<Option<GaussianSigma>>,
    /// Number of MI histogram bins (default: 50).
    pub num_mi_bins: usize,
    /// Fraction of voxels to sample for MI estimation (default: 0.20 = 20%).
    pub sampling_percentage: f32,
    /// RSGD configuration per level.
    /// Length must equal `num_levels`.
    pub rsgd_configs: Vec<RegularStepGdConfig>,
    /// Transform type.
    pub transform_type: GlobalMiTransformType,
    /// Center of rotation (None = image center).
    pub center: Option<[f64; 3]>,
}

impl GlobalMiConfig {
    /// Validate configuration invariants.
    ///
    /// # Invariants
    /// - `num_levels > 0`
    /// - `shrink_factors.len() == num_levels`
    /// - `smoothing_sigmas.len() == num_levels`
    /// - `rsgd_configs.len() == num_levels`
    /// - `num_mi_bins >= 4`
    /// - `sampling_percentage ∈ (0, 1]`
    pub fn validate(&self) -> Result<(), String> {
        if self.num_levels == 0 {
            return Err("num_levels must be > 0".to_string());
        }
        if self.shrink_factors.len() != self.num_levels {
            return Err(format!(
                "shrink_factors length ({}) must equal num_levels ({})",
                self.shrink_factors.len(),
                self.num_levels
            ));
        }
        if self.smoothing_sigmas.len() != self.num_levels {
            return Err(format!(
                "smoothing_sigmas length ({}) must equal num_levels ({})",
                self.smoothing_sigmas.len(),
                self.num_levels
            ));
        }
        if self.rsgd_configs.len() != self.num_levels {
            return Err(format!(
                "rsgd_configs length ({}) must equal num_levels ({})",
                self.rsgd_configs.len(),
                self.num_levels
            ));
        }
        if self.num_mi_bins < 4 {
            return Err(format!(
                "num_mi_bins must be >= 4, got {}",
                self.num_mi_bins
            ));
        }
        if self.sampling_percentage <= 0.0 || self.sampling_percentage > 1.0 {
            return Err(format!(
                "sampling_percentage must be in (0, 1], got {}",
                self.sampling_percentage
            ));
        }
        for (i, cfg) in self.rsgd_configs.iter().enumerate() {
            if let Err(e) = cfg.validate() {
                return Err(format!("rsgd_configs[{}] validation failed: {}", i, e));
            }
        }
        Ok(())
    }

    /// Create a default rigid registration configuration.
    ///
    /// 3 levels with shrink factors [4, 2, 1], sigmas [4.0, 2.0, disabled].
    pub fn rigid_default() -> Self {
        Self {
            num_levels: 3,
            shrink_factors: vec![4, 2, 1],
            smoothing_sigmas: vec![
                Some(GaussianSigma::new_unchecked(4.0)),
                Some(GaussianSigma::new_unchecked(2.0)),
                None,
            ],
            num_mi_bins: 50,
            sampling_percentage: 0.20,
            rsgd_configs: vec![
                RegularStepGdConfig {
                    initial_step_length: 2.0,
                    relaxation_factor: 0.5,
                    minimum_step_length: 1e-4,
                    maximum_step_length: 10.0,
                    gradient_tolerance: 1e-4,
                    maximum_iterations: 100,
                },
                RegularStepGdConfig {
                    initial_step_length: 1.0,
                    relaxation_factor: 0.5,
                    minimum_step_length: 1e-5,
                    maximum_step_length: 5.0,
                    gradient_tolerance: 1e-5,
                    maximum_iterations: 150,
                },
                RegularStepGdConfig {
                    initial_step_length: 0.5,
                    relaxation_factor: 0.5,
                    minimum_step_length: 1e-6,
                    maximum_step_length: 2.0,
                    gradient_tolerance: 1e-6,
                    maximum_iterations: 200,
                },
            ],
            transform_type: GlobalMiTransformType::Rigid,
            center: None,
        }
    }

    /// Create a default affine registration configuration.
    pub fn affine_default() -> Self {
        let mut cfg = Self::rigid_default();
        cfg.transform_type = GlobalMiTransformType::Affine;
        cfg.rsgd_configs[2].maximum_iterations = 300;
        cfg
    }

    /// Create a default translation-only registration configuration.
    pub fn translation_default() -> Self {
        let mut cfg = Self::rigid_default();
        cfg.transform_type = GlobalMiTransformType::Translation;
        cfg.rsgd_configs[0].maximum_iterations = 50;
        cfg.rsgd_configs[1].maximum_iterations = 80;
        cfg.rsgd_configs[2].maximum_iterations = 100;
        cfg
    }
}

impl Default for GlobalMiConfig {
    fn default() -> Self {
        Self::rigid_default()
    }
}

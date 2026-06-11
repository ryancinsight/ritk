//! Shared configuration and result types for Demons-family registration.

use ritk_core::filter::GaussianSigma;

/// Variant selector for Demons registration algorithms.
///
/// Replaces the former `use_diffeomorphic: bool` field, eliminating boolean
/// blindness at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DemonsVariant {
    /// Classic Thirion Demons (displacement-based, non-diffeomorphic).
    #[default]
    Classic,
    /// Diffeomorphic Demons via stationary velocity field exponentiation.
    Diffeomorphic,
}

impl DemonsVariant {
    /// Returns `true` if this variant is [`Diffeomorphic`](DemonsVariant::Diffeomorphic).
    pub fn is_diffeomorphic(self) -> bool {
        matches!(self, Self::Diffeomorphic)
    }
}

/// Configuration for Demons-family registration algorithms.
#[derive(Debug, Clone)]
pub struct DemonsConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Gaussian standard deviation (voxels) applied to the total displacement
    /// field after each iteration (diffusive regularisation). `None` disables
    /// diffusion smoothing.
    pub sigma_diffusion: Option<GaussianSigma>,
    /// Gaussian standard deviation (voxels) applied to the force update before
    /// accumulation (fluid regularisation). `None` disables fluid smoothing.
    pub sigma_fluid: Option<GaussianSigma>,
    /// Maximum per-voxel step length in voxel units.  Forces whose magnitude
    /// exceeds this value are rescaled to exactly `max_step_length`.
    pub max_step_length: f32,
}

impl Default for DemonsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            sigma_diffusion: Some(GaussianSigma::new_unchecked(1.5)),
            sigma_fluid: None,
            max_step_length: 2.0,
        }
    }
}

/// Result returned by a Demons-family registration.
#[derive(Debug, Clone)]
pub struct DemonsResult {
    /// Warped moving image (same shape as fixed image).
    pub warped: Vec<f32>,
    /// Z-component of the final displacement field (voxel units).
    pub disp_z: Vec<f32>,
    /// Y-component of the final displacement field.
    pub disp_y: Vec<f32>,
    /// X-component of the final displacement field.
    pub disp_x: Vec<f32>,
    /// Optional Z-component of the stationary velocity field whose exponential
    /// map produced `disp_z`.
    ///
    /// Present for diffeomorphic Demons results. Absent for displacement-based
    /// variants such as classic Thirion and symmetric Demons.
    pub vel_z: Option<Vec<f32>>,
    /// Optional Y-component of the stationary velocity field.
    pub vel_y: Option<Vec<f32>>,
    /// Optional X-component of the stationary velocity field.
    pub vel_x: Option<Vec<f32>>,
    /// Mean-squared error between fixed and warped moving at the final iteration.
    pub final_mse: f64,
    /// Actual number of iterations performed (may be less than `max_iterations`
    /// if convergence was reached).
    pub num_iterations: usize,
}

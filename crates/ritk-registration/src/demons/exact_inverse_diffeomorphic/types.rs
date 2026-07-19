//! Configuration and result types for inverse-consistent diffeomorphic Demons registration.

use super::super::config::DemonsConfig;

/// Configuration for InverseConsistentDiffeomorphicDemonsRegistration.
#[derive(Debug, Clone)]
pub struct InverseConsistentDemonsConfig {
    /// Shared Demons parameters.
    pub demons: DemonsConfig,
    /// Weight of the backward (inverse) force. Range `[0, 1]`. Default 0.5.
    pub inverse_consistency_weight: f64,
    /// Scaling-and-squaring steps for exp(v). Default 6.
    pub n_squarings: usize,
}

impl Default for InverseConsistentDemonsConfig {
    fn default() -> Self {
        Self {
            demons: DemonsConfig::default(),
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
        }
    }
}

/// Result of InverseConsistentDiffeomorphicDemonsRegistration.
pub struct InverseConsistentDemonsResult {
    /// Moving image warped onto fixed using phi_fwd = exp(v).
    pub warped: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), z-component.
    pub disp_z: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), y-component.
    pub disp_y: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), x-component.
    pub disp_x: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), z-component.
    pub inv_disp_z: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), y-component.
    pub inv_disp_y: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), x-component.
    pub inv_disp_x: Vec<f32>,
    /// Stationary velocity field, z-component.
    pub vel_z: Vec<f32>,
    /// Stationary velocity field, y-component.
    pub vel_y: Vec<f32>,
    /// Stationary velocity field, x-component.
    pub vel_x: Vec<f32>,
    /// Final MSE(F, M o phi_fwd) at convergence.
    pub final_mse: f64,
    /// Number of iterations executed.
    pub num_iterations: usize,
    /// IC residual: meanâ€–Ï†_fwd(Ï†_inv(x)) âˆ’ xâ€–â‚‚.
    pub inverse_consistency_residual: f64,
}

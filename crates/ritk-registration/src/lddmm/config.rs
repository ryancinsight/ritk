//! LDDMM configuration and result types.

use ritk_filter::GaussianSigma;

/// Parameters for LDDMM registration.
#[derive(Debug, Clone)]
pub struct LddmmConfig {
    /// Maximum number of gradient-descent iterations.
    pub max_iterations: usize,
    /// Number of Euler steps for geodesic integration (N\_t).
    pub num_time_steps: usize,
    /// Standard deviation (voxels) of Gaussian kernel K_Ïƒ for the Sobolev norm.
    pub kernel_sigma: GaussianSigma,
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Weight Î» on the regularisation term â€–vâ‚€â€–Â²\_V.
    pub regularization_weight: f64,
    /// Stop when |MSE\_{k} âˆ’ MSE\_{kâˆ’1}| / (MSE\_{kâˆ’1} + Îµ) < threshold.
    pub convergence_threshold: f64,
}

impl Default for LddmmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            num_time_steps: 10,
            kernel_sigma: GaussianSigma::new_unchecked(2.0),
            learning_rate: 0.1,
            regularization_weight: 1.0,
            convergence_threshold: 1e-5,
        }
    }
}

/// Output of LDDMM registration.
#[derive(Debug, Clone)]
pub struct LddmmResult {
    /// Optimised initial velocity (vz, vy, vx) parameterising the geodesic.
    pub initial_velocity: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Displacement field (dz, dy, dx) at t = 1 in voxel units.
    pub displacement_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Moving image warped by Ï†â‚.
    pub warped_moving: Vec<f32>,
    /// Final MSE after the last forward pass.
    pub final_metric: f64,
    /// Number of gradient-descent iterations executed.
    pub num_iterations: usize,
}

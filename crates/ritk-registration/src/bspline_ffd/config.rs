/// Configuration for B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDConfig {
    /// Initial control-point spacing in voxels `[sz, sy, sx]`.
    pub initial_control_spacing: [usize; 3],
    /// Number of multi-resolution levels. Control spacing is halved at each
    /// subsequent level.
    pub num_levels: usize,
    /// Maximum gradient-descent iterations per level.
    pub max_iterations_per_level: usize,
    /// Learning rate (step size) for gradient descent on control displacements.
    pub learning_rate: f64,
    /// Bending-energy regularization weight λ.
    pub regularization_weight: f64,
    /// Convergence threshold: optimization stops when the relative change in
    /// the NCC metric between consecutive iterations falls below this value.
    pub convergence_threshold: f64,
}

impl Default for BSplineFFDConfig {
    fn default() -> Self {
        Self {
            initial_control_spacing: [8, 8, 8],
            num_levels: 3,
            max_iterations_per_level: 100,
            learning_rate: 1.0,
            regularization_weight: 1e-3,
            convergence_threshold: 1e-5,
        }
    }
}

/// Result of B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDResult {
    /// Control-point displacements for each spatial component (dz, dy, dx).
    /// Each `Vec<f32>` has length `control_grid_dims[0] * control_grid_dims[1]
    /// * control_grid_dims[2]`.
    pub control_points: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Control-lattice dimensions `[nz, ny, nx]`.
    pub control_grid_dims: [usize; 3],
    /// Control-point spacing at the finest level `[δz, δy, δx]` in voxels.
    pub control_spacing: [f64; 3],
    /// Moving image warped to the fixed image domain.
    pub warped_moving: Vec<f32>,
    /// Final NCC metric value (higher → better alignment).
    pub final_metric: f64,
    /// Total gradient-descent iterations across all levels.
    pub num_iterations: usize,
}

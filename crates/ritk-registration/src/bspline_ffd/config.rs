use super::ctrl_dims::ControlGridDims;

/// Default bending-energy regularization weight for B-Spline FFD.
pub const DEFAULT_REGULARIZATION_WEIGHT: f64 = 1e-3;
/// Default convergence threshold (relative NCC change) for B-Spline FFD.
pub const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1e-5;

/// Configuration for B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDConfig {
    /// Spacing between B-spline control points in `[depth, rows, cols]` (voxels).
    pub initial_control_spacing: [usize; 3],
    /// Number of multi-resolution levels. Control spacing is halved at each
    /// subsequent level.
    pub num_levels: usize,
    /// Maximum gradient-descent iterations per level.
    pub max_iterations_per_level: usize,
    /// Learning rate (step size) for gradient descent on control displacements.
    pub learning_rate: f64,
    /// Bending-energy regularization weight Î».
    pub regularization_weight: f64,
    /// Convergence threshold: optimization stops when the relative change in
    /// the NCC metric between consecutive iterations falls below this value.
    pub convergence_threshold: f64 }

impl Default for BSplineFFDConfig {
    fn default() -> Self {
        Self {
            initial_control_spacing: [8, 8, 8],
            num_levels: 3,
            max_iterations_per_level: 100,
            learning_rate: 1.0,
            regularization_weight: DEFAULT_REGULARIZATION_WEIGHT,
            convergence_threshold: DEFAULT_CONVERGENCE_THRESHOLD }
    }
}

/// Result of B-Spline FFD registration.
#[derive(Debug, Clone)]
pub struct BSplineFFDResult {
    /// Control-point displacements for each spatial component (dz, dy, dx).
    /// Each `Vec<f32>` has length `control_grid_dims.num_nodes()`.
    pub control_points: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Control grid dimensions `[depth_ctrl, rows_ctrl, cols_ctrl]`.
    pub control_grid_dims: ControlGridDims,
    /// Control-point spacing at the finest level `[Î´z, Î´y, Î´x]` in voxels.
    pub control_spacing: [f64; 3],
    /// Moving image warped to the fixed image domain.
    pub warped_moving: Vec<f32>,
    /// Final NCC metric value (higher â†’ better alignment).
    pub final_metric: f64,
    /// Total gradient-descent iterations across all levels.
    pub num_iterations: usize }

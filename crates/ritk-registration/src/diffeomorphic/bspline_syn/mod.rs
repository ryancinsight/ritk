//! B-Spline Symmetric Normalization (BSplineSyN) registration.
//!
//! # Mathematical Specification
//!
//! BSplineSyN parameterises the stationary velocity fields `vâ‚, vâ‚‚` of the
//! SyN framework using cubic B-spline control-point lattices instead of dense
//! voxel grids. This reduces the number of free parameters and provides
//! intrinsic CÂ²-smooth velocity fields.
//!
//! ## Dense Field Evaluation
//!
//! For voxel position `(z, y, x)` and control-point spacing `s`:
//!
//! `t_d = d / s_d`, `span_d = âŒŠt_dâŒ‹`, `u_d = t_d âˆ’ span_d`
//!
//! `v(z,y,x) = Î£_{l,m,n=0}^{3} Bâ‚—(u_z) Bâ‚˜(u_y) Bâ‚™(u_x) Â· cp[span_z+l, span_y+m, span_x+n]`
//!
//! ## Bending Energy Regularisation
//!
//! Discrete 6-connected Laplacian on the CP lattice:
//!
//! `Î”cp[i,j,k] = Î£_face_neighbours cp[n] âˆ’ count Â· cp[i,j,k]`
//!
//! Weight `Î»` controls regularisation strength.
//!
//! # Memory discipline
//! All volume-sized scratch and local-CC tables are allocated before the
//! iteration loop and rebuilt in place. The fused CC dispatcher creates only
//! an `O(nz)` slice-descriptor vector; all numerical outputs write into
//! caller-provided buffers.
//!
//! # References
//! - Tustison, N. J. & Avants, B. B. (2013). Explicit B-spline regularization
//!   in diffeomorphic image registration. *Frontiers in Neuroinformatics* 7:39.
//! - Rueckert, D. et al. (1999). Nonrigid registration using free-form deformations.
//!   *IEEE TMI* 18(8):712â€“721.

use crate::deformable_field_ops::VelocityField;

mod buffers;
pub(crate) mod primitives;
mod registration;
#[cfg(test)]
mod tests;

// â”€â”€ Public types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Configuration for BSplineSyN registration.
#[derive(Debug, Clone)]
pub struct BSplineSyNConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Control-point spacing in voxels per axis `[sz, sy, sx]`.
    pub control_spacing: [usize; 3],
    /// Gaussian Ïƒ (voxels) applied to dense CC forces before CP accumulation.
    pub sigma_smooth: f64,
    /// Stop when CC variance over the convergence window falls below this.
    pub convergence_threshold: f64,
    /// Number of recent CC values for convergence checking.
    pub convergence_window: usize,
    /// Number of scaling-and-squaring steps for `exp(v)`.
    pub n_squarings: usize,
    /// Radius of local CC window (voxels).
    pub cc_window_radius: usize,
    /// Maximum per-step displacement (voxels) used to normalise the CC gradient
    /// before accumulating into the velocity field.  Mirrors the ANTs
    /// `gradientStep` parameter.  Default: 0.25.
    pub gradient_step: f64,
    /// Bending energy regularisation weight (Laplacian smoothing on CPs).
    pub regularization_weight: f64,
}

/// Result returned by [`BSplineSyNRegistration::register`].
#[derive(Debug, Clone)]
pub struct BSplineSyNResult {
    /// Forward dense velocity field `vâ‚` components (fixedâ†’midpoint), in (z, y, x) order.
    pub forward_field: VelocityField,
    /// Inverse dense velocity field `vâ‚‚` components (movingâ†’midpoint), in (z, y, x) order.
    pub inverse_field: VelocityField,
    /// Fixed image warped to the midpoint by `Ï†â‚ = exp(vâ‚)`.
    pub warped_fixed: Vec<f32>,
    /// Moving image warped to the midpoint by `Ï†â‚‚ = exp(vâ‚‚)`.
    pub warped_moving: Vec<f32>,
    /// Final mean local CC value (higher is better; 1.0 = perfect alignment).
    pub final_cc: f64,
    /// Number of iterations actually performed.
    pub num_iterations: usize,
}

/// BSplineSyN registration engine.
///
/// Represents velocity fields via cubic B-spline control-point lattices,
/// providing intrinsic CÂ²-smoothness and reduced parameter count compared to
/// dense SyN.
#[derive(Debug, Clone)]
pub struct BSplineSyNRegistration {
    /// Algorithm configuration.
    pub config: BSplineSyNConfig,
}

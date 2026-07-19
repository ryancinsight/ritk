//! Diffeomorphic image registration via Symmetric Normalization (SyN).
//!
//! # Mathematical Specification
//!
//! SyN (Avants et al. 2008) minimises the symmetric energy:
//!
//!   E(φ₁, φ₂) = D(I∘φ₁⁻¹, J∘φ₂⁻¹) + Reg(φ₁) + Reg(φ₂)
//!
//! where φ₁ (fixed→midpoint) and φ₂ (moving→midpoint) are independently evolved
//! diffeomorphisms.  The **greedy SyN** variant (implemented here) uses
//! first-order gradient descent on the local cross-correlation (CC) metric and
//! represents each diffeomorphism as the exponential map of a stationary velocity
//! field, computed via scaling-and-squaring.
//!
//! **Local CC gradient** (Avants 2008, eq. 10) for force on φ₁:
//!
//!   `fz[p] = [(J_w[p]−μ_J)/(σ_I·σ_J) − CC·(I_w[p]−μ_I)/σ_I²] · gIz[p]`
//!
//! where cc_num = Σ_{q∈W}(I_w(q)-μ_I)(J_w(q)-μ_J), σ_I = sqrt(Σ_{q∈W}(I_w(q)-μ_I)²),
//! σ_J = sqrt(Σ_{q∈W}(J_w(q)-μ_J)²), CC = cc_num / (σ_I·σ_J), and W is the local
//! window of radius r.
//!
//! # References
//! - Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. (2008).
//!   Symmetric diffeomorphic image registration with cross-correlation:
//!   Evaluating automated labeling of elderly and neurodegenerative brain.
//!   *Medical Image Analysis* 12(1):26–41.

pub mod bspline_syn;
pub mod local_cc;
pub mod multires_syn;
pub mod syn_core;

pub use bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration, BSplineSyNResult};
pub use multires_syn::{InverseConsistency, MultiResSyNConfig, MultiResSyNRegistration};
pub use syn_core::{SyNRegistration, SyNResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for SyN (Symmetric Normalization) registration.
#[derive(Debug, Clone)]
pub struct SyNConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Standard deviation (voxels) of Gaussian regularisation applied to each
    /// velocity field after every update step.
    pub sigma_smooth: f64,
    /// Convergence criterion: stop when the variance of the last
    /// `convergence_window` CC values is below this threshold.
    pub convergence_threshold: f64,
    /// Number of recent CC values to track for convergence checking.
    pub convergence_window: usize,
    /// Number of scaling-and-squaring steps for `exp(v)` (2^n integration steps).
    pub n_squarings: usize,
    /// Radius of the local CC window (voxels).
    pub cc_window_radius: usize,
    /// Maximum per-step displacement (voxels) used to normalise the CC gradient
    /// before accumulating into the velocity field.  Mirrors the ANTs
    /// `gradientStep` parameter.  Default: 0.25.
    pub gradient_step: f64,
}

impl Default for SyNConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            sigma_smooth: 3.0,
            convergence_threshold: 1e-6,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: 2,
            gradient_step: 0.25,
        }
    }
}

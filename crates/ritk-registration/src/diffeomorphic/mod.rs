//! Diffeomorphic image registration via Symmetric Normalization (SyN).
//!
//! # Mathematical Specification
//!
//! SyN (Avants et al. 2008) minimises the symmetric energy:
//!
//!   E(Ï†â‚, Ï†â‚‚) = D(Iâˆ˜Ï†â‚â»Â¹, Jâˆ˜Ï†â‚‚â»Â¹) + Reg(Ï†â‚) + Reg(Ï†â‚‚)
//!
//! where Ï†â‚ (fixedâ†’midpoint) and Ï†â‚‚ (movingâ†’midpoint) are independently evolved
//! diffeomorphisms.  The **greedy SyN** variant (implemented here) uses
//! first-order gradient descent on the local cross-correlation (CC) metric and
//! represents each diffeomorphism as the exponential map of a stationary velocity
//! field, computed via scaling-and-squaring.
//!
//! **Local CC gradient** (Avants 2008, eq. 10) for force on Ï†â‚:
//!
//!   `fz[p] = [(J_w[p]âˆ’Î¼_J)/(Ïƒ_IÂ·Ïƒ_J) âˆ’ CCÂ·(I_w[p]âˆ’Î¼_I)/Ïƒ_IÂ²] Â· gIz[p]`
//!
//! where cc_num = Î£_{qâˆˆW}(I_w(q)-Î¼_I)(J_w(q)-Î¼_J), Ïƒ_I = sqrt(Î£_{qâˆˆW}(I_w(q)-Î¼_I)Â²),
//! Ïƒ_J = sqrt(Î£_{qâˆˆW}(J_w(q)-Î¼_J)Â²), CC = cc_num / (Ïƒ_IÂ·Ïƒ_J), and W is the local
//! window of radius r.
//!
//! # References
//! - Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. (2008).
//!   Symmetric diffeomorphic image registration with cross-correlation:
//!   Evaluating automated labeling of elderly and neurodegenerative brain.
//!   *Medical Image Analysis* 12(1):26â€“41.

pub mod bspline_syn;
pub mod local_cc;
pub mod multires_syn;
pub mod syn_core;

pub use bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration, BSplineSyNResult};
pub use multires_syn::{InverseConsistency, MultiResSyNConfig, MultiResSyNRegistration};
pub use syn_core::{SyNRegistration, SyNResult};

// â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

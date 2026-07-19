//! Multi-Resolution Symmetric Normalization (SyN) registration.
//!
//! # Mathematical Specification
//!
//! Multi-resolution SyN executes the SyN optimization at multiple image
//! resolutions in a coarse-to-fine hierarchy. At level `l` ∈ {0, …, L−1}
//! (0 = coarsest):
//!
//! 1. Compute downsample factor `f = 2^(L − l − 1)`
//! 2. Downsample fixed `F` and moving `M` by factor `f` via average pooling
//! 3. If `l > 0`, upsample velocity fields `v₁, v₂` from level `l−1` to
//!    current resolution via trilinear interpolation with displacement scaling
//! 4. Run SyN iterations at this level (max = `iterations_per_level[l]`)
//! 5. Optionally enforce inverse consistency: `v₁ ← (v₁ − compose(v₁,v₂))/2`
//!
//! ## Downsampling
//!
//! Average pooling with stride `f` in each dimension:
//! `out[oz,oy,ox] = mean(in[oz·f .. min(oz·f+f, D), ...])`
//! Output dimension per axis: `new_d = max(1, d / f)`.
//!
//! ## Upsampling
//!
//! Trilinear interpolation to target dimensions. Displacement component `d` is
//! scaled by `new_dims[d] / old_dims[d]` to preserve physical displacement
//! magnitude across voxel-size changes.
//!
//! ## Local CC Gradient (Avants 2008, eq. 10)
//!
//! `f_z[p] = −2 · cc_num / (var_I · var_J + ε) · (J_w[p] − μ_J) · ∇I_z[p]`
//!
//! where sums are over a local window of radius `r` centred at `p`.
//! One five-channel summed-area-table set serves both force directions and the
//! convergence mean at each iteration; the three results therefore share one
//! window-statistics construction and one voxel traversal. Its volume storage
//! is allocated once per resolution level and rebuilt in place.
//!
//! ## Inverse Consistency Enforcement
//!
//! After each iteration (when enabled), both velocity fields are nudged toward
//! mutual inverse consistency:
//! `c₁ = compose(v₁, v₂); c₂ = compose(v₂, v₁)`
//! `v₁ ← (v₁ − c₁) / 2; v₂ ← (v₂ − c₂) / 2`
//! Both corrections are computed from the pre-update fields to maintain symmetry.
//!
//! # References
//! - Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. (2008).
//!   Symmetric diffeomorphic image registration with cross-correlation.
//!   *Medical Image Analysis* 12(1):26–41.

pub(crate) mod pyramid;
mod registration;

/// Inverse-consistency enforcement policy for SyN velocity field updates.
///
/// Replaces the former `enforce_inverse_consistency: bool` field, eliminating
/// boolean blindness at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseConsistency {
    /// No inverse-consistency enforcement (relaxed update).
    #[default]
    Relaxed,
    /// Enforce inverse consistency via `v ← (v − compose(v₁,v₂)) / 2`.
    Enforced,
}

#[cfg(test)]
mod tests;

/// Configuration for multi-resolution SyN registration.
#[derive(Debug, Clone)]
pub struct MultiResSyNConfig {
    /// Number of resolution levels (e.g., 3 → factors 4×, 2×, 1×).
    pub num_levels: usize,
    /// Maximum iterations at each level. Length must equal `num_levels`.
    pub iterations_per_level: Vec<usize>,
    /// Gaussian regularisation σ (voxels) applied to velocity fields.
    pub sigma_smooth: f64,
    /// Stop when CC variance over the convergence window falls below this.
    pub convergence_threshold: f64,
    /// Number of recent CC values for convergence checking.
    pub convergence_window: usize,
    /// Number of scaling-and-squaring steps for exp(v).
    pub n_squarings: usize,
    /// Radius of local CC window (voxels).
    pub cc_window_radius: usize,
    /// Maximum per-step displacement (voxels) used to normalise the CC gradient
    /// before accumulating into the velocity field. Mirrors the ANTs
    /// `gradientStep` parameter. Default: 0.25.
    pub gradient_step: f64,
    /// Inverse-consistency enforcement policy.
    /// Default: [`InverseConsistency::Relaxed`].
    pub enforce_inverse_consistency: InverseConsistency,
}

/// Multi-resolution SyN registration engine.
#[derive(Debug, Clone)]
pub struct MultiResSyNRegistration {
    pub config: MultiResSyNConfig,
}

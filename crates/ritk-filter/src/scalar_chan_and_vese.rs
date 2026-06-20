//! Chan-Vese dense level set filter with user-provided initial level set.
//!
//! # Algorithm
//!
//! Implements `itk::ScalarChanAndVeseDenseLevelSetImageFilter`. Differs from
//! the existing `ritk_segmentation::ChanVeseSegmentation` in two ways:
//!
//! 1. The initial level set φ₀ is supplied by the caller rather than being
//!    initialised internally as a checkerboard or Otsu bipartition.
//! 2. The feature image u₀ (the image driving the data-fidelity energy) is a
//!    separate input from the level set.
//! 3. Returns the **evolved level set φ** rather than a binary segmentation mask.
//!
//! ## Sign convention
//!
//! Follows ITK's convention: **φ < 0 inside the region of interest**,
//! φ > 0 outside.
//!
//! ## Energy functional (φ < 0 = inside)
//!
//! ```text
//! E(φ, c₁, c₂) = μ · Length(C) + ν · Area(inside)
//!              + λ₁ ∫ |u₀ − c₁|² (1 − H_ε(φ)) dx   [inside term]
//!              + λ₂ ∫ |u₀ − c₂|² H_ε(φ) dx           [outside term]
//! ```
//!
//! where H_ε selects the **outside** region (φ > 0), so 1 − H_ε selects inside (φ < 0).
//!
//! ## Region means
//!
//! ```text
//! c₁ = ∫ u₀ · (1 − H_ε(φ)) dx / ∫ (1 − H_ε(φ)) dx    (inside, φ < 0)
//! c₂ = ∫ u₀ · H_ε(φ) dx        / ∫ H_ε(φ) dx           (outside, φ > 0)
//! ```
//!
//! ## PDE evolution (Euler-Lagrange of E)
//!
//! ```text
//! ∂φ/∂t = δ_ε(φ) [ μ · κ + ν + λ₁(u₀ − c₁)² − λ₂(u₀ − c₂)² ]
//! ```
//!
//! The sign of the data terms is reversed compared to the φ > 0 = inside
//! convention: when u₀ ≈ c₁ at a boundary voxel (φ ≈ 0), the force is negative
//! → φ decreases → voxel moves into the inside region.
//!
//! ## Regularised Heaviside and Dirac
//!
//! ```text
//! H_ε(z) = 0.5 · (1 + (2/π) · arctan(z / ε))
//! δ_ε(z) = (ε / π) / (ε² + z²)
//! ```
//!
//! ## Complexity
//!
//! O(max_iterations · N) where N = total voxels.
//!
//! # References
//!
//! - Chan, T. F. & Vese, L. A. (2001). "Active Contours Without Edges."
//!   *IEEE Transactions on Image Processing*, 10(2), 266–277.

use std::f64::consts::PI;

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::level_set_helpers::compute_curvature_into;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Chan-Vese dense level set filter with user-provided initial level set.
///
/// Returns the evolved level set function φ (not a binary mask). The caller can
/// threshold at φ = 0: voxels with φ < 0 belong to the segmented region.
///
/// # Default parameters
///
/// | Field | Default |
/// |-------|---------|
/// | `number_of_iterations` | 100 |
/// | `lambda1` | 1.0 |
/// | `lambda2` | 1.0 |
/// | `mu` | 1.0 |
/// | `nu` | 0.0 |
/// | `dt` | 0.25 |
/// | `epsilon` | 1.0 |
#[derive(Debug, Clone)]
pub struct ScalarChanAndVeseDenseLevelSet {
    /// Maximum number of PDE evolution iterations.
    pub number_of_iterations: usize,
    /// Data fidelity weight λ₁ for the inside region.
    pub lambda1: f32,
    /// Data fidelity weight λ₂ for the outside region.
    pub lambda2: f32,
    /// Curvature (length) penalty weight μ.
    pub mu: f32,
    /// Area penalty weight ν. Positive values penalise large inside regions.
    pub nu: f32,
    /// Maximum per-voxel step size per iteration.
    ///
    /// The adaptive time step scales per iteration as `dt / max|δ_ε(φ)·force|`,
    /// so no voxel advances more than `dt` in a single step. Matches ITK's
    /// `TimeStep` default of `0.25`.
    pub dt: f32,
    /// Regularisation width ε for Heaviside and Dirac approximations.
    pub epsilon: f32,
}

impl Default for ScalarChanAndVeseDenseLevelSet {
    fn default() -> Self {
        Self {
            number_of_iterations: 100,
            lambda1: 1.0,
            lambda2: 1.0,
            mu: 1.0,
            nu: 0.0,
            dt: 0.25,
            epsilon: 1.0,
        }
    }
}

impl ScalarChanAndVeseDenseLevelSet {
    /// Evolve a level set under the Chan-Vese energy.
    ///
    /// # Arguments
    /// - `initial_level_set`: φ₀ with **φ < 0 inside** the region of interest.
    /// - `feature_image`: u₀ — the scalar image driving the data-fidelity energy.
    ///   Must have the same shape as `initial_level_set`.
    ///
    /// # Returns
    /// The evolved level set φ as a floating-point image.
    ///
    /// # Errors
    /// Returns `Err` if tensor extraction fails or if the image shapes differ.
    pub fn apply<B: Backend>(
        &self,
        initial_level_set: &Image<B, 3>,
        feature_image: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = initial_level_set.shape();
        if dims != feature_image.shape() {
            anyhow::bail!(
                "initial_level_set shape {:?} and feature_image shape {:?} must match",
                dims,
                feature_image.shape()
            );
        }

        let (phi_init, _) = extract_vec(initial_level_set)?;
        let (feat, _) = extract_vec(feature_image)?;

        // Promote to f64 for PDE accuracy.
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();
        let feat_f64: Vec<f64> = feat.iter().map(|&v| v as f64).collect();

        self.evolve(&mut phi, &feat_f64, dims);

        let result: Vec<f32> = phi.iter().map(|&v| v as f32).collect();
        Ok(rebuild(result, dims, initial_level_set))
    }
}

// ── Core PDE ───────────────────────────────────────────────────────────────────

impl ScalarChanAndVeseDenseLevelSet {
    fn evolve(&self, phi: &mut [f64], feat: &[f64], dims: [usize; 3]) {
        let n = dims[0] * dims[1] * dims[2];
        let eps = self.epsilon as f64;
        let mu = self.mu as f64;
        let nu = self.nu as f64;
        let lam1 = self.lambda1 as f64;
        let lam2 = self.lambda2 as f64;
        let dt = self.dt as f64;

        let mut kappa = vec![0.0_f64; n];

        for _ in 0..self.number_of_iterations {
            // ── 1. Region means ───────────────────────────────────────────────
            let (c1, c2) = region_means(feat, phi, eps);

            // ── 2. Curvature κ = div(∇φ/|∇φ|) ─────────────────────────────
            compute_curvature_into(phi, dims, &mut kappa);

            // ── 3. PDE step (adaptive dt) ─────────────────────────────────────
            //
            // Compute per-voxel update values first to find the global maximum,
            // then scale so the largest step equals `dt` (ITK stability criterion:
            // actual_dt = dt / max|δ_ε(φ)·force|).
            let mut delta = vec![0.0_f64; n];
            let mut max_abs = 0.0_f64;
            for i in 0..n {
                let d = dirac(phi[i], eps);
                let diff1 = feat[i] - c1;
                let diff2 = feat[i] - c2;
                // ∂φ/∂t = δ_ε(φ)[μ·κ + ν + λ₁(u₀−c₁)² − λ₂(u₀−c₂)²]
                let force = mu * kappa[i] + nu + lam1 * diff1 * diff1 - lam2 * diff2 * diff2;
                let dv = d * force;
                delta[i] = dv;
                let abs_val = dv.abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }
            // Scale so no voxel moves more than `dt` per iteration.
            let actual_dt = if max_abs > 1e-10 { dt / max_abs } else { dt };
            for i in 0..n {
                phi[i] += actual_dt * delta[i];
            }
        }
    }
}

// ── Region mean computation ───────────────────────────────────────────────────

/// Compute (c₁, c₂) under the φ < 0 = inside convention.
///
/// ```text
/// c₁ = Σ u₀ · (1 − H_ε(φ)) / Σ (1 − H_ε(φ))   (inside, φ < 0)
/// c₂ = Σ u₀ · H_ε(φ)        / Σ H_ε(φ)          (outside, φ > 0)
/// ```
fn region_means(feat: &[f64], phi: &[f64], eps: f64) -> (f64, f64) {
    let mut sum_in = 0.0_f64;
    let mut sum_u_in = 0.0_f64;
    let mut sum_out = 0.0_f64;
    let mut sum_u_out = 0.0_f64;

    for i in 0..feat.len() {
        let h = heaviside(phi[i], eps); // selects outside (φ > 0)
        let omh = 1.0 - h; // selects inside  (φ < 0)
        sum_in += omh;
        sum_u_in += feat[i] * omh;
        sum_out += h;
        sum_u_out += feat[i] * h;
    }

    let c1 = if sum_in > 1e-15 {
        sum_u_in / sum_in
    } else {
        0.0
    };
    let c2 = if sum_out > 1e-15 {
        sum_u_out / sum_out
    } else {
        0.0
    };
    (c1, c2)
}

// ── Inline regularised Heaviside and Dirac ────────────────────────────────────

/// H_ε(z) = 0.5 · (1 + (2/π) · arctan(z / ε))
///
/// Approaches 1 for z → +∞ (outside, φ > 0) and 0 for z → −∞ (inside, φ < 0).
#[inline]
fn heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / PI) * (z / eps).atan())
}

/// δ_ε(z) = (ε / π) / (ε² + z²)
///
/// Derivative of `heaviside` with respect to `z`. Positive everywhere, peak at z = 0.
#[inline]
fn dirac(z: f64, eps: f64) -> f64 {
    (eps / PI) / (eps * eps + z * z)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_scalar_chan_and_vese.rs"]
mod tests;

//! Scalar Chan-Vese dense level set (ITK `ScalarChanAndVeseDenseLevelSetImageFilter`).
//!
//! # Mathematical Specification
//!
//! Ports `sitk.ScalarChanAndVeseDenseLevelSet` — a
//! `MultiphaseDenseFiniteDifferenceImageFilter` (single phase) driven by a
//! `ScalarChanAndVeseLevelSetFunction` over a `RegionBasedLevelSetFunction`.
//!
//! ## Per-pixel update (`RegionBasedLevelSetFunction::ComputeUpdate`)
//!
//! ```text
//! update = δ(φ) · ( μ·κ + λ₁(u₀ − c_in)² − λ₂(u₀ − c_out)² − areaWeight )
//! ```
//!
//! with (ITK uses the Heaviside of −φ throughout, so the **inside** region is φ<0):
//! - `δ(φ) = (1/π)(1/ε)/(1 + (φ/ε)²)` (atan-regularized Dirac),
//! - `c_in  = Σ u₀·(1−H(φ)) / Σ(1−H(φ))` — mean over φ<0 (the output region),
//! - `c_out = Σ u₀·H(φ) / Σ H(φ)` — mean over φ>0,
//! - `κ = [Σ_{i≠j}(−φ_i φ_j φ_ij + φ_jj φ_i²)] / |∇φ|³` (ITK `ComputeCurvature`,
//!   falling back to `…/(1+|∇φ|²)` when `|∇φ| ≤ eps`).
//!
//! ## Time step and reinitialization
//!
//! The dense filter **ignores** its computed time step and applies the ITK
//! hardcoded constant `Δt = 0.08` (`itkMultiphaseDenseFiniteDifferenceImageFilter`
//! `CalculateChange`). After each `φ += 0.08·update`, the level set is
//! **reinitialized every iteration** (`ReinitializeCounter = 1`): threshold at
//! φ ≤ 0 then [`crate::SignedMaurerDistanceMapImageFilter`] (`insideIsPositive = false`),
//! so φ becomes the signed Maurer distance to the region boundary.
//!
//! Output is the **binary segmentation** `(φ < 0) → 1` (matching sitk), not the
//! level set.
//!
//! Validated bit-exact (0 pixel mismatches across iterations 1–8) against
//! `sitk.ScalarChanAndVeseDenseLevelSet`.
//!
//! ## References
//! - Chan, T. F. & Vese, L. A. (2001). "Active Contours Without Edges."
//!   *IEEE TIP*, 10(2), 266–277.
//! - ITK `itkScalarChanAndVeseLevelSetFunction.hxx`,
//!   `itkRegionBasedLevelSetFunction.hxx`,
//!   `itkMultiphaseDenseFiniteDifferenceImageFilter.hxx`.

use std::f64::consts::PI;

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::signed_maurer_core;

/// ITK hardcoded dense-filter time step (`CalculateChange` constant).
const DT: f64 = 0.08;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Scalar Chan-Vese dense level set filter (faithful ITK port).
///
/// Returns the **binary segmentation** (`1.0` where φ < 0, `0.0` elsewhere),
/// bit-exact to `sitk.ScalarChanAndVeseDenseLevelSet`.
///
/// # Default parameters (match ITK)
///
/// | Field | Default |
/// |-------|---------|
/// | `number_of_iterations` | 100 |
/// | `lambda1` | 1.0 |
/// | `lambda2` | 1.0 |
/// | `mu` (curvature weight) | 1.0 |
/// | `nu` (area weight) | 0.0 |
/// | `epsilon` | 1.0 |
#[derive(Debug, Clone)]
pub struct ScalarChanAndVeseDenseLevelSet {
    /// Maximum number of PDE evolution iterations.
    pub number_of_iterations: usize,
    /// Data fidelity weight λ₁ for the inside region.
    pub lambda1: f32,
    /// Data fidelity weight λ₂ for the outside region.
    pub lambda2: f32,
    /// Curvature (length) penalty weight μ (ITK `CurvatureWeight`).
    pub mu: f32,
    /// Area penalty weight (ITK `AreaWeight`); subtracted from the data term.
    pub nu: f32,
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
            epsilon: 1.0,
        }
    }
}

impl ScalarChanAndVeseDenseLevelSet {
    /// Evolve a level set under the Chan-Vese energy and return the binary mask.
    ///
    /// # Arguments
    /// - `initial_level_set`: φ₀ with **φ < 0 inside** the region of interest.
    /// - `feature_image`: u₀ — the scalar image driving the data-fidelity energy.
    ///   Must have the same shape as `initial_level_set`.
    ///
    /// # Returns
    /// The binary segmentation (`1.0` where φ < 0).
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

        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();
        let feat_f64: Vec<f64> = feat.iter().map(|&v| v as f64).collect();

        self.evolve(&mut phi, &feat_f64, dims);

        // Output: binary segmentation (φ < 0 → 1).
        let result: Vec<f32> = phi.iter().map(|&v| if v < 0.0 { 1.0 } else { 0.0 }).collect();
        Ok(rebuild(result, dims, initial_level_set))
    }
}

// ── Core PDE ───────────────────────────────────────────────────────────────────

impl ScalarChanAndVeseDenseLevelSet {
    fn evolve(&self, phi: &mut [f64], feat: &[f64], dims: [usize; 3]) {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let eps = self.epsilon as f64;
        let mu = self.mu as f64;
        let area = self.nu as f64;
        let lam1 = self.lambda1 as f64;
        let lam2 = self.lambda2 as f64;
        let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
        // Clamped (Neumann) accessor for the derivative stencils.
        let g = |phi: &[f64], z: isize, y: isize, x: isize| -> f64 {
            let zc = z.clamp(0, nz as isize - 1) as usize;
            let yc = y.clamp(0, ny as isize - 1) as usize;
            let xc = x.clamp(0, nx as isize - 1) as usize;
            phi[idx(zc, yc, xc)]
        };

        let mut update = vec![0.0_f64; n];
        for _ in 0..self.number_of_iterations {
            // ── Region means (Heaviside of −φ: inside = φ<0) ─────────────────
            let (c_in, c_out) = region_means(feat, phi, eps);

            // ── Per-pixel update ─────────────────────────────────────────────
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let f = idx(z, y, x);
                        let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                        let c = phi[f];
                        let dh = dirac(c, eps);
                        let curv = if nz == 1 {
                            curvature_2d(phi, &g, zi, yi, xi, c)
                        } else {
                            curvature_3d(phi, &g, zi, yi, xi, c)
                        };
                        let din = feat[f] - c_in;
                        let dout = feat[f] - c_out;
                        let glob = lam1 * din * din - lam2 * dout * dout - area;
                        update[f] = dh * (mu * curv + glob);
                    }
                }
            }

            // ── φ += 0.08·update, then per-iteration Maurer reinit ────────────
            for f in 0..n {
                phi[f] += DT * update[f];
            }
            reinitialize(phi, dims);
        }
    }
}

/// Threshold at φ ≤ 0 and replace φ with the signed Maurer distance to the region
/// boundary (`insideIsPositive = false`). A degenerate all-inside / all-outside
/// field has no border, so it is left unchanged.
fn reinitialize(phi: &mut [f64], dims: [usize; 3]) {
    let inside: Vec<bool> = phi.iter().map(|&v| v <= 0.0).collect();
    let any_in = inside.contains(&true);
    let any_out = inside.contains(&false);
    if !any_in || !any_out {
        return;
    }
    let sd = signed_maurer_core(&inside, dims, [1.0, 1.0, 1.0], false, false);
    for (p, &v) in phi.iter_mut().zip(sd.iter()) {
        *p = v as f64;
    }
}

/// ITK `ComputeCurvature` (2-D): `[φ_yy φ_x² + φ_xx φ_y² − 2 φ_x φ_y φ_xy] / |∇φ|³`.
#[inline]
fn curvature_2d(
    phi: &[f64],
    g: &impl Fn(&[f64], isize, isize, isize) -> f64,
    z: isize,
    y: isize,
    x: isize,
    c: f64,
) -> f64 {
    let fx = 0.5 * (g(phi, z, y, x + 1) - g(phi, z, y, x - 1));
    let fy = 0.5 * (g(phi, z, y + 1, x) - g(phi, z, y - 1, x));
    let fxx = g(phi, z, y, x + 1) - 2.0 * c + g(phi, z, y, x - 1);
    let fyy = g(phi, z, y + 1, x) - 2.0 * c + g(phi, z, y - 1, x);
    let fxy = 0.25
        * (g(phi, z, y - 1, x - 1) - g(phi, z, y - 1, x + 1) - g(phi, z, y + 1, x - 1)
            + g(phi, z, y + 1, x + 1));
    let num = fyy * fx * fx + fxx * fy * fy - 2.0 * fx * fy * fxy;
    let gms = fx * fx + fy * fy;
    let gm = gms.sqrt();
    if gm > f64::EPSILON {
        num / (gm * gm * gm)
    } else {
        num / (1.0 + gms)
    }
}

/// ITK `ComputeCurvature` (3-D), the full `Σ_{i≠j}` form over `|∇φ|³`.
#[inline]
fn curvature_3d(
    phi: &[f64],
    g: &impl Fn(&[f64], isize, isize, isize) -> f64,
    z: isize,
    y: isize,
    x: isize,
    c: f64,
) -> f64 {
    let fx = 0.5 * (g(phi, z, y, x + 1) - g(phi, z, y, x - 1));
    let fy = 0.5 * (g(phi, z, y + 1, x) - g(phi, z, y - 1, x));
    let fz = 0.5 * (g(phi, z + 1, y, x) - g(phi, z - 1, y, x));
    let fxx = g(phi, z, y, x + 1) - 2.0 * c + g(phi, z, y, x - 1);
    let fyy = g(phi, z, y + 1, x) - 2.0 * c + g(phi, z, y - 1, x);
    let fzz = g(phi, z + 1, y, x) - 2.0 * c + g(phi, z - 1, y, x);
    let fxy = 0.25
        * (g(phi, z, y - 1, x - 1) - g(phi, z, y - 1, x + 1) - g(phi, z, y + 1, x - 1)
            + g(phi, z, y + 1, x + 1));
    let fxz = 0.25
        * (g(phi, z - 1, y, x - 1) - g(phi, z - 1, y, x + 1) - g(phi, z + 1, y, x - 1)
            + g(phi, z + 1, y, x + 1));
    let fyz = 0.25
        * (g(phi, z - 1, y - 1, x) - g(phi, z - 1, y + 1, x) - g(phi, z + 1, y - 1, x)
            + g(phi, z + 1, y + 1, x));
    // Σ_{i≠j}(−φ_i φ_j φ_ij + φ_jj φ_i²)
    let num = fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) + fz * fz * (fxx + fyy)
        - 2.0 * fx * fy * fxy
        - 2.0 * fx * fz * fxz
        - 2.0 * fy * fz * fyz;
    let gms = fx * fx + fy * fy + fz * fz;
    let gm = gms.sqrt();
    if gm > f64::EPSILON {
        num / (gm * gm * gm)
    } else {
        num / (1.0 + gms)
    }
}

// ── Region mean computation ───────────────────────────────────────────────────

/// Compute (c_in, c_out) with the ITK Heaviside-of-(−φ) convention:
///
/// ```text
/// c_in  = Σ u₀ · (1 − H(φ)) / Σ (1 − H(φ))   (inside, φ < 0 = output region)
/// c_out = Σ u₀ · H(φ)       / Σ H(φ)           (outside, φ > 0)
/// ```
fn region_means(feat: &[f64], phi: &[f64], eps: f64) -> (f64, f64) {
    let mut sum_in = 0.0_f64;
    let mut sum_u_in = 0.0_f64;
    let mut sum_out = 0.0_f64;
    let mut sum_u_out = 0.0_f64;

    for i in 0..feat.len() {
        let h = heaviside(phi[i], eps);
        let omh = 1.0 - h;
        sum_in += omh;
        sum_u_in += feat[i] * omh;
        sum_out += h;
        sum_u_out += feat[i] * h;
    }

    let c_in = if sum_in > 1e-15 { sum_u_in / sum_in } else { 0.0 };
    let c_out = if sum_out > 1e-15 { sum_u_out / sum_out } else { 0.0 };
    (c_in, c_out)
}

// ── Inline regularised Heaviside and Dirac ────────────────────────────────────

/// `H_ε(z) = 0.5 · (1 + (2/π)·arctan(z/ε))` (ITK atan-regularized step).
#[inline]
fn heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / PI) * (z / eps).atan())
}

/// `δ_ε(z) = (1/π)·(1/ε)/(1 + (z/ε)²)` — `H_ε'`, the ITK atan Dirac.
#[inline]
fn dirac(z: f64, eps: f64) -> f64 {
    (1.0 / PI) * (1.0 / eps) / (1.0 + (z / eps) * (z / eps))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_scalar_chan_and_vese.rs"]
mod tests;

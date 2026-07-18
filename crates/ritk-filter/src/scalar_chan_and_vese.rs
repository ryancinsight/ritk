//! Scalar Chan-Vese dense level set (ITK `ScalarChanAndVeseDenseLevelSetImageFilter`).
//!
//! # Mathematical Specification
//!
//! Ports `sitk.ScalarChanAndVeseDenseLevelSet` â€” a
//! `MultiphaseDenseFiniteDifferenceImageFilter` (single phase) driven by a
//! `ScalarChanAndVeseLevelSetFunction` over a `RegionBasedLevelSetFunction`.
//!
//! ## Per-pixel update (`RegionBasedLevelSetFunction::ComputeUpdate`)
//!
//! ```text
//! update = Î´(Ï†) Â· ( Î¼Â·Îº + Î»â‚(uâ‚€ âˆ’ c_in)Â² âˆ’ Î»â‚‚(uâ‚€ âˆ’ c_out)Â² âˆ’ areaWeight )
//! ```
//!
//! with (ITK uses the Heaviside of âˆ’Ï† throughout, so the **inside** region is Ï†<0):
//! - `Î´(Ï†) = (1/Ï€)(1/Îµ)/(1 + (Ï†/Îµ)Â²)` (atan-regularized Dirac),
//! - `c_in  = Î£ uâ‚€Â·(1âˆ’H(Ï†)) / Î£(1âˆ’H(Ï†))` â€” mean over Ï†<0 (the output region),
//! - `c_out = Î£ uâ‚€Â·H(Ï†) / Î£ H(Ï†)` â€” mean over Ï†>0,
//! - `Îº = [Î£_{iâ‰ j}(âˆ’Ï†_i Ï†_j Ï†_ij + Ï†_jj Ï†_iÂ²)] / |âˆ‡Ï†|Â³` (ITK `ComputeCurvature`,
//!   falling back to `â€¦/(1+|âˆ‡Ï†|Â²)` when `|âˆ‡Ï†| â‰¤ eps`).
//!
//! ## Time step and reinitialization
//!
//! The dense filter **ignores** its computed time step and applies the ITK
//! hardcoded constant `Î”t = 0.08` (`itkMultiphaseDenseFiniteDifferenceImageFilter`
//! `CalculateChange`). After each `Ï† += 0.08Â·update`, the level set is
//! **reinitialized every iteration** (`ReinitializeCounter = 1`): threshold at
//! Ï† â‰¤ 0 then [`crate::SignedMaurerDistanceMapImageFilter`] (`insideIsPositive = false`),
//! so Ï† becomes the signed Maurer distance to the region boundary.
//!
//! Output is the **binary segmentation** `(Ï† < 0) â†’ 1` (matching sitk), not the
//! level set.
//!
//! Validated bit-exact (0 pixel mismatches across iterations 1â€“8) against
//! `sitk.ScalarChanAndVeseDenseLevelSet`.
//!
//! ## References
//! - Chan, T. F. & Vese, L. A. (2001). "Active Contours Without Edges."
//!   *IEEE TIP*, 10(2), 266â€“277.
//! - ITK `itkScalarChanAndVeseLevelSetFunction.hxx`,
//!   `itkRegionBasedLevelSetFunction.hxx`,
//!   `itkMultiphaseDenseFiniteDifferenceImageFilter.hxx`.

use std::f64::consts::PI;

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::signed_maurer_core;

/// ITK hardcoded dense-filter time step (`CalculateChange` constant).
const DT: f64 = 0.08;

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Scalar Chan-Vese dense level set filter (faithful ITK port).
///
/// Returns the **binary segmentation** (`1.0` where Ï† < 0, `0.0` elsewhere),
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
    /// Data fidelity weight Î»â‚ for the inside region.
    pub lambda1: f32,
    /// Data fidelity weight Î»â‚‚ for the outside region.
    pub lambda2: f32,
    /// Curvature (length) penalty weight Î¼ (ITK `CurvatureWeight`).
    pub mu: f32,
    /// Area penalty weight (ITK `AreaWeight`); subtracted from the data term.
    pub nu: f32,
    /// Regularisation width Îµ for Heaviside and Dirac approximations.
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
    /// - `initial_level_set`: Ï†â‚€ with **Ï† < 0 inside** the region of interest.
    /// - `feature_image`: uâ‚€ â€” the scalar image driving the data-fidelity energy.
    ///   Must have the same shape as `initial_level_set`.
    ///
    /// # Returns
    /// The binary segmentation (`1.0` where Ï† < 0).
    ///
    /// # Errors
    /// Returns `Err` if tensor extraction fails or if the image shapes differ.
    pub fn apply<B: Backend>(
        &self,
        initial_level_set: &Image<f32, B, 3>,
        feature_image: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
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

        // Output: binary segmentation (Ï† < 0 â†’ 1).
        let result: Vec<f32> = phi
            .iter()
            .map(|&v| if v < 0.0 { 1.0 } else { 0.0 })
            .collect();
        Ok(rebuild(result, dims, initial_level_set))
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        initial_level_set: &ritk_image::native::Image<f32, B, 3>,
        feature_image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = initial_level_set.shape();
        if dims != feature_image.shape() {
            anyhow::bail!(
                "initial_level_set shape {:?} and feature_image shape {:?} must match",
                dims,
                feature_image.shape()
            );
        }

        let (phi_init, _) = ritk_tensor_ops::native::extract_image_vec(initial_level_set)?;
        let (feat, _) = ritk_tensor_ops::native::extract_image_vec(feature_image)?;

        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();
        let feat_f64: Vec<f64> = feat.iter().map(|&v| v as f64).collect();

        self.evolve(&mut phi, &feat_f64, dims);

        let result: Vec<f32> = phi
            .iter()
            .map(|&v| if v < 0.0 { 1.0 } else { 0.0 })
            .collect();
        crate::native_support::rebuild_image(result, dims, initial_level_set, backend)
    }
}

// â”€â”€ Core PDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            // â”€â”€ Region means (Heaviside of âˆ’Ï†: inside = Ï†<0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            let (c_in, c_out) = region_means(feat, phi, eps);

            // â”€â”€ Per-pixel update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            // â”€â”€ Ï† += 0.08Â·update, then per-iteration Maurer reinit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            for f in 0..n {
                phi[f] += DT * update[f];
            }
            reinitialize(phi, dims);
        }
    }
}

/// Threshold at Ï† â‰¤ 0 and replace Ï† with the signed Maurer distance to the region
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

/// ITK `ComputeCurvature` (2-D): `[Ï†_yy Ï†_xÂ² + Ï†_xx Ï†_yÂ² âˆ’ 2 Ï†_x Ï†_y Ï†_xy] / |âˆ‡Ï†|Â³`.
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

/// ITK `ComputeCurvature` (3-D), the full `Î£_{iâ‰ j}` form over `|âˆ‡Ï†|Â³`.
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
    // Î£_{iâ‰ j}(âˆ’Ï†_i Ï†_j Ï†_ij + Ï†_jj Ï†_iÂ²)
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

// â”€â”€ Region mean computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute (c_in, c_out) with the ITK Heaviside-of-(âˆ’Ï†) convention:
///
/// ```text
/// c_in  = Î£ uâ‚€ Â· (1 âˆ’ H(Ï†)) / Î£ (1 âˆ’ H(Ï†))   (inside, Ï† < 0 = output region)
/// c_out = Î£ uâ‚€ Â· H(Ï†)       / Î£ H(Ï†)           (outside, Ï† > 0)
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

    let c_in = if sum_in > 1e-15 {
        sum_u_in / sum_in
    } else {
        0.0
    };
    let c_out = if sum_out > 1e-15 {
        sum_u_out / sum_out
    } else {
        0.0
    };
    (c_in, c_out)
}

// â”€â”€ Inline regularised Heaviside and Dirac â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// `H_Îµ(z) = 0.5 Â· (1 + (2/Ï€)Â·arctan(z/Îµ))` (ITK atan-regularized step).
#[inline]
fn heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / PI) * (z / eps).atan())
}

/// `Î´_Îµ(z) = (1/Ï€)Â·(1/Îµ)/(1 + (z/Îµ)Â²)` â€” `H_Îµ'`, the ITK atan Dirac.
#[inline]
fn dirac(z: f64, eps: f64) -> f64 {
    (1.0 / PI) * (1.0 / eps) / (1.0 + (z / eps) * (z / eps))
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_scalar_chan_and_vese.rs"]
mod tests;

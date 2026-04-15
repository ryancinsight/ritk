//! Diffeomorphic Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! The Diffeomorphic Demons algorithm (Vercauteren et al. 2009) maintains a
//! **stationary velocity field** `v : ℤ³ → ℝ³` and produces the displacement
//! field as its exponential map `φ = exp(v)` via the scaling-and-squaring
//! algorithm.  This guarantees that `φ` is a diffeomorphism (invertible with
//! smooth inverse), unlike the classic Thirion formulation which accumulates
//! displacements directly.
//!
//! **Per-iteration update:**
//! 1. Compute `φ = exp(v)` via scaling-and-squaring (`n_squarings` steps).
//! 2. Warp moving with `φ` → `M_w`.
//! 3. Compute Thirion forces `u` from `(F, M_w, ∇F)`.
//! 4. BCH velocity update (first-order): `v ← v + u`.
//! 5. Diffusive regularisation: `v ← G_{σ_diff} ∗ v`.
//! 6. Compute MSE = mean((F − M_w)²).
//!
//! The first-order BCH approximation `log(exp(v) ∘ exp(u)) ≈ v + u` is valid
//! when `|u|` is small relative to `|v|`, which holds throughout registration
//! because the force step is clamped by `max_step_length`.
//!
//! # References
//! - Vercauteren, T., Pennec, X., Perchant, A. & Ayache, N. (2009).
//!   Diffeomorphic Demons: Efficient non-parametric image registration.
//!   *NeuroImage* 45(S1):S61–S72.
//! - Arsigny, V., Commowick, O., Pennec, X. & Ayache, N. (2006). A
//!   Log-Euclidean Framework for Statistics on Diffeomorphisms. *MICCAI*.

use super::thirion::{thirion_forces, DemonsConfig, DemonsResult};
use crate::deformable_field_ops::{gaussian_smooth_inplace, scaling_and_squaring, warp_image};
use crate::error::RegistrationError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Diffeomorphic Demons registration using stationary velocity fields.
///
/// Produces topologically correct (invertible) deformation fields by
/// representing the transformation as the exponential map of a velocity field.
#[derive(Debug, Clone)]
pub struct DiffeomorphicDemonsRegistration {
    /// Algorithm configuration (shared with Thirion Demons).
    pub config: DemonsConfig,
    /// Number of scaling-and-squaring steps for computing `exp(v)`.
    /// Standard value: 6 (2⁶ = 64 integration steps).
    pub n_squarings: usize,
}

impl DiffeomorphicDemonsRegistration {
    /// Create a registration instance with the given configuration.
    ///
    /// `n_squarings` defaults to 6.
    pub fn new(config: DemonsConfig) -> Self {
        Self {
            config,
            n_squarings: 6,
        }
    }

    /// Create with a custom number of squaring steps.
    pub fn with_squarings(config: DemonsConfig, n_squarings: usize) -> Self {
        Self {
            config,
            n_squarings,
        }
    }

    /// Register `moving` to `fixed` using a stationary velocity field.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if image lengths are inconsistent with `dims`.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<DemonsResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(),
                n
            )));
        }

        // Initialise stationary velocity field to zero.
        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];

        // Pre-compute fixed-image gradient (constant across iterations).
        let (grad_z, grad_y, grad_x) =
            crate::deformable_field_ops::compute_gradient(fixed, dims, spacing);

        let mut final_mse = compute_mse_direct(fixed, moving, &vel_z, &vel_y, &vel_x, dims);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Compute displacement field φ = exp(v) via scaling-and-squaring.
            let (phi_z, phi_y, phi_x) =
                scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, self.n_squarings);

            // 2. Warp moving with φ.
            let m_warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

            // 3. Compute Thirion optical-flow forces from (fixed, M_w, ∇F).
            let (fz, fy, fx) = thirion_forces(
                fixed,
                &m_warped,
                &grad_z,
                &grad_y,
                &grad_x,
                self.config.max_step_length,
            );

            // 4. BCH first-order velocity update: v ← v + u.
            for i in 0..n {
                vel_z[i] += fz[i];
                vel_y[i] += fy[i];
                vel_x[i] += fx[i];
            }

            // 5. Diffusive regularisation of the velocity field.
            if self.config.sigma_diffusion > 0.0 {
                gaussian_smooth_inplace(&mut vel_z, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_y, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_x, dims, self.config.sigma_diffusion);
            }

            // 6. Update MSE using the current velocity field.
            final_mse = compute_mse_direct(fixed, moving, &vel_z, &vel_y, &vel_x, dims);
        }

        // Final warp using the converged velocity field.
        let (phi_z, phi_y, phi_x) =
            scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, self.n_squarings);
        let warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

        Ok(DemonsResult {
            warped,
            disp_z: phi_z,
            disp_y: phi_y,
            disp_x: phi_x,
            final_mse,
            num_iterations: iter,
        })
    }

    /// Compute the inverse displacement field of a registration result.
    ///
    /// # Mathematical Basis
    ///
    /// `DemonsResult` stores `disp = φ = exp(v)` (the forward displacement).
    /// Because the velocity field `v` is not retained after registration,
    /// the inverse is computed by the fixed-point iterative method
    /// (Christensen & Johnson 2001) applied directly to the stored displacement
    /// field.  For SVF results this is equivalent to `exp(−v)` when the
    /// iteration converges fully (Lipschitz constant of the stored field < 1).
    ///
    /// # Arguments
    ///
    /// - `result` — output of [`DiffeomorphicDemonsRegistration::register`].
    /// - `dims`   — volume dimensions `[nz, ny, nx]` (must match registration).
    ///
    /// # Returns
    ///
    /// `(inv_disp_z, inv_disp_y, inv_disp_x)` — inverse displacement components
    /// in voxel units.
    pub fn invert_result(
        &self,
        result: &DemonsResult,
        dims: [usize; 3],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        use super::inverse::{invert_displacement_field, InverseFieldConfig};
        let config = InverseFieldConfig::default();
        let (inv_z, inv_y, inv_x, _) = invert_displacement_field(
            &result.disp_z,
            &result.disp_y,
            &result.disp_x,
            dims,
            &config,
        );
        (inv_z, inv_y, inv_x)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Compute MSE after applying `exp(v)` to warp the moving image.
fn compute_mse_direct(
    fixed: &[f32],
    moving: &[f32],
    vel_z: &[f32],
    vel_y: &[f32],
    vel_x: &[f32],
    dims: [usize; 3],
) -> f64 {
    let (phi_z, phi_y, phi_x) = scaling_and_squaring(vel_z, vel_y, vel_x, dims, 6);
    let warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);
    fixed
        .iter()
        .zip(warped.iter())
        .map(|(&f, &m)| ((f - m) as f64).powi(2))
        .sum::<f64>()
        / fixed.len() as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = fi % nx;
                let iy = (fi / nx) % ny;
                let iz = fi / (ny * nx);
                let sz = std::f32::consts::PI * iz as f32 / nz as f32;
                let sy = std::f32::consts::PI * iy as f32 / ny as f32;
                sz.sin() * sy.cos() * (ix as f32 + 1.0)
            })
            .collect()
    }

    fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let mut out = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if ix >= shift {
                        let src = iz * ny * nx + iy * nx + (ix - shift);
                        out[iz * ny * nx + iy * nx + ix] = data[src];
                    }
                }
            }
        }
        out
    }

    /// Registering identical images must yield near-zero MSE.
    #[test]
    fn identity_registration_near_zero_mse() {
        let dims = [8usize, 8, 8];
        let image = make_test_image(dims);
        let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
            max_iterations: 20,
            ..Default::default()
        });
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_mse < 1e-3,
            "identity MSE should be < 1e-3, got {}",
            result.final_mse
        );
    }

    /// MSE must decrease after registration of translated images.
    #[test]
    fn registration_reduces_mse() {
        let dims = [10usize, 10, 14];
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let initial_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
            max_iterations: 50,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        assert!(
            result.final_mse < initial_mse,
            "MSE should decrease: initial={initial_mse:.6} final={:.6}",
            result.final_mse
        );
        assert!(
            result.final_mse < initial_mse * 0.5,
            "MSE should decrease by ≥50 %: initial={initial_mse:.6} final={:.6}",
            result.final_mse
        );
    }

    /// All components of the final displacement field must be finite.
    #[test]
    fn displacement_field_finite() {
        let dims = [6usize, 6, 8];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 1);
        let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig {
            max_iterations: 15,
            ..Default::default()
        });
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();
        for (&dz, (&dy, &dx)) in result
            .disp_z
            .iter()
            .zip(result.disp_y.iter().zip(result.disp_x.iter()))
        {
            assert!(dz.is_finite(), "disp_z non-finite: {dz}");
            assert!(dy.is_finite(), "disp_y non-finite: {dy}");
            assert!(dx.is_finite(), "disp_x non-finite: {dx}");
        }
    }

    /// Error is returned for length-mismatched inputs.
    #[test]
    fn mismatched_lengths_returns_error() {
        let dims = [4usize, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 4];
        let moving = vec![0.0_f32; 4 * 4 * 5];
        let reg = DiffeomorphicDemonsRegistration::new(DemonsConfig::default());
        assert!(
            reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
                .is_err(),
            "should return error for mismatched lengths"
        );
    }

    /// The scaling-and-squaring exponential map of a zero velocity field is zero.
    #[test]
    fn zero_velocity_zero_displacement() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let image = make_test_image(dims);
        let zero = vec![0.0_f32; n];
        // With zero velocity field, the warp is identity and MSE should be zero.
        let mse = compute_mse_direct(&image, &image, &zero, &zero, &zero, dims);
        assert!(mse < 1e-10, "zero velocity should give zero MSE, got {mse}");
    }
}

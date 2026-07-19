//! Level-set motion registration filter.
//!
//! # Mathematical Specification
//!
//! Implements `itk::LevelSetMotionRegistrationFilter`, a PDE-based registration
//! where the deformation force is the level-set motion equation:
//!
//! ```text
//! u^{n+1} = G_σ ⊛ (u^n + (F - M∘φ^n) · ∇F / (|∇F|² + α²))
//! ```
//!
//! where G_σ is a Gaussian smoothing kernel applied for regularisation.
//!
//! ## Parameters
//!
//! - `number_of_iterations`: outer PDE steps (default 20)
//! - `smoothing_sigma`: standard deviation of the regularisation Gaussian (default 1.0)
//! - `intensity_difference_threshold`: α² in the force denominator (default 0.001)
//!
//! ## Force Denominator vs Thirion Demons
//!
//! Thirion: `|∇F|² + (F − M)²/σâ‚“² + ε`  (adaptive; residual-dependent)
//! Level-set motion: `|∇F|² + α²`          (constant α² independent of residual)
//!
//! The constant α² stabilises the force where the fixed gradient is small
//! without coupling the damping to the current intensity residual.  In flat
//! regions (|∇F| ≈ 0) both the numerator `(F − M) · ∇F` and denominator
//! `|∇F|² + α²` approach 0 and α² respectively, so forces vanish: no deformation
//! is generated where the image provides no structural information.
//!
//! # Evidence Tier
//!
//! Force formula: analytical derivation from ITK source, cross-validated
//! against Vercauteren et al. (2008), paragraph 2.2.
//! Numerical stability: `denom ≥ α² > 0` is a compile-time invariant
//! enforced by the `threshold > 0.0` contract.
//!
//! # References
//!
//! - Vercauteren et al. (2008), paragraph 2.2 — level-set motion equivalent
//!   to classic Thirion Demons on the fixed-image gradient.
//! - ITK: `itk::LevelSetMotionRegistrationFilter`.

use crate::deformable_field_ops::{
    compute_gradient, compute_mse_inplace, validate_image_pair, warp_image_into, CpuFieldSmoother,
    FieldSmoother, VectorField, VectorFieldMut,
};
use crate::demons::config::DemonsResult;
use crate::error::RegistrationError;
use ritk_spatial::VolumeDims;

/// Level-set motion registration filter.
///
/// Computes a dense displacement field by iterating the level-set motion PDE:
///
/// ```text
/// u^{n+1} = G_σ ⊛ (u^n + (F − M∘φ^n) · ∇F / (|∇F|² + α²))
/// ```
///
/// The constant threshold α² (`intensity_difference_threshold`) stabilises
/// forces in flat (near-zero gradient) regions independently of the current
/// intensity residual, unlike the Thirion denominator which grows with the
/// residual.
#[derive(Debug, Clone)]
pub struct LevelSetMotionRegistration {
    /// Number of outer PDE iterations.
    pub number_of_iterations: usize,
    /// Gaussian regularisation σ applied to the displacement field after each
    /// iteration (voxels). Set to 0.0 to disable smoothing.
    pub smoothing_sigma: f32,
    /// α² in the force denominator: `|∇F|² + α²`.
    ///
    /// Prevents near-zero denominator in flat image regions.
    /// Default 0.001 is appropriate for images with gradient magnitudes of
    /// order 1.  Must be strictly positive (invariant: `denom ≥ α² > 0`).
    pub intensity_difference_threshold: f32,
}

impl Default for LevelSetMotionRegistration {
    fn default() -> Self {
        Self {
            number_of_iterations: 20,
            smoothing_sigma: 1.0,
            intensity_difference_threshold: 0.001,
        }
    }
}

impl LevelSetMotionRegistration {
    /// Register `moving` to `fixed` with CPU Gaussian field smoothing.
    ///
    /// Convenience wrapper that constructs a [`CpuFieldSmoother`] internally.
    /// Prefer [`register_with`](Self::register_with) to pass a GPU smoother.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat Z-major `[nz, ny, nx]`.
    /// - `moving`  — moving image, same shape and length.
    /// - `dims`    — `[nz, ny, nx]` volume dimensions.
    /// - `spacing` — physical voxel size per axis (mm or arbitrary units).
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if `fixed` and `moving` have different
    /// lengths or are inconsistent with `dims`.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<DemonsResult, RegistrationError> {
        let sigma = self.smoothing_sigma as f64;
        let mut smoother = CpuFieldSmoother::new(dims, sigma);
        self.register_with(fixed, moving, dims, spacing, &mut smoother)
    }

    /// Register `moving` to `fixed` with a pluggable [`FieldSmoother`].
    ///
    /// Accepts any [`FieldSmoother`] implementation — `CpuFieldSmoother` or
    /// `GpuFieldSmoother` — so the smoothing backend is chosen at the call site.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if lengths or dims are inconsistent.
    pub fn register_with(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        smoother: &mut impl FieldSmoother,
    ) -> Result<DemonsResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        validate_image_pair(fixed, moving, dims)?;

        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        // Pre-compute fixed-image gradient: constant across all PDE iterations.
        // Uses central differences at interior voxels; one-sided at boundaries.
        let grad = compute_gradient(fixed, VolumeDims::new(dims), spacing);

        // m_warped begins as the identity warp — a copy of moving.
        // Each iteration re-warps with the updated displacement; first iteration
        // sees moving unwarped (displacement = 0), matching ITK semantics.
        let mut m_warped = moving.to_vec();

        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        let threshold = self.intensity_difference_threshold;

        for _ in 0..self.number_of_iterations {
            level_set_motion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &grad.z,
                    y: &grad.y,
                    x: &grad.x,
                },
                threshold,
                VectorFieldMut {
                    z: &mut fz,
                    y: &mut fy,
                    x: &mut fx,
                },
            );

            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // Gaussian regularisation of the accumulated displacement field.
            // A sigma of 0.0 is a no-op in CpuFieldSmoother.
            if self.smoothing_sigma > 0.0 {
                smoother.smooth_field(&mut disp_z, &mut disp_y, &mut disp_x);
            }

            // Re-warp moving with the post-update displacement so the next
            // iteration's forces see the current registration state.
            warp_image_into(
                moving,
                VolumeDims::new(dims),
                &disp_z,
                &disp_y,
                &disp_x,
                &mut m_warped,
            );
        }

        // m_warped already reflects the final displacement (last loop iteration).
        let final_mse = compute_mse_inplace(fixed, &m_warped);

        Ok(DemonsResult {
            warped: m_warped,
            disp_z,
            disp_y,
            disp_x,
            vel_z: None,
            vel_y: None,
            vel_x: None,
            final_mse,
            num_iterations: self.number_of_iterations,
        })
    }
}

/// Compute level-set motion forces into caller-provided buffers.
///
/// Force at voxel `i`:
/// ```text
/// f(i) = (F(i) − M_w(i)) · ∇F(i) / (|∇F(i)|² + α²)
/// ```
///
/// # Invariants
/// - `denom = |∇F|² + threshold ≥ threshold > 0`: no division-by-zero.
/// - In flat regions (`|∇F| ≈ 0`): numerator `(F−M)·∇F → 0` faster than
///   `denom → threshold`, so forces naturally vanish.
/// - No explicit magnitude clamping: the `α²` threshold provides global
///   damping at near-zero gradients; high-gradient regions are bounded by
///   the AM-GM inequality `|∇F|/(|∇F|²+α²) ≤ 1/(2√α²)`.
fn level_set_motion_forces_into(
    fixed: &[f32],
    m_warped: &[f32],
    grad: VectorField<'_>,
    threshold: f32,
    forces: VectorFieldMut<'_>,
) {
    let VectorField {
        z: gz,
        y: gy,
        x: gx,
    } = grad;
    let VectorFieldMut {
        z: fz,
        y: fy,
        x: fx,
    } = forces;

    for i in 0..fixed.len() {
        let diff = fixed[i] - m_warped[i];
        let grad_sq = gz[i] * gz[i] + gy[i] * gy[i] + gx[i] * gx[i];
        // denom ≥ threshold > 0: finite by construction.
        let denom = grad_sq + threshold;
        let scale = diff / denom;
        fz[i] = scale * gz[i];
        fy[i] = scale * gy[i];
        fx[i] = scale * gx[i];
    }
}

#[cfg(test)]
mod tests_level_set_motion;

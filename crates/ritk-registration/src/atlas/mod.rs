//! Groupwise atlas construction via iterative template building.
//!
//! # Mathematical Specification
//!
//! Given N subject images {I₁, ..., Iₙ}, the algorithm constructs a
//! population-specific template T by iterating:
//!
//! 1. Initialize T⁰ = (1/N) Σᵢ Iᵢ (voxel-wise mean of all subjects).
//! 2. For iteration k = 1, ..., K:
//!    a. Register each Iᵢ to T^{k−1} via Multi-Resolution SyN → velocity φᵢ,
//!    warped image Iᵢ ∘ φᵢ⁻¹.
//!    b. Compute T̃^k = (1/N) Σᵢ Iᵢ ∘ φᵢ⁻¹ (mean of warped subjects).
//!    c. Compute mean forward velocity: v̄ = (1/N) Σᵢ vᵢ (component-wise).
//!    d. Template sharpening: T^k = warp(T̃^k, exp(−G_σ ∗ v̄)) where G_σ is
//!    Gaussian smoothing applied to the negated mean velocity before
//!    exponentiation via scaling-and-squaring.  This removes mean-drift
//!    bias so the template lies at the geometric centre of the population
//!    in diffeomorphism space.
//!    e. Convergence: stop when ‖T^k − T^{k−1}‖₂ / √n < threshold.
//!
//! # References
//!
//! - Avants, B. B. & Gee, J. C. (2004). Geodesic estimation for large
//!   deformation anatomical shape averaging and interpolation. *NeuroImage*
//!   23:S139–S150.
//! - Guimond, A., Meunier, J. & Thirion, J.-P. (2000). Average brain models:
//!   A convergence study. *Computer Vision and Image Understanding*
//!   77(2):192–210.

pub mod label_fusion;

use crate::deformable_field_ops::{
    scaling_and_squaring, validate_image, warp_image, CpuFieldSmoother, CpuOrGpu,
    FieldSmoother, VelocityField,
};
use crate::diffeomorphic::multires_syn::{MultiResSyNConfig, MultiResSyNRegistration};
use crate::error::RegistrationError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for groupwise atlas construction.
///
/// Controls the outer template-building loop and delegates pairwise
/// registrations to [`MultiResSyNRegistration`] via the embedded
/// [`MultiResSyNConfig`].
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Maximum template-building iterations (must be ≥ 1).
    pub max_iterations: usize,
    /// Convergence threshold on per-voxel RMS change of the template.
    /// The outer loop terminates when ‖T^k − T^{k−1}‖₂ / √n falls below
    /// this value.
    pub convergence_threshold: f64,
    /// Multi-Resolution SyN configuration used for every pairwise
    /// subject-to-template registration.
    pub syn_config: MultiResSyNConfig,
}

/// Per-subject registration result retained from the final atlas iteration.
#[derive(Debug, Clone)]
pub struct SubjectResult {
    /// Forward velocity field (z, y, x) — template → midpoint.
    pub forward_field: VelocityField,
    /// Inverse velocity field (z, y, x) — subject → midpoint.
    pub inverse_field: VelocityField,
    /// Final local CC value for this subject's registration.
    pub final_cc: f64,
}

/// Result of atlas construction.
#[derive(Debug, Clone)]
pub struct AtlasResult {
    /// Final template image (flat `Vec<f32>`, shape `[nz, ny, nx]`).
    pub template: Vec<f32>,
    /// Per-subject registration results from the last iteration.
    /// `subject_results[i]` contains the transforms mapping subject `i` into
    /// template space.
    pub subject_results: Vec<SubjectResult>,
    /// Number of template-building iterations actually performed.
    pub num_iterations: usize,
    /// Per-iteration template RMS change values.  Length equals
    /// `num_iterations`.
    pub convergence_history: Vec<f64>,
}

/// Atlas registration engine implementing iterative template building.
///
/// Constructs a population-specific atlas from N subject images by alternating
/// pairwise SyN registrations to the current template with template refinement
/// and mean-drift sharpening.
#[derive(Debug, Clone)]
pub struct AtlasRegistration {
    /// Algorithm configuration.
    pub config: AtlasConfig,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl AtlasRegistration {
    /// Create a new atlas registration engine.
    pub fn new(config: AtlasConfig) -> Self {
        Self { config }
    }

    /// Build a groupwise atlas from N subject images.
    ///
    /// Convenience wrapper that constructs a [`CpuFieldSmoother`] per resolution
    /// level and delegates to [`build_atlas_with`](AtlasRegistration::build_atlas_with).
    pub fn build_atlas(
        &self,
        subjects: &[&[f32]],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<AtlasResult, RegistrationError> {
        let sigma = self.config.syn_config.sigma_smooth;
        let mut factory =
            |ld: [usize; 3]| -> CpuOrGpu { CpuOrGpu::Cpu(CpuFieldSmoother::new(ld, sigma)) };
        self.build_atlas_with(subjects, dims, spacing, &mut factory)
    }

    /// Build a groupwise atlas from N subject images with a user-provided
    /// [`CpuOrGpu`] factory.
    ///
    /// When the factory returns [`CpuOrGpu::Gpu`], the per-level
    /// velocity-field smoothing in the inner SyN registrations and the
    /// per-iteration template-sharpening smoothing both run on the GPU —
    /// 10–50× faster than the CPU path for typical 256³ fields.
    ///
    /// # Arguments
    /// - `smoother_factory` — creates a smoother for a given `[nz, ny, nx]`
    ///   shape.  Called once per resolution level (the SyN multires
    ///   registration creates its own per-level smoothers) and once for the
    ///   full-resolution template sharpening.
    pub fn build_atlas_with<B: burn::tensor::backend::Backend>(
        &self,
        subjects: &[&[f32]],
        dims: [usize; 3],
        spacing: [f64; 3],
        smoother_factory: &mut impl FnMut([usize; 3]) -> CpuOrGpu<B>,
    ) -> Result<AtlasResult, RegistrationError> {
        let n_subjects = subjects.len();
        if n_subjects == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "subjects slice is empty; at least one subject is required".into(),
            ));
        }
        let [nz, ny, nx] = dims;
        let n_voxels = nz * ny * nx;
        if n_voxels == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "dims product is zero; all dimensions must be >= 1".into(),
            ));
        }
        for (i, s) in subjects.iter().enumerate() {
            if let Err(e) = validate_image(s, dims) {
                return Err(RegistrationError::DimensionMismatch(format!(
                    "subjects[{}]: {}",
                    i, e
                )));
            }
        }
        if self.config.max_iterations == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "max_iterations must be >= 1".into(),
            ));
        }

        // ── Step 1: T⁰ = voxel-wise mean ─────────────────────────────────
        let inv_n = 1.0f32 / n_subjects as f32;
        let mut template = vec![0.0f32; n_voxels];
        for s in subjects {
            for i in 0..n_voxels {
                template[i] += s[i];
            }
        }
        for v in template.iter_mut() {
            *v *= inv_n;
        }

        let syn = MultiResSyNRegistration::new(self.config.syn_config.clone());
        let mut convergence_history = Vec::with_capacity(self.config.max_iterations);
        let mut subject_results: Vec<SubjectResult> = Vec::with_capacity(n_subjects);

        for k in 0..self.config.max_iterations {
            // 2a. Register each subject via SyN with GPU smoother factory.
            let mut syn_results = Vec::with_capacity(n_subjects);
            for s in subjects {
                let res = syn.register_with(&template, s, dims, spacing, smoother_factory)?;
                syn_results.push(res);
            }

            // 2b. T̃^k = (1/N) Σᵢ warped_moving_i.
            let mut new_template = vec![0.0f32; n_voxels];
            for res in &syn_results {
                for (dst, &src) in new_template.iter_mut().zip(res.warped_moving.iter()) {
                    *dst += src;
                }
            }
            for v in new_template.iter_mut() {
                *v *= inv_n;
            }

            // 2c. Mean forward velocity v̄ = (1/N) Σᵢ vᵢ.
            let mut mean_vz = vec![0.0f32; n_voxels];
            let mut mean_vy = vec![0.0f32; n_voxels];
            let mut mean_vx = vec![0.0f32; n_voxels];
            for res in &syn_results {
                for i in 0..n_voxels {
                    mean_vz[i] += res.forward_field.z[i];
                    mean_vy[i] += res.forward_field.y[i];
                    mean_vx[i] += res.forward_field.x[i];
                }
            }
            for i in 0..n_voxels {
                mean_vz[i] *= inv_n;
                mean_vy[i] *= inv_n;
                mean_vx[i] *= inv_n;
            }

            // 2d. Template sharpening via GPU smoother.
            for v in mean_vz.iter_mut() {
                *v = -*v;
            }
            for v in mean_vy.iter_mut() {
                *v = -*v;
            }
            for v in mean_vx.iter_mut() {
                *v = -*v;
            }
            let mut sharp_smoother = smoother_factory(dims);
            sharp_smoother.smooth_field(&mut mean_vz, &mut mean_vy, &mut mean_vx);
            let phi = scaling_and_squaring(
                &mean_vz,
                &mean_vy,
                &mean_vx,
                dims.into(),
                self.config.syn_config.n_squarings,
            );
            let sharpened = warp_image(&new_template, dims.into(), &phi.z, &phi.y, &phi.x);

            // 2e. Convergence.
            let rms = {
                let sum_sq: f64 = sharpened
                    .iter()
                    .zip(template.iter())
                    .map(|(&a, &b)| {
                        let d = (a - b) as f64;
                        d * d
                    })
                    .sum();
                (sum_sq / n_voxels as f64).sqrt()
            };
            convergence_history.push(rms);

            subject_results = syn_results
                .into_iter()
                .map(|r| SubjectResult {
                    forward_field: r.forward_field,
                    inverse_field: r.inverse_field,
                    final_cc: r.final_cc,
                })
                .collect();

            template = sharpened;

            if rms < self.config.convergence_threshold {
                return Ok(AtlasResult {
                    template,
                    subject_results,
                    num_iterations: k + 1,
                    convergence_history,
                });
            }
        }

        Ok(AtlasResult {
            template,
            subject_results,
            num_iterations: self.config.max_iterations,
            convergence_history,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "tests.rs"]
mod tests;

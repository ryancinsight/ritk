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
//!       warped image Iᵢ ∘ φᵢ⁻¹.
//!    b. Compute T̃^k = (1/N) Σᵢ Iᵢ ∘ φᵢ⁻¹ (mean of warped subjects).
//!    c. Compute mean forward velocity: v̄ = (1/N) Σᵢ vᵢ (component-wise).
//!    d. Template sharpening: T^k = warp(T̃^k, exp(−G_σ ∗ v̄)) where G_σ is
//!       Gaussian smoothing applied to the negated mean velocity before
//!       exponentiation via scaling-and-squaring.  This removes mean-drift
//!       bias so the template lies at the geometric centre of the population
//!       in diffeomorphism space.
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

use crate::deformable_field_ops::{gaussian_smooth_inplace, scaling_and_squaring, warp_image};
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
    /// Forward velocity field (vz, vy, vx) — template → midpoint.
    pub forward_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Inverse velocity field (vz, vy, vx) — subject → midpoint.
    pub inverse_field: (Vec<f32>, Vec<f32>, Vec<f32>),
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
    /// Every element of `subjects` must be a flat buffer of length
    /// `dims[0] * dims[1] * dims[2]` in Z-major order.
    ///
    /// # Errors
    ///
    /// - [`RegistrationError::InvalidConfiguration`] if `subjects` is empty
    ///   or `max_iterations` is zero.
    /// - [`RegistrationError::DimensionMismatch`] if any subject length
    ///   differs from the product of `dims`.
    /// - Propagates errors from the underlying [`MultiResSyNRegistration`].
    pub fn build_atlas(
        &self,
        subjects: &[&[f32]],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<AtlasResult, RegistrationError> {
        // ── Validation ────────────────────────────────────────────────────
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
            if s.len() != n_voxels {
                return Err(RegistrationError::DimensionMismatch(format!(
                    "subjects[{}] length {} != dims product {}",
                    i,
                    s.len(),
                    n_voxels
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
        let mut subject_results: Vec<SubjectResult> = Vec::new();

        // ── Step 2: Iterative refinement ──────────────────────────────────
        for k in 0..self.config.max_iterations {
            // 2a. Register each subject Iᵢ to T^{k−1}.
            let mut syn_results = Vec::with_capacity(n_subjects);
            for s in subjects {
                let res = syn.register(&template, s, dims, spacing)?;
                syn_results.push(res);
            }

            // 2b. T̃^k = (1/N) Σᵢ warped_moving_i.
            let mut new_template = vec![0.0f32; n_voxels];
            for res in &syn_results {
                for i in 0..n_voxels {
                    new_template[i] += res.warped_moving[i];
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
                    mean_vz[i] += res.forward_field.0[i];
                    mean_vy[i] += res.forward_field.1[i];
                    mean_vx[i] += res.forward_field.2[i];
                }
            }
            for i in 0..n_voxels {
                mean_vz[i] *= inv_n;
                mean_vy[i] *= inv_n;
                mean_vx[i] *= inv_n;
            }

            // 2d. Template sharpening: warp T̃^k by exp(−G_σ ∗ v̄).
            //     Negate mean velocity.
            for v in mean_vz.iter_mut() {
                *v = -*v;
            }
            for v in mean_vy.iter_mut() {
                *v = -*v;
            }
            for v in mean_vx.iter_mut() {
                *v = -*v;
            }
            //     Smooth the negated velocity for diffeomorphism regularity.
            gaussian_smooth_inplace(&mut mean_vz, dims, self.config.syn_config.sigma_smooth);
            gaussian_smooth_inplace(&mut mean_vy, dims, self.config.syn_config.sigma_smooth);
            gaussian_smooth_inplace(&mut mean_vx, dims, self.config.syn_config.sigma_smooth);
            //     Exponentiate via scaling-and-squaring → displacement field.
            let (dz, dy, dx) = scaling_and_squaring(
                &mean_vz,
                &mean_vy,
                &mean_vx,
                dims,
                self.config.syn_config.n_squarings,
            );
            //     Apply sharpening warp.
            let sharpened = warp_image(&new_template, dims, &dz, &dy, &dx);

            // 2e. Convergence: RMS(T^k − T^{k−1}).
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

            // Retain per-subject results from this (latest) iteration.
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
mod tests {
    use super::*;

    /// Minimal SyN config for fast tests on small images.
    fn test_syn_config() -> MultiResSyNConfig {
        MultiResSyNConfig {
            num_levels: 1,
            iterations_per_level: vec![5],
            sigma_smooth: 1.0,
            convergence_threshold: 1e-6,
            convergence_window: 3,
            n_squarings: 2,
            cc_window_radius: 1,
            enforce_inverse_consistency: false,
        }
    }

    fn test_atlas_config() -> AtlasConfig {
        AtlasConfig {
            max_iterations: 3,
            convergence_threshold: 1e-3,
            syn_config: test_syn_config(),
        }
    }

    // ── Positive tests ────────────────────────────────────────────────────

    /// Two identical constant subjects.
    ///
    /// Analytical expectation: T⁰ = c.  Registration of constant c to
    /// constant c yields zero velocity fields (CC gradient is zero when
    /// image gradient is zero).  Mean of warped subjects = c.  Mean
    /// velocity = 0, so sharpening is identity.  RMS change = 0 →
    /// convergence in one iteration.  Template = c everywhere.
    #[test]
    fn two_identical_constant_subjects_template_equals_constant() {
        let dims = [4, 4, 4];
        let n = 64;
        let val = 3.0f32;
        let s1 = vec![val; n];
        let s2 = vec![val; n];
        let subjects: Vec<&[f32]> = vec![&s1, &s2];

        let reg = AtlasRegistration::new(test_atlas_config());
        let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();

        // Template must be val at every voxel.
        for (i, &v) in result.template.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-4,
                "template[{}] = {} deviates from expected {}",
                i,
                v,
                val
            );
        }
        // Convergence in 1 outer iteration (RMS = 0 < threshold).
        assert_eq!(result.num_iterations, 1);
        assert!(result.convergence_history[0] < 1e-3);
        // One result per subject.
        assert_eq!(result.subject_results.len(), 2);
        // CC for constant images is 0 (no local variance).
        for sr in &result.subject_results {
            assert!(sr.final_cc.abs() < 1e-6, "final_cc = {}", sr.final_cc);
        }
    }

    /// Three uniform images with distinct values.
    ///
    /// Analytical expectation: T⁰ = mean(2, 4, 6) = 4.  Registration of a
    /// uniform image to another uniform image produces zero displacement
    /// (gradient is zero).  Mean of warped subjects = mean of originals = 4.
    /// Sharpening is identity (zero velocity).  RMS change = 0 → converge
    /// in 1 iteration.  Template ≈ 4 everywhere.
    #[test]
    fn three_uniform_images_template_is_mean() {
        let dims = [4, 4, 4];
        let n = 64;
        let s1 = vec![2.0f32; n];
        let s2 = vec![4.0f32; n];
        let s3 = vec![6.0f32; n];
        let subjects: Vec<&[f32]> = vec![&s1, &s2, &s3];
        let expected = 4.0f32;

        let reg = AtlasRegistration::new(test_atlas_config());
        let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();

        for (i, &v) in result.template.iter().enumerate() {
            assert!(
                (v - expected).abs() < 0.1,
                "template[{}] = {} deviates from expected {}",
                i,
                v,
                expected
            );
        }
        assert_eq!(result.subject_results.len(), 3);
        assert!(result.convergence_history[0] < 0.1);
    }

    // ── Boundary tests ────────────────────────────────────────────────────

    /// Single subject: template equals that subject.
    ///
    /// T⁰ = I₁.  Registration of I₁ to I₁ → identity.  Template
    /// unchanged.  RMS = 0 → converge in 1 iteration.
    #[test]
    fn single_subject_template_equals_subject() {
        let dims = [4, 4, 4];
        let n = 64;
        let val = 5.0f32;
        let s = vec![val; n];
        let subjects: Vec<&[f32]> = vec![&s];

        let reg = AtlasRegistration::new(test_atlas_config());
        let result = reg.build_atlas(&subjects, dims, [1.0, 1.0, 1.0]).unwrap();

        for (i, &v) in result.template.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-4,
                "template[{}] = {} deviates from expected {}",
                i,
                v,
                val
            );
        }
        assert_eq!(result.num_iterations, 1);
        assert_eq!(result.subject_results.len(), 1);
    }

    // ── Negative tests ────────────────────────────────────────────────────

    /// Empty subjects slice returns `InvalidConfiguration`.
    #[test]
    fn empty_subjects_returns_error() {
        let subjects: Vec<&[f32]> = vec![];
        let reg = AtlasRegistration::new(test_atlas_config());
        let err = reg
            .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {:?}",
            err
        );
    }

    /// Subject with wrong length returns `DimensionMismatch`.
    #[test]
    fn dimension_mismatch_returns_error() {
        let s1 = vec![1.0f32; 64]; // 4*4*4
        let s2 = vec![1.0f32; 27]; // wrong
        let subjects: Vec<&[f32]> = vec![&s1, &s2];

        let reg = AtlasRegistration::new(test_atlas_config());
        let err = reg
            .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {:?}",
            err
        );
    }

    /// `max_iterations = 0` returns `InvalidConfiguration`.
    #[test]
    fn zero_max_iterations_returns_error() {
        let s = vec![1.0f32; 64];
        let subjects: Vec<&[f32]> = vec![&s];

        let mut cfg = test_atlas_config();
        cfg.max_iterations = 0;
        let reg = AtlasRegistration::new(cfg);
        let err = reg
            .build_atlas(&subjects, [4, 4, 4], [1.0, 1.0, 1.0])
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {:?}",
            err
        );
    }

    /// Zero-volume dims returns `InvalidConfiguration`.
    #[test]
    fn zero_dims_returns_error() {
        let s = vec![1.0f32; 0];
        let subjects: Vec<&[f32]> = vec![&s];

        let reg = AtlasRegistration::new(test_atlas_config());
        let err = reg
            .build_atlas(&subjects, [0, 4, 4], [1.0, 1.0, 1.0])
            .unwrap_err();
        assert!(
            matches!(err, RegistrationError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {:?}",
            err
        );
    }
}

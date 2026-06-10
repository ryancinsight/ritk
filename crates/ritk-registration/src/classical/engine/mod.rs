//! Classical image registration engine.
//!
//! Orchestrates non-ML registration algorithms using pure ndarray primitives:
//! - Kabsch SVD for landmark-based rigid registration
//! - Mutual-information hill-climb for intensity-based rigid/affine registration
//!
//! All operations are deterministic and CPU-based with no deep-learning dependency.

pub mod config;
pub mod metric;
pub mod result;

pub use config::ClassicalConfig;
pub use metric::MutualInformationMetric;
pub use result::RegistrationResult;

use super::error::{RegistrationError, Result};
use super::spatial::{
    apply_affine_perturbation, apply_transform_perturbation, build_homogeneous_matrix,
    center_points, compute_centroid, compute_fre, extract_spatial_transform,
    generate_affine_perturbations, generate_transform_perturbations, kabsch_algorithm,
    SpatialTransform,
};
use crate::validation::{RegistrationQualityMetrics, TemporalQualityMetrics};
use nalgebra::Matrix3;
use ndarray::{Array2, Array3};

// ============================================================================
// Image Registration Engine
// ============================================================================

/// Orchestrator for classical (non-ML) image registration algorithms.
///
/// Provides landmark-based and intensity-based registration using:
/// - Kabsch SVD for rigid landmark alignment
/// - Mutual information hill-climbing for intensity-based rigid/affine registration
#[derive(Debug, Clone)]
pub struct ImageRegistration {
    config: ClassicalConfig,
    similarity: MutualInformationMetric,
}

impl ImageRegistration {
    /// Create a new ImageRegistration engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: ClassicalConfig::default(),
            similarity: MutualInformationMetric::default(),
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: ClassicalConfig, similarity: MutualInformationMetric) -> Self {
        Self { config, similarity }
    }

    /// Landmark-based rigid registration using Kabsch SVD.
    ///
    /// Computes optimal rotation R and translation t minimizing:
    ///   min sum ||p_fixed_i - (R * p_moving_i + t)||^2
    pub fn rigid_registration_landmarks(
        &self,
        fixed: &Array2<f64>,
        moving: &Array2<f64>,
    ) -> Result<RegistrationResult> {
        if fixed.nrows() != moving.nrows() {
            return Err(RegistrationError::InvalidInput(
                "Point sets must have same number of points".to_string(),
            ));
        }
        if fixed.ncols() != 3 || moving.ncols() != 3 {
            return Err(RegistrationError::InvalidInput(
                "Points must be 3D (3 columns)".to_string(),
            ));
        }

        // Compute centroids and center the point sets
        let centroid_fixed = compute_centroid(fixed);
        let centroid_moving = compute_centroid(moving);
        let fixed_centered = center_points(fixed, &centroid_fixed);
        let moving_centered = center_points(moving, &centroid_moving);

        // Kabsch SVD for optimal rotation
        let rotation = kabsch_algorithm(&fixed_centered, &moving_centered)?;

        // Translation: t = centroid_fixed - R * centroid_moving
        let r = Matrix3::new(
            rotation[0],
            rotation[1],
            rotation[2],
            rotation[3],
            rotation[4],
            rotation[5],
            rotation[6],
            rotation[7],
            rotation[8],
        );
        let t = centroid_fixed - r * centroid_moving;
        let translation = [t[0], t[1], t[2]];

        // Build homogeneous transformation matrix
        let transform = build_homogeneous_matrix(&rotation, &translation);

        // Compute Fiducial Registration Error (FRE)
        let fre = compute_fre(fixed, moving, &rotation, &translation);

        Ok(RegistrationResult {
            transform,
            spatial: SpatialTransform::RigidBody {
                rotation,
                translation,
            },
            quality: RegistrationQualityMetrics {
                fre: Some(fre),
                tre: None,
                mutual_information: 0.0,
                correlation_coefficient: 0.0,
                normalized_cross_correlation: 0.0,
                converged: true,
                iterations: 1,
                final_cost: fre,
            },
        })
    }

    /// Intensity-based rigid registration using mutual information optimization.
    ///
    /// Uses gradient-free hill-climbing with rigid-body perturbations (6 DOF).
    pub fn rigid_registration_mutual_info(
        &self,
        volume: &Array3<f64>,
        reference: &Array3<f64>,
        initial_transform: &[f64; 16],
    ) -> Result<RegistrationResult> {
        let mut current_transform = *initial_transform;
        let mut iteration = 0;
        let mut prev_loss = f64::MAX;

        while iteration < self.config.max_iterations {
            let current_loss = -self.similarity.compute(volume, reference);

            // Convergence check
            if (prev_loss - current_loss).abs() < self.config.tolerance {
                break;
            }
            prev_loss = current_loss;

            // Use rigid perturbations (6 DOF: 3 Euler angles + 3 translations)
            let perturbations = generate_transform_perturbations();

            let mut best_loss = current_loss;
            let mut best_perturbation: Option<[f64; 6]> = None;

            for perturb in perturbations.iter() {
                let perturbed = apply_transform_perturbation(&current_transform, perturb);
                let transformed = super::spatial::apply_transform(volume, &perturbed);
                let loss = -self.similarity.compute(&transformed, reference);

                if loss < best_loss {
                    best_loss = loss;
                    best_perturbation = Some(*perturb);
                }
            }

            match best_perturbation {
                Some(best) => {
                    current_transform = apply_transform_perturbation(&current_transform, &best);
                    iteration += 1;
                }
                None => break,
            }
        }

        let spatial = extract_spatial_transform(&current_transform)?;

        Ok(RegistrationResult {
            transform: current_transform,
            spatial,
            quality: RegistrationQualityMetrics {
                fre: None,
                tre: None,
                mutual_information: self.similarity.compute(volume, reference),
                correlation_coefficient: 0.0,
                normalized_cross_correlation: 0.0,
                converged: iteration < self.config.max_iterations,
                iterations: iteration,
                final_cost: prev_loss,
            },
        })
    }

    /// Intensity-based affine registration using mutual information optimization.
    ///
    /// Uses gradient-free hill-climbing with affine perturbations (9 DOF:
    /// 3 Euler angles + 3 translations + 3 anisotropic scales).
    pub fn affine_registration_mutual_info(
        &self,
        volume: &Array3<f64>,
        reference: &Array3<f64>,
        initial_transform: &[f64; 16],
    ) -> Result<RegistrationResult> {
        let mut current_transform = *initial_transform;
        let mut iteration = 0;
        let mut prev_loss = f64::MAX;

        while iteration < self.config.max_iterations {
            let current_loss = -self.similarity.compute(volume, reference);

            // Convergence check
            if (prev_loss - current_loss).abs() < self.config.tolerance {
                break;
            }
            prev_loss = current_loss;

            // Use affine perturbations (9 DOF)
            let perturbations = generate_affine_perturbations();

            let mut best_loss = current_loss;
            let mut best_perturbation: Option<[f64; 9]> = None;

            for perturb in perturbations.iter() {
                let perturbed = apply_affine_perturbation(&current_transform, perturb);
                let transformed = super::spatial::apply_transform(volume, &perturbed);
                let loss = -self.similarity.compute(&transformed, reference);

                if loss < best_loss {
                    best_loss = loss;
                    best_perturbation = Some(*perturb);
                }
            }

            match best_perturbation {
                Some(best) => {
                    current_transform = apply_affine_perturbation(&current_transform, &best);
                    iteration += 1;
                }
                None => break,
            }
        }

        let spatial = extract_spatial_transform(&current_transform)?;

        Ok(RegistrationResult {
            transform: current_transform,
            spatial,
            quality: RegistrationQualityMetrics {
                fre: None,
                tre: None,
                mutual_information: self.similarity.compute(volume, reference),
                correlation_coefficient: 0.0,
                normalized_cross_correlation: 0.0,
                converged: iteration < self.config.max_iterations,
                iterations: iteration,
                final_cost: prev_loss,
            },
        })
    }

    /// Synchronize temporal signals from multi-modal acquisitions.
    ///
    /// Uses cross-correlation phase estimation to find optimal temporal shift.
    pub fn temporal_synchronization(
        &self,
        signal1: &ndarray::Array1<f64>,
        signal2: &ndarray::Array1<f64>,
    ) -> Result<(f64, TemporalQualityMetrics)> {
        use super::temporal::TemporalSync;
        let sync = TemporalSync::new();
        sync.synchronize(signal1, signal2)
    }
}

impl Default for ImageRegistration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

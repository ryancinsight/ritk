use crate::error::Result;
use ndarray::{Array1, Array2, Array3};

use super::intensity::{compute_correlation, compute_mutual_information, compute_ncc};
use super::metrics::RegistrationQualityMetrics;
use super::spatial::{
    apply_affine_perturbation, apply_transform_perturbation, generate_affine_perturbations,
    generate_transform_perturbations,
};
use super::spatial::{
    build_homogeneous_matrix, center_points, compute_centroid, compute_fre,
    extract_spatial_transform, kabsch_algorithm, SpatialTransform,
};
use super::temporal::{temporal_synchronization, TemporalSync};

/// Registration result containing transformation and quality metrics
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Spatial transformation
    pub spatial_transform: Option<SpatialTransform>,
    /// Temporal synchronization
    pub temporal_sync: Option<TemporalSync>,
    /// Registration quality metrics
    pub quality_metrics: RegistrationQualityMetrics,
    /// Transformation matrix (4x4 homogeneous)
    pub transform_matrix: [f64; 16],
    /// Registration confidence [0-1]
    pub confidence: f64,
}

/// Image registration engine
#[derive(Debug)]
pub struct ImageRegistration {
    /// Maximum iterations for optimization
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Regularization parameter for non-rigid registration
    #[allow(dead_code)]
    regularization_weight: f64,
}

impl Default for ImageRegistration {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            regularization_weight: 0.1,
        }
    }
}

impl ImageRegistration {
    /// Create new registration engine with custom parameters
    #[must_use]
    pub fn new(max_iterations: usize, tolerance: f64, regularization_weight: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            regularization_weight,
        }
    }

    /// Perform rigid body registration using landmark points
    ///
    /// # Arguments
    /// * `fixed_landmarks` - Landmark points in fixed image [N, 3]
    /// * `moving_landmarks` - Corresponding landmark points in moving image [N, 3]
    ///
    /// # Returns
    /// Registration result with rigid body transformation
    pub fn rigid_registration_landmarks(
        &self,
        fixed_landmarks: &Array2<f64>,
        moving_landmarks: &Array2<f64>,
    ) -> Result<RegistrationResult> {
        if fixed_landmarks.nrows() != moving_landmarks.nrows() {
            return Err(crate::error::RegistrationError::InvalidInput(
                "Fixed and moving landmark arrays must have same number of points".to_string(),
            ));
        }

        if fixed_landmarks.ncols() != 3 || moving_landmarks.ncols() != 3 {
            return Err(crate::error::RegistrationError::InvalidInput(
                "Landmark arrays must have 3 columns (x, y, z)".to_string(),
            ));
        }

        let _n_points = fixed_landmarks.nrows();

        // Compute centroids
        let fixed_centroid = compute_centroid(fixed_landmarks);
        let moving_centroid = compute_centroid(moving_landmarks);

        // Center the points
        let fixed_centered = center_points(fixed_landmarks, &fixed_centroid);
        let moving_centered = center_points(moving_landmarks, &moving_centroid);

        // Compute rotation matrix using Kabsch algorithm
        let rotation = kabsch_algorithm(&fixed_centered, &moving_centered)?;

        // Compute translation
        let translation = [
            fixed_centroid[0]
                - (rotation[0] * moving_centroid[0]
                    + rotation[1] * moving_centroid[1]
                    + rotation[2] * moving_centroid[2]),
            fixed_centroid[1]
                - (rotation[3] * moving_centroid[0]
                    + rotation[4] * moving_centroid[1]
                    + rotation[5] * moving_centroid[2]),
            fixed_centroid[2]
                - (rotation[6] * moving_centroid[0]
                    + rotation[7] * moving_centroid[1]
                    + rotation[8] * moving_centroid[2]),
        ];

        // Build homogeneous transformation matrix
        let transform_matrix = build_homogeneous_matrix(&rotation, &translation);

        // Compute quality metrics
        let fre = compute_fre(fixed_landmarks, moving_landmarks, &rotation, &translation);
        let quality_metrics = RegistrationQualityMetrics {
            fre: Some(fre),
            tre: None,               // Would need anatomical landmarks for TRE
            mutual_information: 0.0, // Not computed for landmark-based registration
            correlation_coefficient: 0.0,
            normalized_cross_correlation: 0.0,
            converged: true,
            iterations: 1,
            final_cost: fre,
        };

        let spatial_transform = SpatialTransform::RigidBody {
            rotation,
            translation,
        };

        Ok(RegistrationResult {
            spatial_transform: Some(spatial_transform),
            temporal_sync: None,
            quality_metrics,
            transform_matrix,
            confidence: (1.0 / (1.0 + fre)).min(1.0), // Higher FRE = lower confidence
        })
    }

    /// Perform intensity-based registration using mutual information
    ///
    /// # Arguments
    /// * `fixed_image` - Reference image
    /// * `moving_image` - Image to be registered
    /// * `initial_transform` - Initial transformation guess
    ///
    /// # Returns
    /// Registration result with optimized transformation
    pub fn intensity_registration_mutual_info(
        &self,
        fixed_image: &Array3<f64>,
        moving_image: &Array3<f64>,
        initial_transform: &[f64; 16],
    ) -> Result<RegistrationResult> {
        // Simplified mutual information registration
        // In practice, this would use optimization algorithms like Powell's method
        // or gradient descent to maximize mutual information

        let mut current_transform = *initial_transform;
        let mut best_mi = compute_mutual_information(fixed_image, moving_image, &current_transform);
        let mut converged = false;

        for _iteration in 0..self.max_iterations {
            // Try small perturbations to the transformation
            let perturbations = generate_transform_perturbations();

            let mut best_perturbation = None;
            let mut best_perturbation_mi = best_mi;

            for perturbation in &perturbations {
                let test_transform = apply_transform_perturbation(&current_transform, perturbation);
                let test_mi =
                    compute_mutual_information(fixed_image, moving_image, &test_transform);

                if test_mi > best_perturbation_mi {
                    best_perturbation_mi = test_mi;
                    best_perturbation = Some(*perturbation);
                }
            }

            if let Some(perturbation) = best_perturbation {
                if (best_perturbation_mi - best_mi).abs() < self.tolerance {
                    converged = true;
                    break;
                }
                current_transform = apply_transform_perturbation(&current_transform, &perturbation);
                best_mi = best_perturbation_mi;
            } else {
                break;
            }
        }

        // Extract spatial transform from homogeneous matrix
        let spatial_transform = extract_spatial_transform(&current_transform)?;

        let quality_metrics = RegistrationQualityMetrics {
            fre: None,
            tre: None,
            mutual_information: best_mi,
            correlation_coefficient: compute_correlation(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            normalized_cross_correlation: compute_ncc(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            converged,
            iterations: self.max_iterations,
            final_cost: -best_mi, // Negative because we maximize MI but cost should be minimized
        };

        Ok(RegistrationResult {
            spatial_transform: Some(spatial_transform),
            temporal_sync: None,
            quality_metrics,
            transform_matrix: current_transform,
            confidence: best_mi.min(1.0),
        })
    }

    /// Perform affine registration (9 DOF: rotation + translation + anisotropic scale)
    /// using mutual information maximisation.
    ///
    /// ## Algorithm: MI-based 9-DOF Affine Optimisation
    ///
    /// Affine registration subsumes rigid-body registration by additionally allowing
    /// independent scale factors `(sx, sy, sz)` along each axis.  The full affine
    /// homogeneous matrix is:
    ///
    /// ```text
    /// A = [ sx·R  |  t ]     (3×4, padded to 4×4 with [0 0 0 1])
    ///     [  0 0 0 1   ]
    /// ```
    ///
    /// The optimisation is a coordinate-descent hill-climb over MI:
    ///
    /// 1. Start from the identity transform (or supplied `initial_transform`).
    /// 2. At each iteration generate perturbation candidates spanning all 9 DOF
    ///    (3 Euler angles, 3 translation offsets, 3 scale deltas).
    /// 3. Accept the candidate that maximises MI(fixed, transformed_moving).
    /// 4. Converge when the MI improvement is smaller than `tolerance`.
    ///
    /// ## CFL analogue / stability note
    ///
    /// The perturbation step sizes are kept small (Δθ = 0.01 rad, Δt = 1 voxel,
    /// Δs = 0.02) to ensure MI remains smooth and the hill-climb does not overshoot.
    ///
    /// ## References
    ///
    /// - Maes, F., et al. (1997). "Multimodality image registration by maximization of
    ///   mutual information." *IEEE TMI* **16**(2), 187–198.
    /// - Pluim, J. P. W., et al. (2003). "Mutual-information-based registration of
    ///   medical images: a survey." *IEEE TMI* **22**(8), 986–1004.
    pub fn intensity_registration_affine(
        &self,
        fixed_image: &Array3<f64>,
        moving_image: &Array3<f64>,
        initial_transform: &[f64; 16],
    ) -> Result<RegistrationResult> {
        let mut current_transform = *initial_transform;
        let mut best_mi = compute_mutual_information(fixed_image, moving_image, &current_transform);
        let mut converged = false;

        for _iteration in 0..self.max_iterations {
            let perturbations = generate_affine_perturbations();

            let mut best_perturbation: Option<[f64; 9]> = None;
            let mut best_perturbation_mi = best_mi;

            for perturbation in &perturbations {
                let test_transform = apply_affine_perturbation(&current_transform, perturbation);
                let test_mi =
                    compute_mutual_information(fixed_image, moving_image, &test_transform);
                if test_mi > best_perturbation_mi {
                    best_perturbation_mi = test_mi;
                    best_perturbation = Some(*perturbation);
                }
            }

            if let Some(perturbation) = best_perturbation {
                if (best_perturbation_mi - best_mi).abs() < self.tolerance {
                    converged = true;
                    break;
                }
                current_transform = apply_affine_perturbation(&current_transform, &perturbation);
                best_mi = best_perturbation_mi;
            } else {
                converged = true;
                break;
            }
        }

        // Decode the affine 3×4 block into an Affine SpatialTransform
        let affine_matrix = [
            current_transform[0],
            current_transform[1],
            current_transform[2],
            current_transform[3],
            current_transform[4],
            current_transform[5],
            current_transform[6],
            current_transform[7],
            current_transform[8],
            current_transform[9],
            current_transform[10],
            current_transform[11],
        ];
        let spatial_transform = SpatialTransform::Affine {
            matrix: affine_matrix,
        };

        let quality_metrics = RegistrationQualityMetrics {
            fre: None,
            tre: None,
            mutual_information: best_mi,
            correlation_coefficient: compute_correlation(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            normalized_cross_correlation: compute_ncc(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            converged,
            iterations: self.max_iterations,
            final_cost: -best_mi,
        };

        Ok(RegistrationResult {
            spatial_transform: Some(spatial_transform),
            temporal_sync: None,
            quality_metrics,
            transform_matrix: current_transform,
            confidence: best_mi.min(1.0),
        })
    }

    /// Perform temporal synchronization for multi-modal acquisition
    ///
    /// # Arguments
    /// * `reference_signal` - Reference modality timing signal
    /// * `target_signal` - Target modality timing signal
    /// * `sampling_rate` - Sampling frequency \[Hz\]
    ///
    /// # Returns
    /// Temporal synchronization result
    pub fn temporal_synchronization(
        &self,
        reference_signal: &Array1<f64>,
        target_signal: &Array1<f64>,
        sampling_rate: f64,
    ) -> Result<TemporalSync> {
        temporal_synchronization(reference_signal, target_signal, sampling_rate)
    }

    /// Apply spatial transformation to image
    ///
    /// # Arguments
    /// * `image` - Input image to transform
    /// * `transform` - Homogeneous transformation matrix
    ///
    /// # Returns
    /// Transformed image
    pub fn apply_transform(&self, image: &Array3<f64>, transform: &[f64; 16]) -> Array3<f64> {
        super::spatial::apply_transform(image, transform)
    }
}

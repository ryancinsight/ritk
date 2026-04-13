//! Classical image registration engine.
//!
//! Orchestrates non-ML registration algorithms using pure ndarray primitives:
//! - Kabsch SVD for landmark-based rigid registration
//! - Mutual-information hill-climb for intensity-based rigid/affine registration
//!
//! All operations are deterministic and CPU-based with no deep-learning dependency.
use super::error::{RegistrationError, Result};
use super::spatial::{
    apply_affine_perturbation, apply_transform_perturbation, build_homogeneous_matrix,
    center_points, compute_centroid, compute_fre, extract_spatial_transform,
    generate_affine_perturbations, generate_transform_perturbations, kabsch_algorithm,
    SpatialTransform,
};
use crate::validation::{RegistrationQualityMetrics, TemporalQualityMetrics};
use nalgebra::Matrix3;
use ndarray::{Array2, Array3, Axis};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for classical registration algorithms.
#[derive(Debug, Clone)]
pub struct ClassicalConfig {
    /// Maximum number of iterations for optimization.
    pub max_iterations: usize,
    /// Convergence tolerance for similarity metric improvement.
    pub tolerance: f64,
    /// Step size multiplier for perturbations.
    pub step_multiplier: f64,
}

impl Default for ClassicalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            step_multiplier: 1.0,
        }
    }
}

// ============================================================================
// Similarity Metric (ndarray-based, non-ML)
// ============================================================================

/// Mutual Information similarity metric using histogram density estimation.
///
/// Given joint histogram H(a,b) with N bins each:
/// p(a,b) = H(a,b) / sum(H)
/// p(a) = sum_b p(a,b)
/// p(b) = sum_a p(a,b)
/// MI(A,B) = sum_a sum_b p(a,b) * log(p(a,b) / (p(a) * p(b)))
///
/// Normalized MI: NMI = 2*MI / (H(A) + H(B))
#[derive(Debug, Clone)]
pub struct MutualInformationMetric {
    /// Number of histogram bins per intensity dimension.
    pub num_bins: usize,
    /// Minimum intensity value for binning.
    min_intensity: f64,
    /// Width of each histogram bin.
    bin_width: f64,
}

impl MutualInformationMetric {
    /// Create a new Mutual Information metric with explicit parameters.
    pub fn new(num_bins: usize, min_intensity: f64, max_intensity: f64) -> Self {
        let bin_width = (max_intensity - min_intensity) / num_bins as f64;
        Self {
            num_bins,
            min_intensity,
            bin_width,
        }
    }

    /// Compute joint histogram between two volumes.
    fn compute_joint_histogram(&self, fixed: &Array3<f64>, moving: &Array3<f64>) -> Array2<f64> {
        let mut joint_hist = Array2::<f64>::zeros((self.num_bins, self.num_bins));
        let step = std::cmp::max(1, fixed.len() / 10000);

        // Iterate using direct iteration over the arrays
        let fixed_iter = fixed.iter().copied();
        let moving_iter = moving.iter().copied();

        for (f_val, m_val) in fixed_iter.zip(moving_iter).step_by(step) {
            let f_bin = ((f_val - self.min_intensity) / self.bin_width).floor() as usize;
            let m_bin = ((m_val - self.min_intensity) / self.bin_width).floor() as usize;
            if f_bin < self.num_bins && m_bin < self.num_bins {
                joint_hist[[f_bin, m_bin]] += 1.0;
            }
        }

        joint_hist
    }

    /// Compute normalized mutual information: NMI = 2 * MI / (H(X) + H(Y))
    /// where MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
    pub fn compute(&self, fixed: &Array3<f64>, moving: &Array3<f64>) -> f64 {
        let joint_hist = self.compute_joint_histogram(fixed, moving);
        let total = joint_hist.sum();
        if total == 0.0 {
            return 0.0;
        }

        // Compute joint and marginal probabilities
        let p_joint = &joint_hist / total;
        let p_x = p_joint.sum_axis(Axis(1));
        let p_y = p_joint.sum_axis(Axis(0));

        // Compute entropies - pass slices for 1D arrays
        let h_x = self.compute_entropy(p_x.as_slice().unwrap_or(&[]));
        let h_y = self.compute_entropy(p_y.as_slice().unwrap_or(&[]));
        let h_xy = self.compute_joint_entropy(&p_joint);

        // Normalized MI
        if h_x + h_y == 0.0 {
            // Both images are constant (entropy = 0).
            // NMI = 1.0 iff both concentrate at the same histogram bin (identical values).
            let peak_x = p_x
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let peak_y = p_y
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX);
            return if peak_x == peak_y { 1.0 } else { 0.0 };
        }
        let mi = h_x + h_y - h_xy;
        (2.0 * mi) / (h_x + h_y)
    }

    /// Compute entropy H(X) = -sum p(x) * log(p(x))
    fn compute_entropy(&self, p: &[f64]) -> f64 {
        let eps = 1e-10_f64;
        p.iter().filter(|&&x| x > eps).map(|&x| -x * x.ln()).sum()
    }

    /// Compute joint entropy H(X,Y) = -sum p(x,y) * log(p(x,y))
    fn compute_joint_entropy(&self, p_joint: &Array2<f64>) -> f64 {
        let eps = 1e-10_f64;
        p_joint
            .iter()
            .filter(|&&x| x > eps)
            .map(|&x| -x * x.ln())
            .sum()
    }
}

impl Default for MutualInformationMetric {
    fn default() -> Self {
        Self::new(32, 0.0, 255.0)
    }
}

// ============================================================================
// Registration Result
// ============================================================================

/// Result of a classical registration operation.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Final 4x4 homogeneous transformation matrix.
    pub transform: [f64; 16],
    /// Spatial transform classification.
    pub spatial: SpatialTransform,
    /// Registration quality metrics.
    pub quality: RegistrationQualityMetrics,
}

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
    /// min sum ||p_fixed_i - (R * p_moving_i + t)||^2
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigid_landmark_identity() {
        let reg = ImageRegistration::default();
        let fixed =
            Array2::from_shape_vec((3, 3), vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
        let result = reg.rigid_registration_landmarks(&fixed, &fixed).unwrap();

        // Identity transform should have zero FRE
        let fre = result.quality.fre.unwrap();
        assert!(
            fre < 1e-10,
            "FRE for identity transform should be ~0, got {}",
            fre
        );
    }

    #[test]
    fn test_rigid_landmark_known_rotation() {
        let reg = ImageRegistration::default();

        // Fixed points: unit vectors along axes
        let fixed =
            Array2::from_shape_vec((3, 3), vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]).unwrap();

        // Moving points: same points rotated 90 deg around Z-axis
        let moving =
            Array2::from_shape_vec((3, 3), vec![0., 1., 0., -1., 0., 0., 0., 0., 1.]).unwrap();

        let result = reg.rigid_registration_landmarks(&fixed, &moving).unwrap();
        let fre = result.quality.fre.unwrap();
        assert!(
            fre < 1e-6,
            "FRE for 90 deg rotation should be ~0, got {}",
            fre
        );
    }

    #[test]
    fn test_mutual_information_identical() {
        let metric = MutualInformationMetric::default();
        let volume = Array3::from_elem((10, 10, 10), 128.0);
        let nmi = metric.compute(&volume, &volume);

        // Identical volumes have NMI = 1
        assert!(
            (nmi - 1.0).abs() < 1e-6,
            "NMI for identical volumes should be 1.0, got {}",
            nmi
        );
    }

    #[test]
    fn test_mutual_information_different() {
        let metric = MutualInformationMetric::default();
        let vol1 = Array3::from_elem((10, 10, 10), 100.0);
        let vol2 = Array3::from_elem((10, 10, 10), 200.0);
        let nmi = metric.compute(&vol1, &vol2);

        // Different constant volumes have low NMI
        assert!(
            nmi < 1.0,
            "NMI for different constant volumes should be < 1, got {}",
            nmi
        );
    }
}

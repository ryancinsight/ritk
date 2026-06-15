//! Elastic regularization for displacement fields.
//!
//! Elastic regularization combines first-order (diffusion) and second-order
//! (bending) terms, balancing smoothness and flexibility for image registration.
//!
//! The elastic regularization term is derived from linear elasticity theory:
//!
//! R(u) = ∫_Ω [μ|∇u + (∇u)^T|² + λ(div u)²] dx
//!
//! where μ is the shear modulus and λ is the first Lamé parameter.
//!
//! Simplified forms used in practice:
//! - Hyperelastic: Combines membrane energy with volume preservation
//! - Linear elastic: Standard elasticity formulation
//!
//! ## Mathematical Background
//!
//! The elastic energy measures how much a deformation deviates from
//! being rigid (uniform translation/rotation), penalizing both shearing
//! and volume change.
//!
//! ## References
//! - Broit (1981): Original elastic registration formulation
//! - Christensen et al. (1996): Consistent linear elastic registration
//! - Modern variants: Hyperelastic (volume-preserving) formulations

use super::trait_::Regularizer;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Elastic regularizer combining membrane and volume-preserving terms.
///
/// This regularizer balances smoothness with the ability to preserve
/// local volumes during deformation (hyperelastic behavior).
///
/// The regularization term is:
/// R(u) = alpha * membrane_energy + beta * volume_preservation_term
///
/// where:
/// - membrane_energy penalizes spatial variation
/// - volume_preservation_term discourages volume change
///
/// # Example
///
/// ```rust,ignore
/// use ritk_registration::regularization::ElasticRegularizer;
/// use burn::tensor::Tensor;
///
/// // Create hyperelastic regularizer
/// let reg = ElasticRegularizer::hyperelastic(0.1, 0.01);
/// let displacement = Tensor::zeros([1, 2, 64, 64], &device);
/// let loss = reg.compute_loss::<4>(displacement);
/// ```
#[derive(Clone, Debug)]
pub struct ElasticRegularizer {
    /// Weight for membrane (first-order smoothness) term
    alpha: f64,
    /// Weight for volume-preservation term
    beta: f64,
}

impl ElasticRegularizer {
    /// Create a new elastic regularizer with custom weights.
    ///
    /// # Arguments
    /// * `alpha` - Weight for membrane energy (shear modulus)
    /// * `beta` - Weight for volume preservation (bulk modulus)
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Create a hyperelastic regularizer favoring volume preservation.
    ///
    /// This configuration is suitable for applications where maintaining
    /// local volumes is important (e.g., brain registration).
    ///
    /// # Arguments
    /// * `alpha` - Weight for membrane energy (typically ~0.1)
    /// * `beta` - Weight for volume preservation (typically smaller than alpha)
    pub fn hyperelastic(alpha: f64, beta: f64) -> Self {
        Self::new(alpha, beta)
    }

    /// Create a standard linear elastic regularizer.
    ///
    /// Uses equal weighting for both terms.
    pub fn linear(weight: f64) -> Self {
        Self::new(weight, weight)
    }
}

impl Default for ElasticRegularizer {
    fn default() -> Self {
        // Default to hyperelastic with moderate volume preservation
        Self::hyperelastic(0.1, 0.01)
    }
}

impl<B: Backend> Regularizer<B> for ElasticRegularizer {
    fn compute_loss<const D: usize>(&self, displacement: Tensor<B, D>) -> Tensor<B, 1> {
        super::dispatch::dispatch_elastic(displacement, self.alpha, self.beta)
    }

    fn weight(&self) -> f64 {
        self.alpha // Return primary weight
    }

    fn set_weight(&mut self, weight: f64) {
        self.alpha = weight;
        self.beta = weight * 0.1; // Maintain ratio
    }
}

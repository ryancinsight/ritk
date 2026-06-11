//! Combined similarity + regularization registration loss orchestrator.

use burn::tensor::{backend::Backend, Tensor};

use super::grad::{GradLoss, GradientPenalty};
use super::lncc::LocalNCCLoss;
use super::ncc::GlobalNCCLoss;

/// Similarity metric for deep learning registration.
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Ncc,
    Mse,
    GlobalNcc,
}

/// Regularization type for deep learning registration.
#[derive(Debug, Clone, Copy)]
pub enum RegularizationType {
    L2,
    L1,
}

/// Combined registration loss configuration.
#[derive(Debug, Clone)]
pub struct RegistrationLossConfig {
    pub reg_weight: f32,
    pub similarity: SimilarityMetric,
    pub regularization: RegularizationType,
}

impl Default for RegistrationLossConfig {
    fn default() -> Self {
        Self {
            reg_weight: 0.1,
            similarity: SimilarityMetric::Ncc,
            regularization: RegularizationType::L2,
        }
    }
}

/// Orchestrates similarity and regularization losses for end-to-end DL registration.
pub struct RegistrationLoss<B: Backend> {
    config: RegistrationLossConfig,
    ncc_loss: LocalNCCLoss<B>,
    global_ncc_loss: GlobalNCCLoss<B>,
    grad_loss: GradLoss<B>,
}

impl<B: Backend> RegistrationLoss<B> {
    pub fn new(config: RegistrationLossConfig, device: &B::Device) -> Self {
        let ncc_loss = LocalNCCLoss::new(9, device);
        let global_ncc_loss = GlobalNCCLoss::new();
        let grad_loss = match config.regularization {
            RegularizationType::L2 => GradLoss::new(GradientPenalty::L2),
            RegularizationType::L1 => GradLoss::new(GradientPenalty::L1),
        };

        Self {
            config,
            ncc_loss,
            global_ncc_loss,
            grad_loss,
        }
    }

    // NOTE: fixed.clone() and warped.clone() because inner losses take ownership.
    pub fn similarity_loss(&self, fixed: &Tensor<B, 5>, warped: &Tensor<B, 5>) -> Tensor<B, 1> {
        match self.config.similarity {
            SimilarityMetric::Ncc => self.ncc_loss.forward(fixed.clone(), warped.clone()),
            SimilarityMetric::GlobalNcc => {
                self.global_ncc_loss.forward(fixed.clone(), warped.clone())
            }
            SimilarityMetric::Mse => {
                let diff = fixed.clone() - warped.clone();
                diff.powf_scalar(2.0).mean().reshape([1])
            }
        }
    }

    pub fn regularization_loss(&self, displacement: &Tensor<B, 5>) -> Tensor<B, 1> {
        self.grad_loss.forward(displacement.clone())
    }
}

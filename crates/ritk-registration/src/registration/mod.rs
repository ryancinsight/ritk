//! Registration framework with validation, progress tracking, and convergence detection.

//! This module provides a registration workflow with:
//! - Input validation and numerical stability checks
//! - Progress tracking and callbacks
//! - Early stopping and convergence detection
//! - Gradient clipping
//! - Learning rate scheduling
//! - Comprehensive error handling

use crate::progress::EarlyStoppingCallback;
use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use ritk_core::transform::Transform;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::progress::ProgressTracker;

pub mod config;
pub mod dl_registration_loss;
pub mod dl_ssm_registration;
pub mod engine;
pub mod summary;

pub use config::{EarlyStoppingPolicy, RegistrationConfig};
pub use dl_registration_loss::{
    RegistrationLoss, RegistrationLossConfig, RegularizationType, SimilarityMetric,
};
pub use summary::{RegistrationSummary, StopReason};

/// Registration framework with validation and progress tracking.
pub struct Registration<B, O, M, T, const D: usize>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    optimizer: O,
    metric: M,
    config: RegistrationConfig,
    progress_tracker: ProgressTracker,
    early_stopping: Option<Arc<EarlyStoppingCallback>>,
    _phantom: PhantomData<(B, T)>,
}

#[cfg(test)]
mod tests;

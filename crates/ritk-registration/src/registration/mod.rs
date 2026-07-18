//! Registration framework with validation, progress tracking, and convergence detection.

//! This module provides a registration workflow with:
//! - Input validation and numerical stability checks
//! - Progress tracking and callbacks
//! - Early stopping and convergence detection
//! - Gradient clipping
//! - Learning rate scheduling
//! - Comprehensive error handling

use crate::progress::EarlyStoppingCallback;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use ritk_transform::Transform;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::progress::ProgressTracker;

pub mod config;
pub mod dl_ssm_registration;
pub mod engine;
pub mod summary;

pub use config::{EarlyStoppingPolicy, RegistrationConfig};
pub use summary::{RegistrationSummary, StopReason};

/// Registration framework with validation and progress tracking.
pub struct Registration<B, O, M, T, const D: usize>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + Module<f32, B>,
{
    optimizer: O,
    metric: M,
    config: RegistrationConfig,
    progress_tracker: ProgressTracker,
    early_stopping: Option<Arc<EarlyStoppingCallback>>,
    _phantom: PhantomData<(B, T)> }

#[cfg(test)]
mod tests;

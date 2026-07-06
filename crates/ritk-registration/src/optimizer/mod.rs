//! Optimizer module for training transforms.
//!
//! This module provides optimization algorithms for training transforms
//! in image registration. It re-exports Burn's optimizers for convenience.
//!
//! # Examples
//!
//! ```rust,ignore
//! use ritk_registration::optimizer::SgdConfig;
//! use ritk_image::burn::optim::Optimizer;
//!
//! let config = SgdConfig::new().with_learning_rate(0.01);
//! let optimizer = config.init();
//! ```

pub mod adam;
pub mod adaptive_stochastic_gd;
pub mod cma_es;
pub mod gradient_descent;
pub mod momentum;
pub mod regular_step_gd;
pub mod trait_;

pub use adam::AdamOptimizer;
pub use adaptive_stochastic_gd::{AdaptiveStochasticGdConfig, AdaptiveStochasticGradientDescent};
pub use cma_es::{
    CmaEsConfig, CmaEsOptimizer, CmaEsResult, CmaEsStopReason, HistoryPolicy, PopulationEval,
};
pub use gradient_descent::GradientDescent;
pub use momentum::Momentum;
pub use regular_step_gd::{
    ConvergenceFlag, ConvergenceReason, RegularStepGdConfig, RegularStepGradientDescent,
};
pub use trait_::{
    LearningRateScheduler, Optimizer, OptimizerAlgorithm, OptimizerTelemetry, StepDecay,
};

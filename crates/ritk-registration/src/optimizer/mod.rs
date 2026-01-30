//! Optimizer module for training transforms.
//!
//! This module provides optimization algorithms for training transforms
//! in image registration. It re-exports Burn's optimizers for convenience.
//!
//! # Examples
//!
//! ```rust,ignore
//! use ritk_registration::optimizer::SgdConfig;
//! use burn::optim::Optimizer;
//!
//! let config = SgdConfig::new().with_learning_rate(0.01);
//! let optimizer = config.init();
//! ```

pub mod trait_;
pub mod gradient_descent;
pub mod momentum;
pub mod adam;
pub mod lbfgs;

pub use trait_::{Optimizer, LearningRateScheduler, StepDecay};
pub use gradient_descent::GradientDescent;
pub use momentum::Momentum;
pub use adam::AdamOptimizer;
pub use lbfgs::{LbfgsOptimizer, LbfgsConfig};

//! Regularization module for deformation field regularization.
//!
//! This module provides various regularization techniques for constraining
//! deformation fields during image registration, ported from AirLab's regulariser.
//!
//! Regularization is essential for:
//! - Preventing overfitting to local minima
//! - Ensuring smooth and physically plausible deformations
//! - Controlling the magnitude of displacements
//!
//! # Available Regularizers
//!
//! * **DiffusionRegularizer**: First-order smoothness (penalizes gradients)
//! * **CurvatureRegularizer**: Second-order smoothness (penalizes curvature)
//! * **ElasticRegularizer**: Elastic membrane model
//! * **BendingEnergyRegularizer**: Bending energy for B-spline transforms
//! * **TotalVariationRegularizer**: L1 norm of gradients (edge-preserving)

pub mod trait_;
pub mod diffusion;
pub mod curvature;
pub mod elastic;
pub mod bending_energy;
pub mod total_variation;

pub use trait_::Regularizer;
pub use diffusion::DiffusionRegularizer;
pub use curvature::CurvatureRegularizer;
pub use elastic::ElasticRegularizer;
pub use bending_energy::BendingEnergyRegularizer;
pub use total_variation::TotalVariationRegularizer;

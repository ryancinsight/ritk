//! Coeus-autograd differentiable registration primitives.
//!
//! Atlas migration (burn → coeus): the reverse-mode autodiff path for the
//! registration metrics (`docs/coeus_migration.md`, dev-sequence step 6,
//! gate #3 — "registration metrics preserve autodiff tape connectivity; no
//! host extraction on differentiable paths").
//!
//! Each primitive here is built entirely from Coeus autograd [`coeus_autograd::Var`]
//! ops so the reverse pass propagates gradients to the intended leaves:
//!
//! - [`transform`] — differentiable coordinate transforms (translation, affine)
//!   mapping the fixed grid into moving space as a function of trainable
//!   parameters.
//! - [`sampling`] — differentiable interpolation of a moving-image `Var` at
//!   continuous coordinates; the gradient flows to the coordinate leaf (hence
//!   to the transform parameters that produce the coordinates).
//! - [`mse`] — the terminal intensity-difference loss reduction the sampled
//!   intensities feed into.
//! - [`metric`] — the end-to-end composition
//!   `mse(sample(moving, translate(grid, t)), fixed)`, whose gradient reaches
//!   the translation parameters.
//! - [`optim`] — the `Var`-level gradient-descent step a registration loop
//!   uses to update the transform parameters from the metric gradient.
//!
//! The Coeus-native `Metric`/`Transform` trait surface that wraps this
//! composition remains a separate (ADR-gated, `[arch]`) increment.

pub mod metric;
pub mod mse;
pub mod optim;
pub mod sampling;
pub mod transform;

pub use metric::translation_mse_coeus;
pub use mse::mean_squared_error_coeus;
pub use optim::sgd_step_var;
pub use sampling::{sample_linear_1d_coeus, sample_trilinear_coeus};
pub use transform::{affine_transform_coeus, translate_axis_coeus};

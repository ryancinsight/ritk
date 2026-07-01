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
//! - [`sampling`] — differentiable interpolation of a moving-image `Var` at
//!   continuous coordinates; the gradient flows to the coordinate leaf (hence
//!   to the transform parameters that produce the coordinates).
//! - [`mse`] — the terminal intensity-difference loss reduction the sampled
//!   intensities feed into.
//!
//! Composed, `mse(sample(moving, transform(coords)), fixed)` is the
//! differentiable MSE registration metric; the Coeus-native `Metric`/
//! `Transform` trait surface that wraps this composition remains a separate
//! (ADR-gated, `[arch]`) increment.

pub mod mse;
pub mod sampling;

pub use mse::mean_squared_error_coeus;
pub use sampling::{sample_linear_1d_coeus, sample_trilinear_coeus};

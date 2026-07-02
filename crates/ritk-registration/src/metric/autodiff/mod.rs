//! Differentiable registration primitives on the Atlas autodiff engine.
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
//! - [`mse`] / [`ncc`] — the terminal intensity-metric loss reductions the
//!   sampled intensities feed into (MSE; negative normalized cross-correlation).
//! - [`metric`] — the generic end-to-end composition
//!   [`metric::evaluate`] `= metric.reduce(sample(moving, transform(grid)),
//!   fixed)`, dispatching over both seams; `mse_metric`/`affine_mse` are
//!   thin MSE wrappers.
//! - [`optim`] — the `Var`-level gradient-descent step a registration loop
//!   uses to update the transform parameters from the metric gradient.
//! - [`driver`] — [`driver::gradient_descent`], the reusable end-to-end
//!   "run an autodiff registration" entry point composing the above.
//! - [`traits`] — the `Transform` and `Metric` seams
//!   (ADR 0001); [`metric::evaluate`] dispatches over their implementors
//!   ([`transform::Translation`]/[`transform::Affine`], [`mse::Mse`]/
//!   [`ncc::Ncc`]).

pub mod driver;
pub mod metric;
pub mod mse;
pub mod ncc;
pub mod optim;
pub mod sampling;
pub mod traits;
pub mod transform;

pub use driver::{gradient_descent, GradientDescentConfig, RegistrationOutcome};
pub use metric::{affine_mse, evaluate, mse_metric};
pub use mse::{mean_squared_error, Mse};
pub use ncc::{normalized_cross_correlation, Ncc};
pub use optim::sgd_step_var;
pub use sampling::{sample_linear_1d, sample_trilinear};
pub use traits::{Metric, Transform};
pub use transform::{affine_transform, Affine, Translation};

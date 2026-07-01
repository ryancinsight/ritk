//! Coeus-autograd differentiable similarity-loss kernels.
//!
//! Atlas migration (burn → coeus): the first verified increment of the
//! registration-metric autodiff path (`docs/coeus_migration.md`, dev-sequence
//! step 6, gate #3 — "registration metrics preserve autodiff tape
//! connectivity; no host extraction on differentiable paths").
//!
//! [`mean_squared_error_coeus`] is the terminal intensity-difference loss node
//! every intensity metric (MSE, and the numerator/denominator moments of NCC)
//! reduces to. It operates entirely on Coeus autograd [`Var`]s — no host
//! extraction between ops — so the reverse pass propagates gradients into both
//! the moving and the fixed intensity leaves, which is what an upstream
//! differentiable sampler/transform will consume once that increment lands.
//!
//! Scope of this increment (kept honest, per no-mock discipline): this is the
//! loss reduction only. Differentiable image *sampling* (interpolating the
//! moving image at transform-dependent continuous coordinates, the step that
//! makes the loss a function of the transform parameters) is the next gated
//! increment — it depends on Coeus `gather` index semantics and is filed
//! separately, not stubbed here.

use coeus_autograd::{mean, mul, sub, Var};
use coeus_core::{ComputeBackend, Scalar};
use coeus_ops::BackendOps;

/// Differentiable mean squared error between two equal-length intensity
/// vectors, `MSE = mean((moving − fixed)²)`.
///
/// Both inputs are Coeus autograd variables of identical shape; the result is
/// a scalar-shaped (`[1]`) [`Var`] whose creator graph links back to whichever
/// inputs carry `requires_grad`. The computation stays entirely in the autograd
/// graph (`sub` → `mul` → `mean`), so `.backward()` on the result accumulates:
///
/// - `∂MSE/∂moving = (2/N)·(moving − fixed)`
/// - `∂MSE/∂fixed  = −(2/N)·(moving − fixed)`
///
/// with `N` the element count. These are the analytically exact gradients the
/// tests assert against (closed-form oracle, not finite difference alone).
///
/// # Panics
///
/// Panics if the two inputs do not have identical shapes, matching the
/// elementwise `sub` contract (a caller invariant, not input-data error).
pub fn mean_squared_error_coeus<T, B>(moving: &Var<T, B>, fixed: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
{
    let diff = sub(moving, fixed);
    let sq = mul(&diff, &diff);
    mean(&sq)
}

#[cfg(test)]
#[path = "tests_coeus_autograd.rs"]
mod tests;

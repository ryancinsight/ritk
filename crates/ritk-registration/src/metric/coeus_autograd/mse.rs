//! Differentiable mean-squared-error loss reduction on Coeus autograd `Var`s.
//!
//! The terminal intensity-difference loss node every intensity metric (MSE,
//! and the numerator/denominator moments of NCC) reduces to. Stays entirely in
//! the autograd graph — no host extraction between ops (migration gate #3) —
//! so the reverse pass propagates gradients into both intensity leaves. Pairs
//! with [`super::sampling`], which produces the differentiable moving-intensity
//! `Var` this reduction consumes.

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
#[path = "tests_mse.rs"]
mod tests;

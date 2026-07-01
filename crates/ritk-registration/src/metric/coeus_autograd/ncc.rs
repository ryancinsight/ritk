//! Differentiable normalized cross-correlation (NCC) loss reduction on Coeus
//! autograd `Var`s.
//!
//! The second Coeus-native intensity-metric reduction (after [`super::mse`]);
//! together they justify the `CoeusMetric` seam (ADR 0001, deferred to its own
//! increment). Computed via the single-pass algebraic-moments form (Lewis 1995)
//! that the Burn `NormalizedCrossCorrelation` also uses, but entirely on the
//! autograd tape so the reverse pass reaches the sampled-intensity leaf (hence
//! the transform parameters upstream).
//!
//! Returns **negative** NCC as a minimization loss: range `[-1, 1]`, `-1` at
//! perfect correlation.

use coeus_autograd::{div, mul, neg, scalar_add, scalar_div, sqrt, sub, sum, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_ops::BackendOps;

/// Differentiable negative normalized cross-correlation between two equal-length
/// intensity vectors, `loss = −NCC(moving, fixed)`.
///
/// Using raw moments `S_F, S_M, S_FF, S_MM, S_FM` over the `N` elements:
///
/// ```text
/// num = S_FM − S_F·S_M / N
/// d_F = S_FF − S_F² / N,   d_M = S_MM − S_M² / N
/// NCC = num / √(d_F·d_M + ε)
/// ```
///
/// The whole computation stays in the autograd graph (`sum`/`mul`/`sub`/
/// scalar ops/`sqrt`/`div`/`neg`), so `.backward()` accumulates gradients into
/// `moving` and `fixed`. `ε` (a small positive constant added to the variance
/// product) keeps the denominator and its derivative finite for constant or
/// near-constant inputs.
///
/// # Panics
///
/// Panics if the two inputs differ in shape (the elementwise `mul`/`sub`
/// contract) — a caller invariant.
pub fn normalized_cross_correlation_coeus<T, B>(moving: &Var<T, B>, fixed: &Var<T, B>) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = moving.tensor.shape().first().copied().unwrap_or(0) as f64;
    let inv_n_scale = n; // scalar_div divides by this

    let s_f = sum(fixed);
    let s_m = sum(moving);
    let s_ff = sum(&mul(fixed, fixed));
    let s_mm = sum(&mul(moving, moving));
    let s_fm = sum(&mul(fixed, moving));

    // num = S_FM − S_F·S_M / N
    let num = sub(&s_fm, &scalar_div(&mul(&s_f, &s_m), T::from_f64(inv_n_scale)));
    // d_F = S_FF − S_F² / N ; d_M = S_MM − S_M² / N
    let d_f = sub(&s_ff, &scalar_div(&mul(&s_f, &s_f), T::from_f64(inv_n_scale)));
    let d_m = sub(&s_mm, &scalar_div(&mul(&s_m, &s_m), T::from_f64(inv_n_scale)));

    // NCC = num / √(d_F·d_M + ε); loss = −NCC.
    let eps = T::from_f64(1e-10);
    let denominator = sqrt(&scalar_add(&mul(&d_f, &d_m), eps));
    neg(&div(&num, &denominator))
}

#[cfg(test)]
#[path = "tests_ncc.rs"]
mod tests;

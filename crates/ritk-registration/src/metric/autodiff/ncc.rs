//! Differentiable normalized cross-correlation (NCC) loss reduction on Coeus
//! autograd `Var`s.
//!
//! The second intensity-metric reduction (after [`super::mse`]);
//! together they justify the `Metric` seam (ADR 0001, deferred to its own
//! increment). Computed via the single-pass algebraic-moments form (Lewis 1995)
//! that the Coeus `NormalizedCrossCorrelation` also uses, but entirely on the
//! autograd tape so the reverse pass reaches the sampled-intensity leaf (hence
//! the transform parameters upstream).
//!
//! Returns **negative** NCC as a minimization loss: range `[-1, 1]`, `-1` at
//! perfect correlation.

use coeus_autograd::{div, mul, neg, scalar_add, scalar_div, sqrt, sub, sum, Var};
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement,
};
use coeus_ops::BackendOps;

use super::traits::Metric;

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
pub fn normalized_cross_correlation<T, B>(moving: &Var<T, B>, fixed: &Var<T, B>) -> Var<T, B>
where
    T: Float + FloatElement + leto_ops::Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = moving.tensor.shape().first().copied().unwrap_or(0);
    // `scalar_div` divides by this count, converted in `T`'s own precision.
    let n_val = <T as FloatElement>::from_count(n);

    let s_f = sum(fixed).expect("normalized_cross_correlation: backend sum failed");
    let s_m = sum(moving).expect("normalized_cross_correlation: backend sum failed");
    let fixed_sq =
        mul(fixed, fixed).expect("normalized_cross_correlation: backend multiply failed");
    let s_ff = sum(&fixed_sq).expect("normalized_cross_correlation: backend sum failed");
    let moving_sq =
        mul(moving, moving).expect("normalized_cross_correlation: backend multiply failed");
    let s_mm = sum(&moving_sq).expect("normalized_cross_correlation: backend sum failed");
    let fixed_moving =
        mul(fixed, moving).expect("normalized_cross_correlation: backend multiply failed");
    let s_fm = sum(&fixed_moving).expect("normalized_cross_correlation: backend sum failed");

    // num = S_FM − S_F·S_M / N
    let s_f_s_m = mul(&s_f, &s_m).expect("normalized_cross_correlation: backend multiply failed");
    let s_f_s_m_n = scalar_div(&s_f_s_m, n_val)
        .expect("normalized_cross_correlation: backend scalar_div failed");
    let num =
        sub(&s_fm, &s_f_s_m_n).expect("normalized_cross_correlation: backend subtract failed");
    // d_F = S_FF − S_F² / N ; d_M = S_MM − S_M² / N
    let s_f_sq = mul(&s_f, &s_f).expect("normalized_cross_correlation: backend multiply failed");
    let s_f_sq_n = scalar_div(&s_f_sq, n_val)
        .expect("normalized_cross_correlation: backend scalar_div failed");
    let d_f = sub(&s_ff, &s_f_sq_n).expect("normalized_cross_correlation: backend subtract failed");
    let s_m_sq = mul(&s_m, &s_m).expect("normalized_cross_correlation: backend multiply failed");
    let s_m_sq_n = scalar_div(&s_m_sq, n_val)
        .expect("normalized_cross_correlation: backend scalar_div failed");
    let d_m = sub(&s_mm, &s_m_sq_n).expect("normalized_cross_correlation: backend subtract failed");

    // NCC = num / √(d_F·d_M + ε); loss = −NCC.
    let eps = <T as FloatElement>::from_f64(1e-10);
    let d_f_d_m = mul(&d_f, &d_m).expect("normalized_cross_correlation: backend multiply failed");
    let denom_arg =
        scalar_add(&d_f_d_m, eps).expect("normalized_cross_correlation: backend scalar_add failed");
    let denominator = sqrt(&denom_arg).expect("normalized_cross_correlation: backend sqrt failed");

    let ratio =
        div(&num, &denominator).expect("normalized_cross_correlation: backend divide failed");
    neg(&ratio).expect("normalized_cross_correlation: backend negate failed")
}

/// Negative-normalized-cross-correlation metric ([`Metric`] implementor).
/// Zero-sized; the reduction has no configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ncc;

impl<T, B> Metric<T, B> for Ncc
where
    T: Float + FloatElement + leto_ops::Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn reduce(&self, sampled: &Var<T, B>, fixed: &Var<T, B>) -> Var<T, B> {
        normalized_cross_correlation(sampled, fixed)
    }
}

#[cfg(test)]
#[path = "tests_ncc.rs"]
mod tests;

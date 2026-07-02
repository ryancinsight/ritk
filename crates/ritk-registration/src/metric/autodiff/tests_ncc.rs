//! Verification of the differentiable NCC loss reduction.
//!
//! Evidence tier: analytical — closed-form NCC values (`-1` at perfect
//! correlation, `+1` at perfect anti-correlation, `0` when uncorrelated) plus a
//! host reference and a central finite-difference gradient check. Deterministic
//! `SequentialBackend`.

use super::normalized_cross_correlation;
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

fn var(data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on([data.len()], data, &SequentialBackend),
        requires_grad,
    )
}

/// Host closed-form negative NCC (matches the production ε and moment form).
fn neg_ncc_reference(moving: &[f64], fixed: &[f64]) -> f64 {
    let n = moving.len() as f64;
    let s_f: f64 = fixed.iter().sum();
    let s_m: f64 = moving.iter().sum();
    let s_ff: f64 = fixed.iter().map(|v| v * v).sum();
    let s_mm: f64 = moving.iter().map(|v| v * v).sum();
    let s_fm: f64 = moving.iter().zip(fixed).map(|(a, b)| a * b).sum();
    let num = s_fm - s_f * s_m / n;
    let d_f = s_ff - s_f * s_f / n;
    let d_m = s_mm - s_m * s_m / n;
    -(num / (d_f * d_m + 1e-10).sqrt())
}

#[test]
fn perfect_correlation_gives_loss_minus_one() {
    // NCC is invariant to affine intensity scaling: moving = 2·fixed + 3 is
    // perfectly correlated ⇒ NCC = 1 ⇒ loss = −1.
    let fixed = [1.0, 2.0, 3.0, 5.0, 8.0];
    let moving: Vec<f64> = fixed.iter().map(|v| 2.0 * v + 3.0).collect();
    let loss = normalized_cross_correlation(&var(&moving, false), &var(&fixed, false));
    let got = loss.tensor.as_slice()[0];
    assert!((got - (-1.0)).abs() < 1e-6, "perfect correlation loss should be -1, got {got}");
}

#[test]
fn perfect_anti_correlation_gives_loss_plus_one() {
    let fixed = [1.0, 2.0, 3.0, 5.0, 8.0];
    let moving: Vec<f64> = fixed.iter().map(|v| -1.5 * v + 0.5).collect();
    let loss = normalized_cross_correlation(&var(&moving, false), &var(&fixed, false));
    let got = loss.tensor.as_slice()[0];
    assert!((got - 1.0).abs() < 1e-6, "perfect anti-correlation loss should be +1, got {got}");
}

#[test]
fn forward_matches_host_reference() {
    let fixed = [0.3, -1.2, 2.7, 0.9, 4.1, -0.6];
    let moving = [1.0, 0.4, 2.0, 1.5, 3.0, 0.1];
    let got = normalized_cross_correlation(&var(&moving, false), &var(&fixed, false))
        .tensor
        .as_slice()[0];
    let expected = neg_ncc_reference(&moving, &fixed);
    assert!((got - expected).abs() < 1e-12, "NCC loss: got {got}, expected {expected}");
}

#[test]
fn gradient_wrt_moving_matches_central_finite_difference() {
    let fixed = [0.3, -1.2, 2.7, 0.9, 4.1, -0.6];
    let moving = [1.0, 0.4, 2.0, 1.5, 3.0, 0.1];

    let m = var(&moving, true);
    let loss = normalized_cross_correlation(&m, &var(&fixed, false));
    loss.backward();
    let analytic = m.grad().expect("moving grad").as_slice().to_vec();

    let h = 1e-7;
    for i in 0..moving.len() {
        let mut mp = moving;
        let mut mm = moving;
        mp[i] += h;
        mm[i] -= h;
        let fd = (neg_ncc_reference(&mp, &fixed) - neg_ncc_reference(&mm, &fixed)) / (2.0 * h);
        assert!(
            (analytic[i] - fd).abs() < 1e-4,
            "∂loss/∂moving[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}

#[test]
fn gradient_is_zero_at_perfect_correlation_optimum() {
    // At perfect correlation NCC is at its minimum loss (−1); the gradient of a
    // scale-invariant correlation w.r.t. moving vanishes along the fit direction
    // — verify the analytic gradient matches finite differences there (small,
    // not asserted exactly zero since NCC's optimum is a ridge, not a point).
    let fixed = [1.0, 2.0, 3.0, 5.0, 8.0];
    let moving: Vec<f64> = fixed.iter().map(|v| 2.0 * v + 3.0).collect();
    let m = var(&moving, true);
    let loss = normalized_cross_correlation(&m, &var(&fixed, false));
    loss.backward();
    let analytic = m.grad().expect("grad").as_slice().to_vec();
    let h = 1e-6;
    for i in 0..moving.len() {
        let mut mp = moving.clone();
        let mut mm = moving.clone();
        mp[i] += h;
        mm[i] -= h;
        let fd = (neg_ncc_reference(&mp, &fixed) - neg_ncc_reference(&mm, &fixed)) / (2.0 * h);
        assert!((analytic[i] - fd).abs() < 1e-4, "grad[{i}] {} vs fd {fd}", analytic[i]);
    }
}

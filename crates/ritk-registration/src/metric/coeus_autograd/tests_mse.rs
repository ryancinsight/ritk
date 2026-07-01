//! Verification of the Coeus-autograd differentiable MSE loss kernel.
//!
//! Evidence tier: analytical (closed-form value and closed-form gradient),
//! cross-checked with central finite differences. Uses the deterministic
//! single-threaded `SequentialBackend` so the tape and reductions are
//! reproducible (no reduction-order variance to bound).

use super::mean_squared_error_coeus;
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

/// Closed-form MSE over f64 slices, used as the value oracle.
fn mse_reference(moving: &[f64], fixed: &[f64]) -> f64 {
    let n = moving.len() as f64;
    moving
        .iter()
        .zip(fixed.iter())
        .map(|(&m, &f)| (m - f) * (m - f))
        .sum::<f64>()
        / n
}

#[test]
fn forward_value_matches_closed_form() {
    let moving = [1.0, 2.5, -3.0, 4.0, 0.5];
    let fixed = [1.5, 2.0, -2.0, 4.5, -0.5];
    let loss = mean_squared_error_coeus(&var(&moving, false), &var(&fixed, false));
    let got = loss.tensor.as_slice()[0];
    let expected = mse_reference(&moving, &fixed);
    assert!(
        (got - expected).abs() < 1e-12,
        "MSE value: got {got}, expected {expected}"
    );
}

#[test]
fn gradient_wrt_moving_matches_closed_form() {
    let moving = [1.0, 2.5, -3.0, 4.0, 0.5];
    let fixed = [1.5, 2.0, -2.0, 4.5, -0.5];
    let n = moving.len() as f64;

    let m = var(&moving, true);
    let f = var(&fixed, false);
    let loss = mean_squared_error_coeus(&m, &f);
    loss.backward();

    let grad = m.grad().expect("moving requires_grad, grad must exist");
    let grad = grad.as_slice();
    // ∂MSE/∂moving_i = (2/N)·(moving_i − fixed_i)
    for i in 0..moving.len() {
        let expected = 2.0 / n * (moving[i] - fixed[i]);
        assert!(
            (grad[i] - expected).abs() < 1e-12,
            "grad_moving[{i}]: got {}, expected {expected}",
            grad[i]
        );
    }
}

#[test]
fn gradient_wrt_fixed_is_negated_moving_gradient() {
    let moving = [1.0, 2.5, -3.0, 4.0, 0.5];
    let fixed = [1.5, 2.0, -2.0, 4.5, -0.5];
    let n = moving.len() as f64;

    let m = var(&moving, false);
    let f = var(&fixed, true);
    let loss = mean_squared_error_coeus(&m, &f);
    loss.backward();

    let grad = f.grad().expect("fixed requires_grad, grad must exist");
    let grad = grad.as_slice();
    // ∂MSE/∂fixed_i = −(2/N)·(moving_i − fixed_i)
    for i in 0..fixed.len() {
        let expected = -2.0 / n * (moving[i] - fixed[i]);
        assert!(
            (grad[i] - expected).abs() < 1e-12,
            "grad_fixed[{i}]: got {}, expected {expected}",
            grad[i]
        );
    }
}

#[test]
fn gradient_matches_central_finite_difference() {
    let moving = [0.3, -1.2, 2.7, 0.9];
    let fixed = [0.1, -1.0, 2.0, 1.5];

    let m = var(&moving, true);
    let f = var(&fixed, false);
    let loss = mean_squared_error_coeus(&m, &f);
    loss.backward();
    let analytic = m.grad().expect("grad").as_slice().to_vec();

    // Central difference: (f(x+h) − f(x−h)) / 2h per element of `moving`.
    // h chosen for f64: cube-root-of-epsilon regime, well away from
    // catastrophic cancellation for these O(1) magnitudes.
    let h = 1e-6;
    for i in 0..moving.len() {
        let mut mp = moving;
        let mut mm = moving;
        mp[i] += h;
        mm[i] -= h;
        let fd = (mse_reference(&mp, &fixed) - mse_reference(&mm, &fixed)) / (2.0 * h);
        assert!(
            (analytic[i] - fd).abs() < 1e-6,
            "grad[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}

#[test]
fn zero_loss_and_zero_gradient_at_perfect_match() {
    let vals = [1.0, 2.0, 3.0, 4.0];
    let m = var(&vals, true);
    let f = var(&vals, false);
    let loss = mean_squared_error_coeus(&m, &f);
    assert_eq!(loss.tensor.as_slice()[0], 0.0);
    loss.backward();
    for (i, &g) in m.grad().expect("grad").as_slice().iter().enumerate() {
        assert_eq!(g, 0.0, "grad[{i}] should be zero at perfect match");
    }
}

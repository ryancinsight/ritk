//! Verification of the Coeus-autograd `Var` gradient-descent step.
//!
//! Evidence tier: analytical. For `loss = Σ x²`, `∂loss/∂x = 2x`, so one step
//! at learning rate `lr` yields `x' = x − lr·2x`. Deterministic
//! `SequentialBackend`.

use super::sgd_step_var;
use coeus_autograd::{mul, sum, Var};
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

fn var(data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on([data.len()], data, &SequentialBackend),
        requires_grad,
    )
}

#[test]
fn step_moves_parameter_along_negative_gradient() {
    // loss = Σ x²  ⇒  grad = 2x  ⇒  x' = x − lr·2x.
    let x0 = [1.0, -2.0, 3.0];
    let lr = 0.1;
    let x = var(&x0, true);
    let loss = sum(&mul(&x, &x));
    loss.backward();
    let stepped = sgd_step_var(&x, lr);

    let got = stepped.tensor.as_slice();
    for (k, &xv) in x0.iter().enumerate() {
        let expected = xv - lr * 2.0 * xv;
        assert!(
            (got[k] - expected).abs() < 1e-12,
            "stepped[{k}]: got {}, expected {expected}",
            got[k]
        );
    }
}

#[test]
fn stepped_parameter_is_a_fresh_requires_grad_leaf() {
    // The returned leaf must itself accumulate gradients (so the next descent
    // iteration can backprop through it).
    let x = var(&[2.0], true);
    let loss = sum(&mul(&x, &x));
    loss.backward();
    let stepped = sgd_step_var(&x, 0.5);

    // New forward on the stepped parameter: loss2 = Σ (stepped)² ⇒ grad = 2·stepped.
    let loss2 = sum(&mul(&stepped, &stepped));
    loss2.backward();
    let g = stepped.grad().expect("stepped leaf must be requires_grad").as_slice()[0];
    let expected = 2.0 * stepped.tensor.as_slice()[0];
    assert!((g - expected).abs() < 1e-12, "stepped-leaf grad: got {g}, expected {expected}");
}

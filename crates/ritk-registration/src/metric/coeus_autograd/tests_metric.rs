//! End-to-end verification of the differentiable translation-MSE metric.
//!
//! Evidence tier: analytical. On a moving image that is a linear ramp along x
//! (`moving[z,y,x] = x`) with a fixed image that is the same ramp shifted by
//! one voxel in x, the metric and its parameter gradient have closed forms:
//! at translation `tx`, `sampled_k = grid_x_k + tx`, `fixed_k = grid_x_k + 1`,
//! so `loss(tx) = (tx − 1)²` and `∂loss/∂tx = 2(tx − 1)` (= −2 at `tx = 0`).
//! The y/z axes are degenerate (`Z = Y = 1`), so their gradients are exactly
//! zero. Cross-checked with a self-consistent central finite difference on the
//! metric's own forward. Deterministic `SequentialBackend`.

use super::translation_mse_coeus;
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;
const DIMS: [usize; 3] = [1, 1, 6];

fn var(data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on([data.len()], data, &SequentialBackend),
        requires_grad,
    )
}

/// Ramp moving image `moving[x] = x` flattened for `DIMS = [1, 1, 6]`.
fn ramp_moving() -> Vec<f64> {
    (0..6).map(|x| x as f64).collect()
}

/// Evaluate the metric's scalar loss at a given x-translation (y = z = 0),
/// with the fixed image set to the ramp shifted by +1 voxel in x.
fn loss_at_tx(grid_x: &[f64], fixed: &[f64], tx: f64) -> f64 {
    let moving = ramp_moving();
    let n = grid_x.len();
    let zeros = vec![0.0; n];
    let out = translation_mse_coeus(
        &var(&moving, false),
        DIMS,
        &var(fixed, false),
        &var(&zeros, false),
        &var(&zeros, false),
        &var(grid_x, false),
        &var(&[0.0], false),
        &var(&[0.0], false),
        &var(&[tx], false),
    );
    out.tensor.as_slice()[0]
}

#[test]
fn zero_loss_and_zero_gradient_at_identity_alignment() {
    // fixed = moving sampled at the grid itself ⇒ perfectly aligned at t = 0.
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [1.0, 2.0, 3.0]; // ramp f(x) = x at those grid points
    let moving = ramp_moving();
    let zeros = vec![0.0; grid_x.len()];
    let (tz, ty, tx) = (var(&[0.0], true), var(&[0.0], true), var(&[0.0], true));

    let loss = translation_mse_coeus(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var(&zeros, false),
        &var(&zeros, false),
        &var(&grid_x, false),
        &tz,
        &ty,
        &tx,
    );
    assert!(loss.tensor.as_slice()[0].abs() < 1e-12, "loss should be 0 at alignment");
    loss.backward();
    assert!(tx.grad().expect("grad").as_slice()[0].abs() < 1e-12, "∂loss/∂tx ~ 0");
    assert!(ty.grad().expect("grad").as_slice()[0].abs() < 1e-12, "∂loss/∂ty ~ 0");
    assert!(tz.grad().expect("grad").as_slice()[0].abs() < 1e-12, "∂loss/∂tz ~ 0");
}

#[test]
fn gradient_points_toward_alignment_at_known_offset() {
    // fixed = ramp shifted +1 in x ⇒ aligning translation is tx = +1.
    // At tx = 0: closed-form loss = (0 − 1)² = 1, ∂loss/∂tx = 2(0 − 1) = −2
    // (negative ⇒ increasing tx reduces loss ⇒ points toward alignment).
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [2.0, 3.0, 4.0]; // f(x + 1) = x + 1 at those grid points
    let moving = ramp_moving();
    let zeros = vec![0.0; grid_x.len()];
    let (tz, ty, tx) = (var(&[0.0], true), var(&[0.0], true), var(&[0.0], true));

    let loss = translation_mse_coeus(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var(&zeros, false),
        &var(&zeros, false),
        &var(&grid_x, false),
        &tz,
        &ty,
        &tx,
    );
    assert!((loss.tensor.as_slice()[0] - 1.0).abs() < 1e-12, "loss should be 1 at tx=0");
    loss.backward();

    let gx = tx.grad().expect("grad").as_slice()[0];
    assert!((gx - (-2.0)).abs() < 1e-12, "∂loss/∂tx should be -2, got {gx}");
    assert!(gx < 0.0, "gradient must point toward +tx alignment");
    // Degenerate y/z axes contribute no gradient.
    assert!(ty.grad().expect("grad").as_slice()[0].abs() < 1e-12);
    assert!(tz.grad().expect("grad").as_slice()[0].abs() < 1e-12);
}

#[test]
fn tx_gradient_matches_self_consistent_finite_difference() {
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [2.0, 3.0, 4.0];
    let moving = ramp_moving();
    let zeros = vec![0.0; grid_x.len()];
    let tx = var(&[0.0], true);
    let loss = translation_mse_coeus(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var(&zeros, false),
        &var(&zeros, false),
        &var(&grid_x, false),
        &var(&[0.0], false),
        &var(&[0.0], false),
        &tx,
    );
    loss.backward();
    let analytic = tx.grad().expect("grad").as_slice()[0];

    let h = 1e-6;
    let fd = (loss_at_tx(&grid_x, &fixed, h) - loss_at_tx(&grid_x, &fixed, -h)) / (2.0 * h);
    assert!((analytic - fd).abs() < 1e-5, "analytic {analytic}, finite-diff {fd}");
}

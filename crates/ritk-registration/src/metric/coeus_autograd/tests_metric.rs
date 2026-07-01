//! End-to-end verification of the generic `mse_metric` over translation and
//! affine `CoeusTransform`s.
//!
//! Evidence tier: analytical. Translation case: a moving image that is a linear
//! ramp along x (`moving[z,y,x] = x`), a `[1,1,6]` volume so y/z are degenerate,
//! and a fixed image shifted by one voxel in x — giving closed forms
//! `loss(tx) = (tx − 1)²`, `∂loss/∂tx = 2(tx − 1)` (= −2 at `tx = 0`), with the
//! degenerate axes contributing exactly zero gradient. Affine case: a linear
//! field (trilinear-exact) gives a closed-form reference and convex landscape.
//! Cross-checked with self-consistent central finite differences on the
//! metric's own forward. Deterministic `SequentialBackend`.

use super::super::optim::sgd_step_var;
use super::super::transform::Translation;
use super::{affine_mse_coeus, mse_metric};
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

fn var_shaped(shape: &[usize], data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on(shape.to_vec(), data, &SequentialBackend),
        requires_grad,
    )
}

/// Ramp moving image `moving[x] = x` flattened for `DIMS = [1, 1, 6]`.
fn ramp_moving() -> Vec<f64> {
    (0..6).map(|x| x as f64).collect()
}

/// `[N, 3]` grid rows `(z=0, y=0, x)` for the degenerate-axes translation case.
fn x_grid_flat(grid_x: &[f64]) -> Vec<f64> {
    grid_x.iter().flat_map(|&x| [0.0, 0.0, x]).collect()
}

/// Translation-MSE scalar loss at x-translation `tx` (via `mse_metric` +
/// `Translation`), fixed = ramp shifted +1 in x.
fn loss_at_tx(grid_x: &[f64], fixed: &[f64], tx: f64) -> f64 {
    let gf = x_grid_flat(grid_x);
    let out = mse_metric(
        &var(&ramp_moving(), false),
        DIMS,
        &var(fixed, false),
        &var_shaped(&[grid_x.len(), 3], &gf, false),
        &Translation {
            t: var(&[0.0, 0.0, tx], false),
        },
    );
    out.tensor.as_slice()[0]
}

#[test]
fn translation_zero_loss_and_zero_gradient_at_identity_alignment() {
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [1.0, 2.0, 3.0]; // ramp f(x) = x at those grid points
    let gf = x_grid_flat(&grid_x);
    let t = var(&[0.0, 0.0, 0.0], true);
    let loss = mse_metric(
        &var(&ramp_moving(), false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[grid_x.len(), 3], &gf, false),
        &Translation { t: t.clone() },
    );
    assert!(loss.tensor.as_slice()[0].abs() < 1e-12, "loss should be 0 at alignment");
    loss.backward();
    for (j, &g) in t.grad().expect("grad").as_slice().iter().enumerate() {
        assert!(g.abs() < 1e-12, "∂loss/∂t[{j}] should be ~0 at alignment, got {g}");
    }
}

#[test]
fn translation_gradient_points_toward_alignment_at_known_offset() {
    // fixed = ramp shifted +1 in x ⇒ ∂loss/∂t = [0, 0, −2] at t = 0.
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [2.0, 3.0, 4.0];
    let gf = x_grid_flat(&grid_x);
    let t = var(&[0.0, 0.0, 0.0], true);
    let loss = mse_metric(
        &var(&ramp_moving(), false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[grid_x.len(), 3], &gf, false),
        &Translation { t: t.clone() },
    );
    assert!((loss.tensor.as_slice()[0] - 1.0).abs() < 1e-12, "loss should be 1 at t=0");
    loss.backward();
    let g = t.grad().expect("grad").as_slice().to_vec();
    assert!(g[0].abs() < 1e-12 && g[1].abs() < 1e-12, "degenerate axes must have zero gradient");
    assert!((g[2] - (-2.0)).abs() < 1e-12, "∂loss/∂tx should be -2, got {}", g[2]);
    assert!(g[2] < 0.0, "gradient must point toward +tx alignment");
}

#[test]
fn translation_tx_gradient_matches_self_consistent_finite_difference() {
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [2.0, 3.0, 4.0];
    let gf = x_grid_flat(&grid_x);
    let t = var(&[0.0, 0.0, 0.0], true);
    let loss = mse_metric(
        &var(&ramp_moving(), false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[grid_x.len(), 3], &gf, false),
        &Translation { t: t.clone() },
    );
    loss.backward();
    let analytic = t.grad().expect("grad").as_slice()[2];

    let h = 1e-6;
    let fd = (loss_at_tx(&grid_x, &fixed, h) - loss_at_tx(&grid_x, &fixed, -h)) / (2.0 * h);
    assert!((analytic - fd).abs() < 1e-5, "analytic {analytic}, finite-diff {fd}");
}

#[test]
fn translation_gradient_descent_converges_to_the_true_offset() {
    // GD on mse_metric + Translation converges tx → 1 (fixed = ramp shifted +1).
    let grid_x = [1.0, 2.0, 3.0];
    let fixed = [2.0, 3.0, 4.0];
    let gf = x_grid_flat(&grid_x);
    let moving = ramp_moving();
    let lr = 0.25;

    let mut t = var(&[0.0, 0.0, 0.0], true);
    let mut prev_loss = f64::INFINITY;
    for step in 0..20 {
        let loss = mse_metric(
            &var(&moving, false),
            DIMS,
            &var(&fixed, false),
            &var_shaped(&[grid_x.len(), 3], &gf, false),
            &Translation { t: t.clone() },
        );
        let lv = loss.tensor.as_slice()[0];
        assert!(
            lv < prev_loss || lv < 1e-14,
            "loss must strictly decrease each step (step {step}: {lv} !< {prev_loss})"
        );
        prev_loss = lv;
        loss.backward();
        t = sgd_step_var(&t, lr);
    }
    let tx = t.tensor.as_slice()[2];
    assert!((tx - 1.0).abs() < 1e-6, "tx must converge to 1.0, got {tx}");
    assert!(prev_loss < 1e-10, "final loss must be ~0, got {prev_loss}");
}

// ── Affine-MSE metric ────────────────────────────────────────────────────────

const AFF_DIMS: [usize; 3] = [4, 4, 4];

/// Linear moving image `m[z,y,x] = 0.5 + 0.3z + 0.2y + 0.1x`, flattened for
/// `AFF_DIMS`. Trilinear interpolation of a linear field is exact at in-bounds
/// points, so sampling equals the closed-form linear functional — giving an
/// exact reference and a convex optimization landscape.
fn linear_moving() -> Vec<f64> {
    let [nz, ny, nx] = AFF_DIMS;
    let mut v = vec![0.0; nz * ny * nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                v[z * ny * nx + y * nx + x] = 0.5 + 0.3 * z as f64 + 0.2 * y as f64 + 0.1 * x as f64;
            }
        }
    }
    v
}

/// Closed-form value of the linear moving field at a continuous point.
fn linear_value(z: f64, y: f64, x: f64) -> f64 {
    0.5 + 0.3 * z + 0.2 * y + 0.1 * x
}

/// Grid of interior points as `[N,3]` rows `(z, y, x)`; small affine
/// perturbations keep the transformed coordinates strictly in-bounds.
fn grid_points() -> Vec<[f64; 3]> {
    vec![
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.5],
        [1.5, 2.0, 1.0],
        [1.0, 1.5, 2.0],
    ]
}

/// Host affine-then-linear-sample-then-MSE reference (exact for the linear
/// image; `out[n] = m(R·p_n + t)`, `loss = mean((out − fixed)²)`).
fn affine_mse_reference(grid: &[[f64; 3]], fixed: &[f64], r: &[f64; 9], t: &[f64; 3]) -> f64 {
    let mut acc = 0.0;
    for (p, &f) in grid.iter().zip(fixed.iter()) {
        let zp = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
        let yp = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
        let xp = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
        let d = linear_value(zp, yp, xp) - f;
        acc += d * d;
    }
    acc / grid.len() as f64
}

/// Evaluate the affine metric's scalar loss at concrete `r`/`t` (no grad).
fn affine_loss(grid_flat: &[f64], n: usize, fixed: &[f64], r: &[f64; 9], t: &[f64; 3]) -> f64 {
    let moving = linear_moving();
    let out = affine_mse_coeus(
        &var(&moving, false),
        AFF_DIMS,
        &var(fixed, false),
        &var_shaped(&[n, 3], grid_flat, false),
        &var_shaped(&[3, 3], r, false),
        &var(t, false),
    );
    out.tensor.as_slice()[0]
}

#[test]
fn affine_metric_identity_is_zero_loss_and_zero_gradient() {
    let grid = grid_points();
    let n = grid.len();
    let grid_flat: Vec<f64> = grid.iter().flatten().copied().collect();
    // fixed = moving sampled at the grid itself ⇒ aligned at R = I, t = 0.
    let fixed: Vec<f64> = grid.iter().map(|p| linear_value(p[0], p[1], p[2])).collect();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let r = var_shaped(&[3, 3], &identity, true);
    let t = var(&[0.0, 0.0, 0.0], true);
    let loss = affine_mse_coeus(
        &var(&linear_moving(), false),
        AFF_DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &grid_flat, false),
        &r,
        &t,
    );
    assert!(loss.tensor.as_slice()[0].abs() < 1e-12, "identity loss should be 0");
    loss.backward();
    for &g in r.grad().expect("R grad").as_slice() {
        assert!(g.abs() < 1e-10, "identity ∂loss/∂R entry should be ~0, got {g}");
    }
    for &g in t.grad().expect("t grad").as_slice() {
        assert!(g.abs() < 1e-10, "identity ∂loss/∂t entry should be ~0, got {g}");
    }
}

#[test]
fn affine_metric_forward_matches_host_reference() {
    let grid = grid_points();
    let n = grid.len();
    let grid_flat: Vec<f64> = grid.iter().flatten().copied().collect();
    let fixed: Vec<f64> = grid.iter().map(|p| linear_value(p[0], p[1], p[2])).collect();
    // Small rotation-ish perturbation of identity, keeping coords in-bounds.
    let r = [1.0, 0.02, 0.0, 0.01, 1.0, 0.03, 0.0, 0.02, 1.0];
    let t = [0.05, -0.03, 0.04];

    let got = affine_loss(&grid_flat, n, &fixed, &r, &t);
    let expected = affine_mse_reference(&grid, &fixed, &r, &t);
    assert!((got - expected).abs() < 1e-12, "affine metric loss: got {got}, expected {expected}");
}

#[test]
fn affine_metric_gradient_matches_self_consistent_finite_difference() {
    let grid = grid_points();
    let n = grid.len();
    let grid_flat: Vec<f64> = grid.iter().flatten().copied().collect();
    let fixed: Vec<f64> = grid.iter().map(|p| linear_value(p[0], p[1], p[2])).collect();
    let r0 = [1.0, 0.02, 0.0, 0.01, 1.0, 0.03, 0.0, 0.02, 1.0];
    let t0 = [0.05, -0.03, 0.04];

    let r = var_shaped(&[3, 3], &r0, true);
    let t = var(&t0, true);
    let loss = affine_mse_coeus(
        &var(&linear_moving(), false),
        AFF_DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &grid_flat, false),
        &r,
        &t,
    );
    loss.backward();
    let gr = r.grad().expect("R grad").as_slice().to_vec();
    let gt = t.grad().expect("t grad").as_slice().to_vec();

    let h = 1e-6;
    for i in 0..9 {
        let mut rp = r0;
        let mut rm = r0;
        rp[i] += h;
        rm[i] -= h;
        let fd = (affine_loss(&grid_flat, n, &fixed, &rp, &t0)
            - affine_loss(&grid_flat, n, &fixed, &rm, &t0))
            / (2.0 * h);
        assert!((gr[i] - fd).abs() < 1e-5, "∂loss/∂R[{i}]: analytic {}, fd {fd}", gr[i]);
    }
    for i in 0..3 {
        let mut tp = t0;
        let mut tm = t0;
        tp[i] += h;
        tm[i] -= h;
        let fd = (affine_loss(&grid_flat, n, &fixed, &r0, &tp)
            - affine_loss(&grid_flat, n, &fixed, &r0, &tm))
            / (2.0 * h);
        assert!((gt[i] - fd).abs() < 1e-5, "∂loss/∂t[{i}]: analytic {}, fd {fd}", gt[i]);
    }
}

#[test]
fn affine_metric_gradient_descent_reduces_loss_to_zero() {
    // End-to-end optimizability of the affine composition (tape through
    // matmul → slice → reshape → trilinear → mse). R is held at identity; t is
    // optimized to align with a translated target. The linear moving field
    // makes the loss a convex quadratic, so GD robustly drives it to ~0.
    // (The single-ramp field constrains only the combination slope·t, so t is
    // not uniquely recovered — the alignment objective, i.e. loss → 0, is.)
    let grid = grid_points();
    let n = grid.len();
    let grid_flat: Vec<f64> = grid.iter().flatten().copied().collect();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    // Target: fixed = moving sampled at grid translated by t*.
    let t_star = [0.3, -0.2, 0.4];
    let fixed: Vec<f64> = grid
        .iter()
        .map(|p| linear_value(p[0] + t_star[0], p[1] + t_star[1], p[2] + t_star[2]))
        .collect();

    let moving = linear_moving();
    let lr = 0.5;
    let mut t = var(&[0.0, 0.0, 0.0], true);
    let mut prev = f64::INFINITY;
    for step in 0..200 {
        let loss = affine_mse_coeus(
            &var(&moving, false),
            AFF_DIMS,
            &var(&fixed, false),
            &var_shaped(&[n, 3], &grid_flat, false),
            &var_shaped(&[3, 3], &identity, false),
            &t,
        );
        let lv = loss.tensor.as_slice()[0];
        assert!(
            lv <= prev + 1e-12,
            "loss must not increase (step {step}: {lv} > {prev})"
        );
        prev = lv;
        loss.backward();
        t = sgd_step_var(&t, lr);
    }
    assert!(prev < 1e-8, "affine-metric GD must drive loss to ~0, got {prev}");
}

//! Verification of the `CoeusTransform` seam and the generic `mse_metric`
//! dispatching over it (ADR 0001).
//!
//! Evidence tier: differential — the trait path must be value-identical to the
//! verified free-function primitives it wraps; plus a gradient-descent
//! convergence check through the trait. Deterministic `SequentialBackend`.

use super::super::metric::{affine_mse_coeus, evaluate, mse_metric};
use super::super::mse::{mean_squared_error_coeus, Mse};
use super::super::ncc::{normalized_cross_correlation_coeus, Ncc};
use super::super::optim::sgd_step_var;
use super::super::sampling::sample_trilinear_coeus;
use super::super::transform::{affine_transform_coeus, Affine, Translation};
use super::CoeusTransform;
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;
const DIMS: [usize; 3] = [4, 4, 4];

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

fn linear_moving() -> Vec<f64> {
    let [nz, ny, nx] = DIMS;
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

fn grid_flat() -> (Vec<f64>, usize) {
    let grid = [
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.5],
        [1.5, 2.0, 1.0],
        [1.0, 1.5, 2.0],
    ];
    (grid.iter().flatten().copied().collect(), grid.len())
}

#[test]
fn affine_transform_struct_matches_free_function() {
    let (gf, n) = grid_flat();
    let r0 = [1.0, 0.02, 0.0, 0.01, 1.0, 0.03, 0.0, 0.02, 1.0];
    let t0 = [0.05, -0.03, 0.04];
    let grid = var_shaped(&[n, 3], &gf, false);

    let via_fn = affine_transform_coeus(&grid, &var_shaped(&[3, 3], &r0, false), &var(&t0, false));
    let via_trait = Affine {
        r: var_shaped(&[3, 3], &r0, false),
        t: var(&t0, false),
    }
    .transform_points(&grid);

    let a = via_fn.tensor.as_slice();
    let b = via_trait.tensor.as_slice();
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < 1e-12, "affine struct vs fn mismatch at {i}: {x} vs {y}");
    }
}

#[test]
fn translation_struct_adds_offset_to_every_row() {
    let (gf, n) = grid_flat();
    let t0 = [0.3, -0.2, 0.4];
    let grid = var_shaped(&[n, 3], &gf, false);
    let out = Translation { t: var(&t0, false) }.transform_points(&grid);
    let got = out.tensor.as_slice();
    for row in 0..n {
        for j in 0..3 {
            let expected = gf[row * 3 + j] + t0[j];
            assert!(
                (got[row * 3 + j] - expected).abs() < 1e-12,
                "translation row {row} col {j}: got {}, expected {expected}",
                got[row * 3 + j]
            );
        }
    }
}

#[test]
fn translation_parameter_gradient_sums_over_points() {
    // `t` broadcasts across all N points, so ∂(Σ out)/∂t_j = N (the summing
    // backward of the broadcast) — the property the removed per-axis
    // `translate_axis_coeus` used to assert, now on the `Translation` struct.
    use coeus_autograd::sum;
    let (gf, n) = grid_flat();
    let t = var(&[0.0, 0.0, 0.0], true);
    let out = Translation { t: t.clone() }.transform_points(&var_shaped(&[n, 3], &gf, false));
    sum(&out).backward();
    for (j, &g) in t.grad().expect("t grad").as_slice().iter().enumerate() {
        assert!(
            (g - n as f64).abs() < 1e-12,
            "∂(Σout)/∂t[{j}] should equal N={n}, got {g}"
        );
    }
}

#[test]
fn mse_metric_with_affine_matches_affine_mse_free_function() {
    // The generic trait-dispatched metric must equal the free-function path it
    // now delegates through — forward value identical.
    let (gf, n) = grid_flat();
    let moving = linear_moving();
    let fixed: Vec<f64> = (0..n).map(|k| 0.5 + gf[k * 3]).collect(); // arbitrary fixed
    let r0 = [1.0, 0.02, 0.0, 0.01, 1.0, 0.03, 0.0, 0.02, 1.0];
    let t0 = [0.05, -0.03, 0.04];

    let via_free = affine_mse_coeus(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &var_shaped(&[3, 3], &r0, false),
        &var(&t0, false),
    );
    let via_trait = mse_metric(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &Affine {
            r: var_shaped(&[3, 3], &r0, false),
            t: var(&t0, false),
        },
    );
    assert!(
        (via_free.tensor.as_slice()[0] - via_trait.tensor.as_slice()[0]).abs() < 1e-14,
        "trait metric must equal free-function metric"
    );
}

#[test]
fn gradient_descent_through_translation_trait_reduces_loss() {
    // Optimize a Translation via the generic metric; linear field ⇒ convex ⇒
    // GD drives the alignment loss to ~0.
    let (gf, n) = grid_flat();
    let moving = linear_moving();
    // Target: fixed = moving sampled at grid translated by t*.
    let t_star = [0.3, -0.2, 0.4];
    let fixed: Vec<f64> = (0..n)
        .map(|k| {
            0.5 + 0.3 * (gf[k * 3] + t_star[0])
                + 0.2 * (gf[k * 3 + 1] + t_star[1])
                + 0.1 * (gf[k * 3 + 2] + t_star[2])
        })
        .collect();

    let lr = 0.5;
    let mut t = var(&[0.0, 0.0, 0.0], true);
    let mut prev = f64::INFINITY;
    for step in 0..200 {
        let loss = mse_metric(
            &var(&moving, false),
            DIMS,
            &var(&fixed, false),
            &var_shaped(&[n, 3], &gf, false),
            &Translation { t: t.clone() },
        );
        let lv = loss.tensor.as_slice()[0];
        assert!(lv <= prev + 1e-12, "loss must not increase (step {step}: {lv} > {prev})");
        prev = lv;
        loss.backward();
        t = sgd_step_var(&t, lr);
    }
    assert!(prev < 1e-8, "GD through the Translation trait must drive loss to ~0, got {prev}");
}

// ── CoeusMetric seam (Mse / Ncc dispatched via evaluate) ─────────────────────

#[test]
fn evaluate_with_mse_matches_mse_metric() {
    // The generic `evaluate` with the `Mse` reduction must equal the `mse_metric`
    // convenience it now underlies.
    let (gf, n) = grid_flat();
    let moving = linear_moving();
    let fixed: Vec<f64> = (0..n).map(|k| 0.5 + gf[k * 3]).collect();
    let t0 = [0.1, -0.05, 0.2];

    let via_mse_metric = mse_metric(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &Translation { t: var(&t0, false) },
    );
    let via_evaluate = evaluate(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &Mse,
        &Translation { t: var(&t0, false) },
    );
    assert!(
        (via_mse_metric.tensor.as_slice()[0] - via_evaluate.tensor.as_slice()[0]).abs() < 1e-14,
        "evaluate+Mse must equal mse_metric"
    );
}

#[test]
fn evaluate_with_ncc_matches_manual_sample_then_reduce() {
    // `evaluate` with `Ncc` must equal manually sampling at the transformed grid
    // and applying the NCC reduction — verifying the seam dispatches to the NCC
    // reduction, not MSE.
    let (gf, n) = grid_flat();
    let moving = linear_moving();
    let fixed: Vec<f64> = (0..n).map(|k| 0.7 + 0.5 * gf[k * 3 + 1]).collect();
    let t0 = [0.1, -0.05, 0.2];

    let via_evaluate = evaluate(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &Ncc,
        &Translation { t: var(&t0, false) },
    );

    // Manual path: translate grid, split, sample, NCC-reduce.
    let transformed = Translation { t: var(&t0, false) }.transform_points(&var_shaped(&[n, 3], &gf, false));
    let cz: Vec<f64> = (0..n).map(|k| transformed.tensor.as_slice()[k * 3]).collect();
    let cy: Vec<f64> = (0..n).map(|k| transformed.tensor.as_slice()[k * 3 + 1]).collect();
    let cx: Vec<f64> = (0..n).map(|k| transformed.tensor.as_slice()[k * 3 + 2]).collect();
    let sampled = sample_trilinear_coeus(
        &var(&moving, false),
        DIMS,
        &var(&cz, false),
        &var(&cy, false),
        &var(&cx, false),
    );
    let manual = normalized_cross_correlation_coeus(&sampled, &var(&fixed, false));

    assert!(
        (via_evaluate.tensor.as_slice()[0] - manual.tensor.as_slice()[0]).abs() < 1e-12,
        "evaluate+Ncc must equal manual sample-then-NCC-reduce"
    );
    // Sanity: it is NOT the MSE value (the seam actually switched reductions).
    let mse_val = mean_squared_error_coeus(&sampled, &var(&fixed, false)).tensor.as_slice()[0];
    assert!(
        (via_evaluate.tensor.as_slice()[0] - mse_val).abs() > 1e-6,
        "NCC and MSE reductions must differ for this input"
    );
}

#[test]
fn evaluate_ncc_gradient_reaches_affine_r_and_is_shift_invariant_in_t() {
    // The NCC metric's gradient flows through the full tape to the affine `R`
    // (which changes the sampled *pattern*). It is exactly zero w.r.t. `t` here:
    // translating a linear moving field only shifts the sampled mean, and NCC is
    // invariant to additive shifts — a defining property, verified end-to-end.
    let (gf, n) = grid_flat();
    let moving = linear_moving();
    let fixed: Vec<f64> = (0..n).map(|k| 0.4 + 0.3 * gf[k * 3] + 0.1 * gf[k * 3 + 2]).collect();
    let r = var_shaped(&[3, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], true);
    let t = var(&[0.05, 0.0, -0.05], true);

    let loss = evaluate(
        &var(&moving, false),
        DIMS,
        &var(&fixed, false),
        &var_shaped(&[n, 3], &gf, false),
        &Ncc,
        &Affine { r: r.clone(), t: t.clone() },
    );
    loss.backward();

    let gr = r.grad().expect("R grad").as_slice().to_vec();
    let gt = t.grad().expect("t grad").as_slice().to_vec();
    assert!(gr.iter().any(|g| g.abs() > 1e-9), "NCC gradient must reach R");
    assert!(gr.iter().chain(gt.iter()).all(|g| g.is_finite()), "all gradients finite");
    // NCC shift-invariance: translating a linear field is a pure mean shift.
    for (j, &g) in gt.iter().enumerate() {
        assert!(g.abs() < 1e-9, "NCC ∂loss/∂t[{j}] should be ~0 (shift-invariance), got {g}");
    }
}

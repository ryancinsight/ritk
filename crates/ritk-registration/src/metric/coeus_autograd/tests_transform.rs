//! Verification of the differentiable per-axis translation primitive.
//!
//! Evidence tier: analytical. `out = coords + t`, so `∂out_k/∂t = 1` for every
//! point, and a `sum` loss gives `∂(Σ out)/∂t = N`. Deterministic
//! `SequentialBackend`.

use super::{affine_transform_coeus, translate_axis_coeus};
use coeus_autograd::{sum, Var};
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

fn var(data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on([data.len()], data, &SequentialBackend),
        requires_grad,
    )
}

/// Build a rank-`shape` `Var` from flat row-major data.
fn var_shaped(shape: &[usize], data: &[f64], requires_grad: bool) -> Var<f64, B> {
    Var::new(
        Tensor::<f64, B>::from_slice_on(shape.to_vec(), data, &SequentialBackend),
        requires_grad,
    )
}

#[test]
fn forward_adds_translation_to_every_coordinate() {
    let coords = [0.0, 1.0, 2.5, -3.0];
    let out = translate_axis_coeus(&var(&coords, false), &var(&[1.5], false));
    let got = out.tensor.as_slice();
    for (k, &c) in coords.iter().enumerate() {
        assert!((got[k] - (c + 1.5)).abs() < 1e-12, "out[{k}]: got {}", got[k]);
    }
}

#[test]
fn parameter_gradient_is_the_point_count() {
    // ∂(Σ_k (coords_k + t))/∂t = Σ_k 1 = N.
    let coords = [0.0, 1.0, 2.0, 3.0, 4.0];
    let n = coords.len() as f64;
    let t = var(&[0.0], true);
    let out = translate_axis_coeus(&var(&coords, false), &t);
    sum(&out).backward();
    let g = t.grad().expect("t requires_grad").as_slice()[0];
    assert!((g - n).abs() < 1e-12, "∂(Σout)/∂t should equal N={n}, got {g}");
}

#[test]
fn coordinate_leaf_also_receives_gradient() {
    // add is differentiable in both operands: ∂(Σ out)/∂coords_k = 1.
    let coords = var(&[0.0, 1.0, 2.0], true);
    let out = translate_axis_coeus(&coords, &var(&[0.5], false));
    sum(&out).backward();
    for (k, &g) in coords.grad().expect("grad").as_slice().iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-12, "∂out/∂coords[{k}] should be 1, got {g}");
    }
}

// ── Affine transform ─────────────────────────────────────────────────────────

/// Host affine reference: out[n,j] = Σ_k coords[n,k]·R[j,k] + t[j].
fn affine_reference(coords: &[[f64; 3]], r: &[f64; 9], t: &[f64; 3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(coords.len() * 3);
    for p in coords {
        for j in 0..3 {
            let v = r[j * 3] * p[0] + r[j * 3 + 1] * p[1] + r[j * 3 + 2] * p[2] + t[j];
            out.push(v);
        }
    }
    out
}

#[test]
fn affine_forward_matches_host_reference_under_rotation_shear() {
    // A 90° rotation about z (x→y, y→−x) plus a shear and non-unit scale, so
    // every R entry participates — the discriminating case for the matmul path.
    #[rustfmt::skip]
    let r = [
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.5,
        0.0,  0.0, 2.0,
    ];
    let t = [1.0, -2.0, 0.5];
    let coords = [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0], [0.0, 0.0, 0.0]];
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();

    let out = affine_transform_coeus(
        &var_shaped(&[coords.len(), 3], &flat, false),
        &var_shaped(&[3, 3], &r, false),
        &var(&t, false),
    );
    let got = out.tensor.as_slice();
    let expected = affine_reference(&coords, &r, &t);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-12, "affine[{i}]: got {g}, expected {e}");
    }
}

#[test]
fn affine_translation_gradient_is_the_point_count() {
    // loss = Σ out ⇒ ∂loss/∂t_j = Σ_n 1 = N (t_j adds to every point's j-th out).
    let coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let n = coords.len() as f64;
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let t = var(&[0.0, 0.0, 0.0], true);
    let out = affine_transform_coeus(
        &var_shaped(&[coords.len(), 3], &flat, false),
        &var_shaped(&[3, 3], &identity, false),
        &t,
    );
    sum(&out).backward();
    for (j, &g) in t.grad().expect("t grad").as_slice().iter().enumerate() {
        assert!((g - n).abs() < 1e-12, "∂(Σout)/∂t[{j}] should be N={n}, got {g}");
    }
}

#[test]
fn affine_matrix_gradient_matches_closed_form_and_finite_difference() {
    // loss = Σ out. ∂loss/∂R[j,k] = Σ_n coords[n,k] (independent of j since each
    // output row j sums coords·R[j,:]). Verify closed form + finite difference.
    let coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 2.0]];
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    let col_sums = [
        coords.iter().map(|p| p[0]).sum::<f64>(),
        coords.iter().map(|p| p[1]).sum::<f64>(),
        coords.iter().map(|p| p[2]).sum::<f64>(),
    ];
    let r0 = [0.2, -0.5, 1.0, 0.3, 0.8, -0.1, 1.0, 0.0, 0.5];
    let t0 = [0.0, 0.0, 0.0];

    let r = var_shaped(&[3, 3], &r0, true);
    let out = affine_transform_coeus(
        &var_shaped(&[coords.len(), 3], &flat, false),
        &r,
        &var(&t0, false),
    );
    sum(&out).backward();
    let grad = r.grad().expect("R grad").as_slice().to_vec();

    for j in 0..3 {
        for k in 0..3 {
            let analytic_expected = col_sums[k];
            let g = grad[j * 3 + k];
            assert!(
                (g - analytic_expected).abs() < 1e-10,
                "∂loss/∂R[{j},{k}]: got {g}, expected {analytic_expected}"
            );
            // Self-consistent central finite difference on Σ out.
            let h = 1e-6;
            let mut rp = r0;
            let mut rm = r0;
            rp[j * 3 + k] += h;
            rm[j * 3 + k] -= h;
            let fd = (affine_reference(&coords, &rp, &t0).iter().sum::<f64>()
                - affine_reference(&coords, &rm, &t0).iter().sum::<f64>())
                / (2.0 * h);
            assert!((g - fd).abs() < 1e-5, "R[{j},{k}] grad {g} vs fd {fd}");
        }
    }
}

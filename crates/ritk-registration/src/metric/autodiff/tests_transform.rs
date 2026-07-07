//! Verification of the differentiable affine point transform.
//!
//! Evidence tier: analytical (host reference + closed-form / finite-difference
//! parameter gradients). Deterministic `SequentialBackend`. The `Translation`
//! and `Affine` `Transform` impls are exercised in `tests_traits.rs`.

use super::affine_transform;
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

    let out = affine_transform(
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
    let out = affine_transform(
        &var_shaped(&[coords.len(), 3], &flat, false),
        &var_shaped(&[3, 3], &identity, false),
        &t,
    );
    sum(&out).backward();
    for (j, &g) in t.grad().expect("t grad").as_slice().iter().enumerate() {
        assert!(
            (g - n).abs() < 1e-12,
            "∂(Σout)/∂t[{j}] should be N={n}, got {g}"
        );
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
    let out = affine_transform(
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

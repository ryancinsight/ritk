//! Verification of the differentiable per-axis translation primitive.
//!
//! Evidence tier: analytical. `out = coords + t`, so `∂out_k/∂t = 1` for every
//! point, and a `sum` loss gives `∂(Σ out)/∂t = N`. Deterministic
//! `SequentialBackend`.

use super::translate_axis_coeus;
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

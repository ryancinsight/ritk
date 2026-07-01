//! Verification of the Coeus-autograd differentiable 1-D linear sampler.
//!
//! Evidence tier: analytical. For a linear ramp `signal[i] = a + b·i`, the
//! interpolated value at continuous `x` is `a + b·x` and its coordinate
//! gradient is the constant slope `b` — a closed-form oracle. Also asserts the
//! `gather` value-gradient path, edge-clamp behavior, and a finite-difference
//! cross-check on a non-ramp signal. Deterministic `SequentialBackend`.

use super::sample_linear_1d_coeus;
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

/// Reference: linear interpolation of `signal` at `x` with edge clamping.
fn interp_reference(signal: &[f64], x: f64) -> f64 {
    let max_index = signal.len() - 1;
    let floor = x.floor();
    let f = x - floor;
    let i0 = if floor <= 0.0 {
        0
    } else if floor >= max_index as f64 {
        max_index
    } else {
        floor as usize
    };
    let i1 = if floor + 1.0 <= 0.0 {
        0
    } else if floor + 1.0 >= max_index as f64 {
        max_index
    } else {
        (floor + 1.0) as usize
    };
    signal[i0] * (1.0 - f) + signal[i1] * f
}

#[test]
fn forward_matches_linear_interp_reference() {
    let signal: Vec<f64> = (0..8).map(|i| 2.0 + 0.5 * i as f64).collect();
    let coords = [0.0, 1.5, 3.25, 6.999];
    let out = sample_linear_1d_coeus(&var(&signal, false), &var(&coords, false));
    let got = out.tensor.as_slice();
    for (k, &x) in coords.iter().enumerate() {
        let expected = interp_reference(&signal, x);
        assert!(
            (got[k] - expected).abs() < 1e-12,
            "sample[{k}] at x={x}: got {}, expected {expected}",
            got[k]
        );
    }
}

#[test]
fn coordinate_gradient_of_ramp_is_the_slope() {
    // signal[i] = a + b·i  ⇒  ∂(sample at x)/∂x = b for all in-bounds x.
    let a = 2.0;
    let b = 0.5;
    let signal: Vec<f64> = (0..8).map(|i| a + b * i as f64).collect();
    let coords = [0.0, 1.5, 3.25, 6.5];

    let c = var(&coords, true);
    let out = sample_linear_1d_coeus(&var(&signal, false), &c);
    sum(&out).backward();

    let grad = c.grad().expect("coords requires_grad").as_slice().to_vec();
    for (k, &g) in grad.iter().enumerate() {
        assert!(
            (g - b).abs() < 1e-12,
            "∂sample/∂x[{k}] should equal ramp slope {b}, got {g}"
        );
    }
}

#[test]
fn coordinate_gradient_matches_central_finite_difference() {
    // Non-ramp signal so the slope varies per interval.
    let signal = [1.0, 4.0, 2.0, 9.0, 3.0, 7.0];
    let coords = [0.3, 1.8, 2.5, 4.1];

    let c = var(&coords, true);
    let out = sample_linear_1d_coeus(&var(&signal, false), &c);
    sum(&out).backward();
    let analytic = c.grad().expect("grad").as_slice().to_vec();

    let h = 1e-6;
    for (k, &x) in coords.iter().enumerate() {
        let fd = (interp_reference(&signal, x + h) - interp_reference(&signal, x - h)) / (2.0 * h);
        assert!(
            (analytic[k] - fd).abs() < 1e-5,
            "grad[{k}]: analytic {}, finite-diff {fd}",
            analytic[k]
        );
    }
}

#[test]
fn gather_gradient_flows_to_signal() {
    // Sample exactly at integer coordinate 2.0: weight w1=0, w0=1, so
    // out = signal[2]. ∂(sum out)/∂signal[2] = 1, all other entries 0.
    let signal = [10.0, 11.0, 12.0, 13.0, 14.0];
    let s = var(&signal, true);
    let out = sample_linear_1d_coeus(&s, &var(&[2.0], false));
    sum(&out).backward();
    let grad = s.grad().expect("signal requires_grad").as_slice().to_vec();
    let expected = [0.0, 0.0, 1.0, 0.0, 0.0];
    for (i, (&g, &e)) in grad.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "signal grad[{i}]: got {g}, expected {e}"
        );
    }
}

#[test]
fn edge_clamp_extrapolates_flat_with_zero_gradient() {
    // Below the signal both corner indices clamp to 0, so out = signal[0] and
    // the coordinate gradient (signal[i1] − signal[i0]) is exactly zero.
    let signal = [3.0, 5.0, 9.0];
    let c = var(&[-1.5], true);
    let out = sample_linear_1d_coeus(&var(&signal, false), &c);
    assert!((out.tensor.as_slice()[0] - 3.0).abs() < 1e-12);
    sum(&out).backward();
    assert!(
        c.grad().expect("grad").as_slice()[0].abs() < 1e-12,
        "flat extrapolation must have zero coordinate gradient"
    );
}

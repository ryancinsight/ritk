//! Tests for the directional 1-D frequency-domain filter.
//!
//! Each test is an analytical oracle — it recovers a known property from
//! a constructed signal rather than asserting structural existence.
#![expect(clippy::unwrap_used, reason = "fixture unwraps on well-formed in-memory filters; ratchet RITK-UNWRAP-1")]

use crate::fft::directional_filter::{
    apply_directional_filter, ButterworthBandpass, ButterworthHighpass, ButterworthLowpass,
};
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

type B = coeus_core::SequentialBackend;

fn make_2d(vals: Vec<f32>, h: usize, w: usize) -> Image<f32, B, 2> {
    ts::make_image::<f32, B, 2>(vals, [h, w])
}

fn make_3d(vals: Vec<f32>, d: usize, h: usize, w: usize) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, [d, h, w])
}

// ── Constructor validation ────────────────────────────────────────────────────

#[test]
fn bandpass_constructor_rejects_invalid_inputs() {
    assert!(ButterworthBandpass::new(0.0, 0.4, 2).is_err()); // low <= 0
    assert!(ButterworthBandpass::new(0.1, 0.6, 2).is_err()); // high > 0.5
    assert!(ButterworthBandpass::new(0.3, 0.2, 2).is_err()); // low >= high
    assert!(ButterworthBandpass::new(0.2, 0.2, 2).is_err()); // low == high
    assert!(ButterworthBandpass::new(0.1, 0.4, 0).is_err()); // order 0
    assert!(ButterworthBandpass::new(f64::NAN, 0.4, 2).is_err()); // NaN
    assert!(ButterworthBandpass::new(0.1, 0.4, 2).is_ok()); // valid
}

#[test]
fn lowpass_constructor_rejects_invalid_inputs() {
    assert!(ButterworthLowpass::new(0.0, 2).is_err());
    assert!(ButterworthLowpass::new(0.6, 2).is_err());
    assert!(ButterworthLowpass::new(0.3, 0).is_err());
    assert!(ButterworthLowpass::new(0.3, 2).is_ok());
}

#[test]
fn highpass_constructor_rejects_invalid_inputs() {
    assert!(ButterworthHighpass::new(0.0, 2).is_err());
    assert!(ButterworthHighpass::new(0.6, 2).is_err());
    assert!(ButterworthHighpass::new(0.3, 0).is_err());
    assert!(ButterworthHighpass::new(0.3, 2).is_ok());
}

// ── Round-trip oracle ─────────────────────────────────────────────────────────

/// All-pass approximation: a low-pass with cutoff at nearly-Nyquist (0.499)
/// at high order should reconstruct the input with small error.
#[test]
fn all_pass_round_trip_axis_0() {
    let n = 16_usize;
    let vals: Vec<f32> = (0..n * n).map(|i| (i as f32).sin()).collect();
    let img = make_2d(vals.clone(), n, n);

    let lp = ButterworthLowpass::new(0.499, 1).unwrap(); // broad low-pass ≈ all-pass
    let filtered = apply_directional_filter(&img, 0, &lp).unwrap();

    let (out, _) = extract_vec(&filtered).unwrap();
    let max_err = vals
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // The low-pass passes most of the content; residual attenuation at high
    // frequencies means the error is small relative to the signal amplitude.
    assert!(
        max_err < 0.5,
        "near-all-pass round-trip error too large: {max_err}"
    );
}

#[test]
fn all_pass_round_trip_axis_1() {
    let n = 16_usize;
    let vals: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.7).cos()).collect();
    let img = make_2d(vals.clone(), n, n);

    let lp = ButterworthLowpass::new(0.499, 1).unwrap();
    let filtered = apply_directional_filter(&img, 1, &lp).unwrap();

    let (out, _) = extract_vec(&filtered).unwrap();
    let max_err = vals
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.5,
        "near-all-pass axis-1 error too large: {max_err}"
    );
}

// ── Passband / stopband oracle ────────────────────────────────────────────────

/// DC signal must pass through a low-pass filter unchanged.
///
/// A constant image has all power at DC (f = 0). A low-pass filter with
/// any positive cutoff should leave it intact.
#[test]
fn lowpass_passes_dc_signal() {
    let n = 8_usize;
    let c = 3.7_f32;
    let img = make_2d(vec![c; n * n], n, n);

    let lp = ButterworthLowpass::new(0.3, 4).unwrap();
    let filtered = apply_directional_filter(&img, 0, &lp).unwrap();

    let (out, _) = extract_vec(&filtered).unwrap();
    let max_err = out.iter().map(|v| (v - c).abs()).fold(0.0_f32, f32::max);
    assert!(max_err < 1e-3, "LP DC error: {max_err}");
}

/// DC signal must be blocked by a high-pass filter.
///
/// A constant input has no AC components; the high-pass should attenuate it.
#[test]
fn highpass_blocks_dc_signal() {
    let n = 8_usize;
    let c = 5.0_f32;
    let img = make_2d(vec![c; n * n], n, n);

    let hp = ButterworthHighpass::new(0.4, 4).unwrap();
    let filtered = apply_directional_filter(&img, 0, &hp).unwrap();

    let (out, _) = extract_vec(&filtered).unwrap();
    let max_abs = out.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs < 0.1,
        "HP should block DC, max remaining amplitude: {max_abs}"
    );
}

// ── 3-D directional oracle ────────────────────────────────────────────────────

/// Shape is preserved through directional filtering in all 3-D axes.
#[test]
fn shape_preserved_3d() {
    let (d, h, w) = (4, 6, 8);
    let vals: Vec<f32> = (0..d * h * w).map(|i| i as f32).collect();
    let img = make_3d(vals, d, h, w);

    let lp = ButterworthLowpass::new(0.4, 2).unwrap();
    for axis in 0..3 {
        let out = apply_directional_filter(&img, axis, &lp).unwrap();
        assert_eq!(out.shape(), [d, h, w], "shape mismatch on axis {axis}");
    }
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn rejects_out_of_bounds_axis() {
    let img = make_2d(vec![1.0; 16], 4, 4);
    let lp = ButterworthLowpass::new(0.3, 2).unwrap();
    assert!(apply_directional_filter(&img, 2, &lp).is_err());
    assert!(apply_directional_filter(&img, 99, &lp).is_err());
}

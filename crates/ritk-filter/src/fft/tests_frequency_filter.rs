//! Tests for frequency-domain filters.
//!
//! Each test is derived from a mathematically verifiable property of the
//! frequency-domain filtering pipeline.

use crate::fft::frequency_filter::{FftFilterKind, FrequencyDomainFilter};
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

type B = LegacyBurnBackend;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_image_2d(vals: Vec<f32>, h: usize, w: usize) -> Image<B, 2> {
    ts::make_image::<B, 2>(vals, [h, w])
}

fn make_image_3d(vals: Vec<f32>, d: usize, h: usize, w: usize) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, [d, h, w])
}

// ── 2-D tests ─────────────────────────────────────────────────────────────────

/// `apply_2d` must preserve the input spatial shape `[H, W]`.
#[test]
fn shape_preserved_through_filter() {
    let img = make_image_2d(vec![1.0_f32; 64], 8, 8);
    for kind in &[
        FftFilterKind::IdealLowPass,
        FftFilterKind::IdealHighPass,
        FftFilterKind::ButterworthLowPass,
        FftFilterKind::ButterworthHighPass,
    ] {
        let result = FrequencyDomainFilter::new()
            .apply(&img, *kind, 0.3, 2)
            .unwrap();
        assert_eq!(
            result.shape(),
            [8_usize, 8_usize],
            "shape must be preserved for {kind:?}"
        );
    }
}

/// Ideal low-pass filter with a generous cutoff preserves a constant image.
///
/// Proof: A constant image has all energy at DC (zero frequency).
/// An ideal low-pass filter with `cutoff > 0` passes DC.  Therefore,
/// the output is the original constant within floating-point precision.
///
/// Tolerance: 1e-4 (round-trip FFT error on constant data).
#[test]
fn constant_image_preserved_by_low_pass() {
    let img = make_image_2d(vec![42.0_f32; 64], 8, 8);
    let result = FrequencyDomainFilter::new()
        .apply(&img, FftFilterKind::IdealLowPass, 0.3, 2)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 42.0).abs() < 1e-4,
            "low-pass must preserve constant at index {i}: expected 42.0, got {v:.6}"
        );
    }
}

/// Ideal high-pass filter removes all energy from a constant image.
///
/// Proof: A constant image has zero energy at all non-DC frequencies.
/// An ideal high-pass filter with `cutoff > 0` blocks DC.  Therefore,
/// the output is all zeros.
///
/// Tolerance: 1e-4 (round-trip FFT error).
#[test]
fn constant_image_removed_by_high_pass() {
    let img = make_image_2d(vec![42.0_f32; 64], 8, 8);
    let result = FrequencyDomainFilter::new()
        .apply(&img, FftFilterKind::IdealHighPass, 0.3, 2)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-4,
            "high-pass must remove constant at index {i}: expected 0.0, got {v:.6}"
        );
    }
}

/// Ideal low-pass filter attenuates a checkerboard pattern.
///
/// A checkerboard `(-1)^(i+j)` has its dominant energy at the highest
/// frequencies (near Nyquist, radius ≈ 0.354 in normalised coordinates).
/// An ideal low-pass with `cutoff = 0.2` blocks these high frequencies,
/// producing a near-zero output.
///
/// Tolerance: RMS < 0.5 (the original RMS is 1.0 for the ±1 pattern).
#[test]
fn checkerboard_attenuated_by_low_pass() {
    let vals: Vec<f32> = (0..64)
        .map(|i| {
            let r = i / 8;
            let c = i % 8;
            if (r + c) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let img = make_image_2d(vals, 8, 8);
    let result = FrequencyDomainFilter::new()
        .apply(&img, FftFilterKind::IdealLowPass, 0.2, 2)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    assert!(
        rms < 0.5,
        "low-pass must attenuate checkerboard (high-frequency pattern): RMS = {rms:.6}"
    );
}

/// Butterworth low-pass filter output is finite (sanity check).
#[test]
fn butterworth_low_pass_gives_finite_output() {
    let vals: Vec<f32> = (0..100).map(|i| (i as f64 * 0.314).sin() as f32).collect();
    let img = make_image_2d(vals, 10, 10);
    let result = FrequencyDomainFilter::new()
        .apply(&img, FftFilterKind::ButterworthLowPass, 0.3, 4)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Butterworth low-pass output at index {i} must be finite, got {v}"
        );
    }
}

/// Butterworth high-pass filter output is finite (sanity check).
#[test]
fn butterworth_high_pass_gives_finite_output() {
    let vals: Vec<f32> = (0..100).map(|i| (i as f64 * 0.314).sin() as f32).collect();
    let img = make_image_2d(vals, 10, 10);
    let result = FrequencyDomainFilter::new()
        .apply(&img, FftFilterKind::ButterworthHighPass, 0.3, 4)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Butterworth high-pass output at index {i} must be finite, got {v}"
        );
    }
}

// ── 3-D tests ─────────────────────────────────────────────────────────────────

/// `apply_3d` must preserve the input shape `[D, H, W]`.
#[test]
fn shape_preserved_through_filter_volume() {
    let vol = make_image_3d(vec![1.0_f32; 64], 4, 4, 4);
    for kind in &[
        FftFilterKind::IdealLowPass,
        FftFilterKind::IdealHighPass,
        FftFilterKind::ButterworthLowPass,
        FftFilterKind::ButterworthHighPass,
    ] {
        let result = FrequencyDomainFilter::new()
            .apply(&vol, *kind, 0.3, 2)
            .unwrap();
        assert_eq!(
            result.shape(),
            [4_usize, 4_usize, 4_usize],
            "3-D shape must be preserved for {kind:?}"
        );
    }
}

/// Ideal low-pass preserves a constant volume.
#[test]
fn constant_volume_preserved_by_low_pass() {
    let vol = make_image_3d(vec![std::f32::consts::PI; 64], 4, 4, 4);
    let result = FrequencyDomainFilter::new()
        .apply(&vol, FftFilterKind::IdealLowPass, 0.3, 2)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - std::f32::consts::PI).abs() < 1e-4,
            "3-D low-pass must preserve constant at index {i}: expected 3.14, got {v:.6}"
        );
    }
}

/// Ideal high-pass removes a constant volume.
#[test]
fn constant_volume_removed_by_high_pass() {
    let vol = make_image_3d(vec![std::f32::consts::PI; 64], 4, 4, 4);
    let result = FrequencyDomainFilter::new()
        .apply(&vol, FftFilterKind::IdealHighPass, 0.3, 2)
        .unwrap();
    let (out, _) = extract_vec(&result).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-4,
            "3-D high-pass must remove constant at index {i}: expected 0.0, got {v:.6}"
        );
    }
}

/// Invalid cutoff must be rejected.
#[test]
fn invalid_cutoff_rejected() {
    let img = make_image_2d(vec![1.0_f32; 16], 4, 4);
    for bad_cutoff in &[0.0, -0.1, 0.6, 1.0] {
        assert!(
            FrequencyDomainFilter::new()
                .apply(&img, FftFilterKind::IdealLowPass, *bad_cutoff, 2)
                .is_err(),
            "cutoff = {bad_cutoff} must be rejected"
        );
    }
}

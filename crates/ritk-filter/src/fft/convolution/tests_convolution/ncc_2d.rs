//! Tests for 2-D `FftNormalizedCorrelationFilter`.

use super::conv_2d::make_image_2d;
use crate::fft::FftNormalizedCorrelationFilter;
use ritk_core::filter::ops::extract_vec;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// Cross-correlation output shape must equal input shape.
#[test]
fn ncc_output_shape_matches_input() {
    let img = make_image_2d(vec![1.0_f32; 64], 8, 8);
    let tmpl = make_image_2d(vec![1.0_f32; 9], 3, 3);

    let result = FftNormalizedCorrelationFilter::<B>::new(&tmpl)
        .unwrap()
        .apply(&img)
        .unwrap();

    assert_eq!(
        result.shape(),
        [8_usize, 8_usize],
        "NCC output shape must equal input shape"
    );
}

/// Cross-correlation of a constant image with a constant template is zero
/// because the template is mean-subtracted (T̂ = T − mean(T) = 0).
///
/// Proof: T̂ = 1 − 1 = 0; FFT(T̂) = 0; out = IFFT(FFT(I) · 0) / ‖0‖ = 0.
/// (When template_norm = 0, the implementation returns 0 by convention.)
#[test]
fn ncc_zero_mean_template_gives_zero_output() {
    let img = make_image_2d(vec![3.0_f32; 36], 6, 6);
    let tmpl = make_image_2d(vec![2.0_f32; 9], 3, 3); // constant → T̂ = 0

    let result = FftNormalizedCorrelationFilter::<B>::new(&tmpl)
        .unwrap()
        .apply(&img)
        .unwrap();

    let (out_vals, _) = extract_vec(&result).unwrap();
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "constant template → zero-mean → NCC must be 0 at index {i}, got {v:.8}"
        );
    }
}

/// Cross-correlation output is finite for a realistic image and non-trivial template.
#[test]
fn ncc_output_is_finite() {
    let img_vals: Vec<f32> = (0..100).map(|i| (i as f32 * 0.314).sin()).collect();
    let img = make_image_2d(img_vals, 10, 10);

    // Non-constant template so T̂ ≠ 0.
    let tmpl_vals: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let tmpl = make_image_2d(tmpl_vals, 3, 3);

    let result = FftNormalizedCorrelationFilter::<B>::new(&tmpl)
        .unwrap()
        .apply(&img)
        .unwrap();

    let (out_vals, _) = extract_vec(&result).unwrap();
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "NCC output at index {i} must be finite, got {v}"
        );
    }
}

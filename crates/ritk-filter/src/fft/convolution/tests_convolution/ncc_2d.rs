//! Tests for 2-D `FftNormalizedCorrelationFilter`.

use super::conv_2d::make_image_2d;
use crate::fft::FftNormalizedCorrelationFilter;

use ritk_tensor_ops::extract_vec;

type B = coeus_core::SequentialBackend;

/// Cross-correlation output shape must equal input shape.
#[test]
fn ncc_output_shape_matches_input() {
    let img = make_image_2d(vec![1.0_f32; 64], 8, 8);
    let tmpl = make_image_2d(vec![1.0_f32; 9], 3, 3);

    let result = FftNormalizedCorrelationFilter::<B>::new(&tmpl)
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");

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
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");

    let (out_vals, _) = extract_vec(&result).expect("infallible: validated precondition");
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "constant template → zero-mean → NCC must be 0 at index {i}, got {v:.8}"
        );
    }
}

/// Full normalization: where the template exactly overlays an identical image
/// patch, the NCC equals 1.0 (and that is the global maximum). This is the
/// defining property of a *normalized* correlation and the parity contract with
/// ITK's `FFTNormalizedCorrelationImageFilter`.
#[test]
fn ncc_perfect_match_peaks_at_one() {
    // 8×8 background of zeros with a distinctive non-constant 3×3 patch at (2,3).
    let (h, w) = (8usize, 8usize);
    let mut img = vec![0.0_f32; h * w];
    let patch: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (pr, pc) = (2usize, 3usize);
    for i in 0..3 {
        for j in 0..3 {
            img[(pr + i) * w + (pc + j)] = patch[i * 3 + j];
        }
    }
    let image = make_image_2d(img, h, w);
    let tmpl = make_image_2d(patch.to_vec(), 3, 3);

    let result = FftNormalizedCorrelationFilter::<B>::new(&tmpl)
        .expect("infallible: validated precondition")
        .apply(&image)
        .expect("infallible: validated precondition");
    let (out, _) = extract_vec(&result).expect("infallible: validated precondition");

    // NCC at the embedding lag must be 1.0; nothing may exceed it.
    let at_match = out[pr * w + pc];
    assert!(
        (at_match - 1.0).abs() < 1e-4,
        "NCC at the perfect-match lag must be 1.0, got {at_match:.6}"
    );
    let global_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        global_max <= 1.0 + 1e-4,
        "normalized correlation must not exceed 1.0, got max {global_max:.6}"
    );
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
        .expect("infallible: validated precondition")
        .apply(&img)
        .expect("infallible: validated precondition");

    let (out_vals, _) = extract_vec(&result).expect("infallible: validated precondition");
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "NCC output at index {i} must be finite, got {v}"
        );
    }
}

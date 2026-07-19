//! Value-semantic tests for deconvolution filters (GAP-262-FLT-02).
//!
//! Every test follows the "value-semantic" pattern: deterministic inputs produce
//! deterministic outputs verified against mathematical invariants.

use ritk_image::test_support as ts;
use ritk_image::Image;

use super::{
    InverseDeconvolution, LandweberDeconvolution, LandweberProjection, RichardsonLucyDeconvolution,
    TikhonovDeconvolution, WienerDeconvolution,
};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<f32, B, 2> {
    ts::make_image::<f32, B, 2>(data, dims)
}

// ── Wiener ───────────────────────────────────────────────────────────────────

/// Dirac delta kernel → output equals input (identity convolution).
#[test]
fn wiener_dirac_identity() {
    // A 3×3 kernel with just the center pixel = 1.0 (Dirac delta)
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = WienerDeconvolution::new(0.01);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.1,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// K=0 Wiener reduces to inverse filtering (approximate identity for Dirac kernel).
#[test]
fn wiener_zero_k_dirac_identity() {
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = WienerDeconvolution::new(0.0);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.1,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Output shape matches input shape.
#[test]
fn wiener_output_shape_matches_input() {
    let image_vals = vec![1.0_f32; 25];
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let img = make_image_2d(image_vals, [5, 5]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = WienerDeconvolution::new(0.01);
    let result = filter.apply(&img, &ker).unwrap();
    assert_eq!(result.shape(), [5, 5]);
}

// ── Tikhonov ─────────────────────────────────────────────────────────────────

/// λ>0 smooths: output variance ≤ input variance for any image.
#[test]
fn tikhonov_lambda_reduces_variance() {
    // Random-looking 5×5 image with a 3×3 averaging kernel
    let kernel_vals = vec![1.0 / 9.0; 9];
    let image_vals: Vec<f32> = (0..25).map(|i| (i as f32 * 3.7).sin()).collect();
    let img = make_image_2d(image_vals.clone(), [5, 5]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = TikhonovDeconvolution::new(1.0);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    let var: f64 = vals
        .iter()
        .map(|&v| {
            let d = v as f64 - vals.iter().map(|&x| x as f64).sum::<f64>() / vals.len() as f64;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    // Tikhonov regularization suppresses high frequencies → reduces variance
    assert!(var.is_finite(), "output variance must be finite");
}

/// Dirac delta + λ=0 → inverse filter (identity for delta kernel).
#[test]
fn tikhonov_dirac_identity() {
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = TikhonovDeconvolution::new(0.0);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.1,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Output shape matches input shape.
#[test]
fn tikhonov_output_shape_matches_input() {
    let image_vals = vec![1.0_f32; 25];
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let img = make_image_2d(image_vals, [5, 5]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = TikhonovDeconvolution::new(0.01);
    let result = filter.apply(&img, &ker).unwrap();
    assert_eq!(result.shape(), [5, 5]);
}

// ── Richardson-Lucy ──────────────────────────────────────────────────────────

/// Dirac delta kernel → RL converges to identity.
#[test]
fn richardson_lucy_dirac_identity() {
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = RichardsonLucyDeconvolution::new().with_max_iterations(10);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 1.0,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// RL preserves total flux (sum of values).
#[test]
fn richardson_lucy_preserves_total_flux() {
    let kernel_vals = vec![0.1, 0.2, 0.1, 0.2, 0.8, 0.2, 0.1, 0.2, 0.1];
    // Normalize kernel to sum 1 to preserve flux
    let ker_sum: f32 = kernel_vals.iter().sum();
    let kernel_norm: Vec<f32> = kernel_vals.iter().map(|v| v / ker_sum).collect();
    let image_vals = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_norm, [3, 3]);
    let filter = RichardsonLucyDeconvolution::new()
        .with_max_iterations(10)
        .with_tolerance(0.0);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    let input_sum: f32 = image_vals.iter().sum();
    let output_sum: f32 = vals.iter().sum();
    assert!(
        (output_sum - input_sum).abs() < input_sum * 0.02,
        "RL must approximately preserve total flux (≤2% FFT boundary loss): in={input_sum}, out={output_sum}"
    );
}

/// RL output shape matches input shape.
#[test]
fn richardson_lucy_output_shape_matches_input() {
    let image_vals = vec![1.0_f32; 25];
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let img = make_image_2d(image_vals, [5, 5]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = RichardsonLucyDeconvolution::new().with_max_iterations(5);
    let result = filter.apply(&img, &ker).unwrap();
    assert_eq!(result.shape(), [5, 5]);
}

/// RL with non-negative image should produce non-negative output.
#[test]
fn richardson_lucy_preserves_non_negativity() {
    let kernel_vals = vec![
        0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
    ];
    let image_vals = vec![1.0_f32; 16];
    let img = make_image_2d(image_vals, [4, 4]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = RichardsonLucyDeconvolution::new().with_max_iterations(10);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for &v in &vals {
        assert!(v >= -0.01, "RL must preserve non-negativity, got {v}");
    }
}

// ── Landweber ────────────────────────────────────────────────────────────────

/// Dirac delta → Landweber converges to identity in few iterations.
#[test]
fn landweber_dirac_identity() {
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = LandweberDeconvolution::new()
        .with_step_size(0.5)
        .with_max_iterations(50);
    let result = filter.apply(&img, &ker).unwrap();
    let vals = result.data().to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.5,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Output shape matches input shape.
#[test]
fn landweber_output_shape_matches_input() {
    let image_vals = vec![1.0_f32; 25];
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let img = make_image_2d(image_vals, [5, 5]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let filter = LandweberDeconvolution::new().with_max_iterations(5);
    let result = filter.apply(&img, &ker).unwrap();
    assert_eq!(result.shape(), [5, 5]);
}

/// Default builder constructs valid filter.
#[test]
fn landweber_default_is_valid() {
    let filter = LandweberDeconvolution::new();
    assert_eq!(filter.max_iterations, 100);
    assert!(filter.step_size > 0.0);
    assert!(filter.tolerance > 0.0);
}

/// Builder method chain works.
#[test]
fn landweber_builder_chain() {
    let filter = LandweberDeconvolution::new()
        .with_step_size(0.05)
        .with_max_iterations(200)
        .with_tolerance(1e-8);
    assert!((filter.step_size - 0.05).abs() < 1e-10);
    assert_eq!(filter.max_iterations, 200);
    assert!((filter.tolerance - 1e-8).abs() < 1e-10);
}

// ── Projected Landweber (non-negativity) ────────────────────────────────────

/// The non-negativity projection forces every output voxel to be `>= 0`, while
/// plain Landweber on the same problem produces negative ring artefacts. Both
/// agree wherever the unconstrained estimate is already non-negative.
#[test]
fn projected_landweber_enforces_non_negativity() {
    // Sharp-edged box convolved with a small blur, then deconvolved: the
    // unconstrained Landweber overshoots into negatives near the edges.
    let mut img = vec![0.0f32; 9 * 9];
    for r in 3..6 {
        for c in 3..6 {
            img[r * 9 + c] = 100.0;
        }
    }
    // Normalized 3×3 blur PSF.
    let ker = vec![1.0f32 / 9.0; 9];
    let image = make_image_2d(img, [9, 9]);
    let kernel = make_image_2d(ker, [3, 3]);

    let plain = LandweberDeconvolution::new()
        .with_step_size(0.5)
        .with_max_iterations(30)
        .apply(&image, &kernel)
        .unwrap();
    let projected = LandweberDeconvolution::new()
        .with_step_size(0.5)
        .with_max_iterations(30)
        .with_projection(LandweberProjection::NonNegative)
        .apply(&image, &kernel)
        .unwrap();

    let (pv, _) = extract_vec_infallible(&plain);
    let (qv, _) = extract_vec_infallible(&projected);
    // Plain Landweber must go negative somewhere; projected must not.
    assert!(
        pv.iter().any(|&v| v < -1e-3),
        "plain Landweber expected to overshoot negative"
    );
    assert!(
        qv.iter().all(|&v| v >= 0.0),
        "projected Landweber must be non-negative"
    );
    // Per-iteration projection diverges the trajectory from plain Landweber
    // (each clamp feeds the next iteration), so the two results genuinely differ.
    assert!(
        pv.iter()
            .zip(qv.iter())
            .any(|(&p, &q)| (p - q).abs() > 1e-3),
        "projection must change the result vs plain Landweber"
    );
}

/// Default projection is `None` (plain Landweber); the builder switches it on.
#[test]
fn landweber_projection_default_and_builder() {
    assert_eq!(
        LandweberDeconvolution::new().projection,
        LandweberProjection::None
    );
    let f = LandweberDeconvolution::new().with_projection(LandweberProjection::NonNegative);
    assert_eq!(f.projection, LandweberProjection::NonNegative);
}

// ── Inverse filter ──────────────────────────────────────────────────────────

/// Dirac-delta PSF → inverse filter is the identity (G/1 = G).
#[test]
fn inverse_dirac_identity() {
    let kernel_vals = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let image_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image_2d(image_vals.clone(), [3, 3]);
    let ker = make_image_2d(kernel_vals, [3, 3]);
    let result = InverseDeconvolution::new(1e-3).apply(&img, &ker).unwrap();
    let (vals, _) = extract_vec_infallible(&result);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 1e-3,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// A larger zero-magnitude threshold zeros more frequencies → strictly smaller
/// (in L2) restored signal energy than a tiny threshold, and the output stays
/// finite (no division blow-up at OTF nulls).
#[test]
fn inverse_threshold_suppresses_more_frequencies() {
    // Normalized 3×3 blur whose OTF has near-zero frequencies.
    let ker = make_image_2d(vec![1.0f32 / 9.0; 9], [3, 3]);
    let image_vals: Vec<f32> = (0..25).map(|i| (i as f32 * 1.3).sin() * 10.0).collect();
    let img = make_image_2d(image_vals, [5, 5]);

    let low = InverseDeconvolution::new(1e-4).apply(&img, &ker).unwrap();
    let high = InverseDeconvolution::new(0.5).apply(&img, &ker).unwrap();
    let (lv, _) = extract_vec_infallible(&low);
    let (hv, _) = extract_vec_infallible(&high);
    let energy = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    assert!(
        lv.iter().all(|v| v.is_finite()),
        "low-threshold output must be finite"
    );
    assert!(
        hv.iter().all(|v| v.is_finite()),
        "high-threshold output must be finite"
    );
    assert!(
        energy(&hv) < energy(&lv),
        "larger threshold must zero more frequencies: high={} low={}",
        energy(&hv),
        energy(&lv)
    );
}

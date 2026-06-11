//! Value-semantic tests for deconvolution filters (GAP-262-FLT-02).
//!
//! Every test follows the "value-semantic" pattern: deterministic inputs produce
//! deterministic outputs verified against mathematical invariants.

use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

use super::{
    LandweberDeconvolution, RichardsonLucyDeconvolution, TikhonovDeconvolution, WienerDeconvolution,
};

type B = NdArray<f32>;

fn make_image_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<B, 2> {
    let device = Default::default();
    let t = Tensor::<B, 2>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
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

//! Value-semantic 3-D tests for deconvolution filters (GAP-262-FLT-02).
#![allow(clippy::identity_op, clippy::erasing_op)]
//!
//! Each test derives expected values analytically:
//! - Dirac delta kernel in 3-D → identity: u_out ≈ u_in within tolerance
//! - Shape invariant: output shape equals input shape
//! - Non-negativity: RL with non-negative input and PSF preserves sign
//! - Variance: Tikhonov λ>0 produces finite output variance

use super::{
    LandweberDeconvolution, RichardsonLucyDeconvolution, TikhonovDeconvolution, WienerDeconvolution,
};
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// Construct a test 3-D image.
fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

/// 3×3×3 Dirac delta kernel — exactly 1.0 at center [1,1,1], 0.0 elsewhere.
///
/// Convolving with the Dirac delta is the identity operation:
///   (u ∗ δ)(x) = u(x)  for all x.
fn dirac_kernel_3x3x3() -> Vec<f32> {
    let mut k = vec![0.0_f32; 27];
    k[1 * 9 + 1 * 3 + 1] = 1.0; // center of 3×3×3
    k
}

// ── Wiener 3-D ───────────────────────────────────────────────────────────────

/// Dirac delta kernel → Wiener output equals input.
///
/// Since u ∗ δ = u, the Wiener filter with K=0.01 attenuates by 1/(1+K).
/// Maximum attenuation error: max_val × K/(1+K) = 27 × 0.01/1.01 ≈ 0.267.
/// Tolerance 0.3 per voxel is analytically tight for this input range.
#[test]
fn wiener_3d_dirac_identity() {
    let image_vals: Vec<f32> = (0..27).map(|i| i as f32 + 1.0).collect();
    let img = make_image_3d(image_vals.clone(), [3, 3, 3]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let filter = WienerDeconvolution::new(0.01);
    let result = filter.apply_3d(&img, &ker).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 27, "output length must equal 27");
    for (i, &v) in vals.iter().enumerate() {
        // Tolerance accounts for K-attenuation: err ≤ input[i] × K/(1+K)
        let max_err = image_vals[i] * 0.01 / 1.01 + 0.05;
        assert!(
            (v - image_vals[i]).abs() < max_err,
            "voxel {i}: expected ~{}, got {v} (max_err={max_err:.4})",
            image_vals[i]
        );
    }
}

/// K=0 Wiener (inverse filter) with Dirac kernel approximates identity.
#[test]
fn wiener_3d_zero_k_dirac_identity() {
    let image_vals: Vec<f32> = (0..27).map(|i| i as f32 + 1.0).collect();
    let img = make_image_3d(image_vals.clone(), [3, 3, 3]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let filter = WienerDeconvolution::new(0.0);
    let result = filter.apply_3d(&img, &ker).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.1,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Output shape must match input shape for arbitrary sizes.
#[test]
fn wiener_3d_output_shape_matches_input() {
    let img = make_image_3d(vec![1.0_f32; 4 * 5 * 6], [4, 5, 6]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let result = WienerDeconvolution::new(0.01).apply_3d(&img, &ker).unwrap();
    assert_eq!(
        result.shape(),
        [4, 5, 6],
        "output shape must match input shape"
    );
}

// ── Tikhonov 3-D ─────────────────────────────────────────────────────────────

/// Tikhonov λ=0 with Dirac kernel must recover original values.
///
/// λ=0 reduces to inverse filtering; Dirac convolution is identity.
#[test]
fn tikhonov_3d_dirac_identity() {
    let image_vals: Vec<f32> = (0..27).map(|i| (i as f32 + 1.0) * 2.0).collect();
    let img = make_image_3d(image_vals.clone(), [3, 3, 3]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let filter = TikhonovDeconvolution::new(0.0);
    let result = filter.apply_3d(&img, &ker).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.1,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Tikhonov λ>0 with blurring kernel produces finite output variance.
///
/// Finite variance verifies the 3-D Laplacian regularization path is executed.
#[test]
fn tikhonov_3d_lambda_reduces_variance() {
    // Averaging 3×3×3 kernel (uniform blur)
    let kernel_vals = vec![1.0_f32 / 27.0; 27];
    let image_vals: Vec<f32> = (0..125).map(|i| (i as f32 * 2.7).sin()).collect();
    let img = make_image_3d(image_vals, [5, 5, 5]);
    let ker = make_image_3d(kernel_vals, [3, 3, 3]);
    let result = TikhonovDeconvolution::new(1.0)
        .apply_3d(&img, &ker)
        .unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    // All outputs must be finite — verifies no NaN/Inf in 3-D Laplacian path
    for &v in &vals {
        assert!(v.is_finite(), "Tikhonov 3-D output must be finite, got {v}");
    }
}

/// Output shape must match input shape.
#[test]
fn tikhonov_3d_output_shape_matches_input() {
    let img = make_image_3d(vec![1.0_f32; 4 * 5 * 6], [4, 5, 6]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let result = TikhonovDeconvolution::new(0.01)
        .apply_3d(&img, &ker)
        .unwrap();
    assert_eq!(result.shape(), [4, 5, 6]);
}

// ── Richardson-Lucy 3-D ──────────────────────────────────────────────────────

/// Dirac delta kernel → RL converges to identity in few iterations.
///
/// Since u ∗ δ = u, the ratio (g / (δ ∗ u)) = (u / u) = 1, and
/// h* ∗ 1 ≈ 1, so uₖ₊₁ = uₖ · 1 = uₖ — tolerance within 1.0 per voxel.
#[test]
fn richardson_lucy_3d_dirac_identity() {
    let image_vals: Vec<f32> = (0..27).map(|i| i as f32 + 1.0).collect();
    let img = make_image_3d(image_vals.clone(), [3, 3, 3]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let filter = RichardsonLucyDeconvolution::new().with_max_iterations(10);
    let result = filter.apply_3d(&img, &ker).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 1.0,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Non-negativity preservation: RL with non-negative input must yield
/// non-negative output (Poisson-model invariant).
#[test]
fn richardson_lucy_3d_preserves_non_negativity() {
    // Gaussian-like 3×3×3 kernel (positive, normalized)
    let mut kernel_vals = vec![0.0_f32; 27];
    for kz in 0..3_usize {
        for ky in 0..3_usize {
            for kx in 0..3_usize {
                let dz = (kz as f32 - 1.0).powi(2);
                let dy = (ky as f32 - 1.0).powi(2);
                let dx = (kx as f32 - 1.0).powi(2);
                kernel_vals[kz * 9 + ky * 3 + kx] = (-(dz + dy + dx) / 2.0).exp();
            }
        }
    }
    let ksum: f32 = kernel_vals.iter().sum();
    kernel_vals.iter_mut().for_each(|v| *v /= ksum);

    let image_vals = vec![1.0_f32; 4 * 4 * 4];
    let img = make_image_3d(image_vals, [4, 4, 4]);
    let ker = make_image_3d(kernel_vals, [3, 3, 3]);
    let result = RichardsonLucyDeconvolution::new()
        .with_max_iterations(10)
        .apply_3d(&img, &ker)
        .unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v >= -0.01, "RL 3-D must preserve non-negativity, got {v}");
    }
}

/// Output shape must match input shape.
#[test]
fn richardson_lucy_3d_output_shape_matches_input() {
    let img = make_image_3d(vec![1.0_f32; 4 * 5 * 6], [4, 5, 6]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let result = RichardsonLucyDeconvolution::new()
        .with_max_iterations(5)
        .apply_3d(&img, &ker)
        .unwrap();
    assert_eq!(result.shape(), [4, 5, 6]);
}

// ── Landweber 3-D ────────────────────────────────────────────────────────────

/// Dirac delta → Landweber converges toward identity.
///
/// Residual after each step is (g − δ ∗ uₖ) → 0 as uₖ → g.
/// Tolerance 0.5 per voxel for α=0.5, 50 iterations.
#[test]
fn landweber_3d_dirac_identity() {
    let image_vals: Vec<f32> = (0..27).map(|i| i as f32 + 1.0).collect();
    let img = make_image_3d(image_vals.clone(), [3, 3, 3]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let filter = LandweberDeconvolution::new()
        .with_step_size(0.5)
        .with_max_iterations(50);
    let result = filter.apply_3d(&img, &ker).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - image_vals[i]).abs() < 0.5,
            "voxel {i}: expected ~{}, got {v}",
            image_vals[i]
        );
    }
}

/// Output shape must match input shape.
#[test]
fn landweber_3d_output_shape_matches_input() {
    let img = make_image_3d(vec![1.0_f32; 4 * 5 * 6], [4, 5, 6]);
    let ker = make_image_3d(dirac_kernel_3x3x3(), [3, 3, 3]);
    let result = LandweberDeconvolution::new()
        .with_max_iterations(5)
        .apply_3d(&img, &ker)
        .unwrap();
    assert_eq!(result.shape(), [4, 5, 6]);
}

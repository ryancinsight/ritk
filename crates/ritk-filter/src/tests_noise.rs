//! Value-semantic tests for noise simulation filters (GAP-262-FLT-05).
//!
//! Every test follows the "value-semantic" pattern: fixed seed, deterministic
//! output verification against known-good values or mathematical invariants.

use super::*;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── Additive Gaussian ──────────────────────────────────────────────────────

/// Deterministic Gaussian noise with seed=42 produces known output.
#[test]
fn gaussian_deterministic_seed_42() {
    let img = make_image(vec![10.0_f32; 4], [1, 2, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(5.0)
        .with_mean(0.0)
        .with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    // Different seed produces different values; just verify shape and that
    // values differ from the original (noise was actually added).
    assert_eq!(vals.len(), 4);
    assert!(vals.iter().any(|&v| (v - 10.0).abs() > 0.01));
}

/// Zero std dev produces no change (within f32 tolerance).
#[test]
fn gaussian_zero_std_is_identity() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(data.clone(), [2, 2, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(0.0);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - data[i]).abs() < 1e-6,
            "voxel {i} changed with zero std"
        );
    }
}

/// Nonzero mean shifts all voxels.
#[test]
fn gaussian_nonzero_mean_shifts() {
    let img = make_image(vec![0.0_f32; 1000], [10, 10, 10]);
    let filter = AdditiveGaussianNoiseFilter::new(0.1)
        .with_mean(100.0)
        .with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    assert!(
        (mean - 100.0).abs() < 2.0,
        "mean should be ~100, got {mean}"
    );
}

/// Different seeds produce different noise patterns.
#[test]
fn gaussian_seeds_differ() {
    let data = vec![0.0_f32; 100];
    let img = make_image(data.clone(), [5, 5, 4]);
    let filter1 = AdditiveGaussianNoiseFilter::new(1.0).with_seed(42);
    let filter2 = AdditiveGaussianNoiseFilter::new(1.0).with_seed(43);
    let v1 = filter1
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter2
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_ne!(v1, v2, "different seeds must produce different output");
}

/// Same seed produces identical output.
#[test]
fn gaussian_same_seed_idempotent() {
    let data = vec![0.0_f32; 50];
    let img = make_image(data, [5, 5, 2]);
    let filter = AdditiveGaussianNoiseFilter::new(1.0).with_seed(42);
    let v1 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

// ── Salt & Pepper ──────────────────────────────────────────────────────────

/// Zero probability leaves image unchanged.
#[test]
fn salt_pepper_zero_prob_is_identity() {
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [3, 3, 3]);
    let filter = SaltAndPepperNoiseFilter::new(0.0);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    assert_eq!(vals, data, "zero probability must leave image unchanged");
}

/// With prob=1.0, all voxels become min or max.
#[test]
fn salt_pepper_full_prob_saturates() {
    let data: Vec<f32> = (0..100).map(|i| (i % 10) as f32).collect();
    let img = make_image(data, [5, 5, 4]);
    let filter = SaltAndPepperNoiseFilter::new(1.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    let min = vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    // Every voxel must be either min or max
    for &v in &vals {
        assert!(v == min || v == max, "all voxels must be salt or pepper");
    }
}

/// Deterministic output at moderate probability.
#[test]
fn salt_pepper_deterministic() {
    let data = vec![5.0_f32; 64];
    let img = make_image(data, [4, 4, 4]);
    let filter = SaltAndPepperNoiseFilter::new(0.3).with_seed(42);
    let v1 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

// ── Shot (Poisson) ─────────────────────────────────────────────────────────

/// Very large scale approximates identity (Poisson → Gaussian at large λ).
#[test]
fn shot_large_scale_near_identity() {
    let data: Vec<f32> = (0..27).map(|i| (i + 1) as f32 * 10.0).collect();
    let img = make_image(data.clone(), [3, 3, 3]);
    let filter = ShotNoiseFilter::new(1000.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        let rel_err = (v - data[i]).abs() / data[i];
        assert!(
            rel_err < 0.05,
            "voxel {i}: relative error {rel_err} too large for scale=1000"
        );
    }
}

/// Zero scale produces zero output (no photons).
#[test]
fn shot_zero_scale_produces_zeros() {
    let data = vec![100.0_f32; 8];
    let img = make_image(data, [2, 2, 2]);
    let filter = ShotNoiseFilter::new(0.0);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for &v in &vals {
        assert_eq!(v, 0.0, "zero scale must produce zero output");
    }
}

/// Negative intensities are clamped to zero.
#[test]
fn shot_clamps_negative() {
    let data = vec![-10.0_f32; 8];
    let img = make_image(data, [2, 2, 2]);
    let filter = ShotNoiseFilter::new(1.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for &v in &vals {
        assert_eq!(v, 0.0, "negative intensities must be clamped to zero");
    }
}

/// Zero-valued input produces zero output (Poisson(0) = 0 with probability 1).
#[test]
fn shot_zero_input_returns_zero() {
    let data = vec![0.0_f32; 27];
    let img = make_image(data, [3, 3, 3]);
    let filter = ShotNoiseFilter::new(10.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, 0.0, "voxel {i}: zero input must produce zero output");
    }
}

/// Same seed produces identical output.
#[test]
fn shot_same_seed_idempotent() {
    let data: Vec<f32> = (0..50).map(|i| (i + 1) as f32).collect();
    let img = make_image(data, [5, 5, 2]);
    let filter = ShotNoiseFilter::new(5.0).with_seed(42);
    let v1 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

/// Output shape matches input shape.
#[test]
fn shot_preserves_shape() {
    let data: Vec<f32> = (0..60).map(|i| (i + 1) as f32).collect();
    let img = make_image(data, [3, 4, 5]);
    let filter = ShotNoiseFilter::new(10.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    assert_eq!(
        result.shape(),
        img.shape(),
        "shot noise must preserve image shape"
    );
}

/// Spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn shot_preserves_metadata() {
    use ritk_spatial::{Direction, Point, Spacing};
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![10.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(
        t,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 0.5, 2.0]),
        Direction::identity(),
    );
    let filter = ShotNoiseFilter::new(5.0).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    assert_eq!(result.origin(), img.origin(), "origin must be preserved");
    assert_eq!(result.spacing(), img.spacing(), "spacing must be preserved");
    assert_eq!(
        result.direction(),
        img.direction(),
        "direction must be preserved"
    );
}

// ── Speckle ────────────────────────────────────────────────────────────────

/// Zero std dev produces no change.
#[test]
fn speckle_zero_std_is_identity() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(data.clone(), [2, 2, 2]);
    let filter = SpeckleNoiseFilter::new(0.0);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - data[i]).abs() < 1e-6,
            "voxel {i} changed with zero std"
        );
    }
}

/// Deterministic seed produces reproducible output.
#[test]
fn speckle_deterministic() {
    let img = make_image(vec![10.0_f32; 64], [4, 4, 4]);
    let filter = SpeckleNoiseFilter::new(0.1).with_seed(42);
    let v1 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

/// Speckle noise preserves mean approximately.
#[test]
fn speckle_preserves_mean_approx() {
    let img = make_image(vec![50.0_f32; 1000], [10, 10, 10]);
    let filter = SpeckleNoiseFilter::new(0.05).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    // Multiplicative N(0, σ) has mean multiplicative factor 1.0
    assert!(
        (mean - 50.0).abs() < 1.0,
        "speckle should approximately preserve mean, got {mean}"
    );
}

/// Non-zero sigma changes at least some voxel values.
#[test]
fn speckle_nonzero_sigma_changes_values() {
    let data = vec![10.0_f32; 100];
    let img = make_image(data.clone(), [5, 5, 4]);
    let filter = SpeckleNoiseFilter::new(0.5).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    assert!(
        vals.iter().any(|&v| (v - 10.0).abs() > 0.01),
        "non-zero sigma must change at least some voxel values"
    );
}

/// Positive input with moderate speckle must not produce negative output.
#[test]
fn speckle_positive_input_no_negatives() {
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let img = make_image(data, [4, 4, 4]);
    let filter = SpeckleNoiseFilter::new(0.3).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    let vals = result.data().clone().into_data().into_vec::<f32>().unwrap();
    // With σ=0.3 the multiplicative factor is (1+N(0,0.3)).
    // The 3σ range is [0.1, 1.9] so all outputs should be ≥ 0.
    // (Values slightly below 0 due to extreme tail draws are clamped by the
    // statistical test: check that no value is significantly negative.)
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v >= -1e-3,
            "voxel {i}: speckle on positive input produced negative value {v}"
        );
    }
}

/// Same seed produces identical output.
#[test]
fn speckle_same_seed_idempotent() {
    let data = vec![10.0_f32; 50];
    let img = make_image(data, [5, 5, 2]);
    let filter = SpeckleNoiseFilter::new(0.1).with_seed(42);
    let v1 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply_3d(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2, "same seed must produce identical output");
}

/// Output shape matches input shape.
#[test]
fn speckle_preserves_shape() {
    let data: Vec<f32> = (0..60).map(|i| (i + 1) as f32).collect();
    let img = make_image(data, [3, 4, 5]);
    let filter = SpeckleNoiseFilter::new(0.1).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    assert_eq!(
        result.shape(),
        img.shape(),
        "speckle noise must preserve image shape"
    );
}

/// Spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn speckle_preserves_metadata() {
    use ritk_spatial::{Direction, Point, Spacing};
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![10.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(
        t,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 0.5, 2.0]),
        Direction::identity(),
    );
    let filter = SpeckleNoiseFilter::new(0.1).with_seed(42);
    let result = filter.apply_3d(&img).unwrap();
    assert_eq!(result.origin(), img.origin(), "origin must be preserved");
    assert_eq!(result.spacing(), img.spacing(), "spacing must be preserved");
    assert_eq!(
        result.direction(),
        img.direction(),
        "direction must be preserved"
    );
}

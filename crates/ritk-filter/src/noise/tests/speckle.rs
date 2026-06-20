use super::*;

/// Zero std dev produces no change.
#[test]
fn speckle_zero_std_is_identity() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(data.clone(), [2, 2, 2]);
    let filter = SpeckleNoiseFilter::new(0.0);
    let result = filter.apply(&img).unwrap();
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
        .apply(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply(&img)
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
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
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
        .apply(&img)
        .unwrap()
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let v2 = filter
        .apply(&img)
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
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
    assert_eq!(result.origin(), img.origin(), "origin must be preserved");
    assert_eq!(result.spacing(), img.spacing(), "spacing must be preserved");
    assert_eq!(
        result.direction(),
        img.direction(),
        "direction must be preserved"
    );
}

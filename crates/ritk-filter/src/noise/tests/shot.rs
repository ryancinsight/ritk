use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Very large scale approximates identity (Poisson → Gaussian at large λ).
#[test]
fn shot_large_scale_near_identity() {
    let data: Vec<f32> = (0..27).map(|i| (i + 1) as f32 * 10.0).collect();
    let img = make_image(data.clone(), [3, 3, 3]);
    let filter = ShotNoiseFilter::new(1000.0).with_seed(42);
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
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
fn shot_preserves_shape() {
    let data: Vec<f32> = (0..60).map(|i| (i + 1) as f32).collect();
    let img = make_image(data, [3, 4, 5]);
    let filter = ShotNoiseFilter::new(10.0).with_seed(42);
    let result = filter.apply(&img).unwrap();
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
    let result = filter.apply(&img).unwrap();
    assert_eq!(result.origin(), img.origin(), "origin must be preserved");
    assert_eq!(result.spacing(), img.spacing(), "spacing must be preserved");
    assert_eq!(
        result.direction(),
        img.direction(),
        "direction must be preserved"
    );
}

#[test]
fn native_shot_matches_tensor_sequence_across_sampling_regimes() {
    let values = vec![1.0, 10.0]; // scale=10 selects Poisson then normal approximation.
    let image = NativeImage::from_flat_on(
        values.clone(),
        [1, 1, 2],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let filter = ShotNoiseFilter::new(10.0).with_seed(42);
    let native = filter
        .apply_native(&image, &SequentialBackend)
        .expect("native shot noise succeeds");
    let tensor = filter.apply(&make_image(values, [1, 1, 2])).unwrap();
    assert_eq!(
        native.data_slice().expect("contiguous native output"),
        tensor.data_slice().as_ref()
    );
    assert_eq!(native.origin(), image.origin());
    assert_eq!(native.spacing(), image.spacing());
    assert_eq!(native.direction(), image.direction());
}

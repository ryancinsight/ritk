use super::*;
use crate::native_support::{make_native_image, make_native_image_with_metadata, native_vals};
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};

#[test]
fn test_at_alpha_gives_midpoint() {
    // At I(x) = alpha, sigmoid(0) = 0.5, so output = range*0.5 + min = 0.5
    let img = make_native_image(vec![2.0_f32], [1, 1, 1]); // alpha = 2.0
    let f = SigmoidImageFilter::new(2.0, 1.0, 0.0, 1.0);
    let result = native_vals(&f.apply_native(&img, &SequentialBackend).expect("apply_native should succeed"));
    let expected = 0.5_f32;
    assert!(
        (result[0] - expected).abs() < 1e-5,
        "at alpha -> 0.5, got {}",
        result[0]
    );
}

#[test]
fn test_monotone_increasing_with_positive_beta() {
    let vals = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
    let img = make_native_image(vals, [1, 1, 5]);
    let f = SigmoidImageFilter::new(2.0, 1.0, 0.0, 1.0);
    let result = native_vals(&f.apply_native(&img, &SequentialBackend).expect("apply_native should succeed"));
    for i in 0..result.len() - 1 {
        assert!(
            result[i] < result[i + 1],
            "sigmoid must be monotone increasing, positions {} and {}",
            i,
            i + 1
        );
    }
}

#[test]
fn test_output_range_bounded() {
    let vals: Vec<f32> = (-50i32..=50).map(|i| i as f32).collect();
    let img = make_native_image(vals, [1, 1, 101]);
    let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
    let result = native_vals(&f.apply_native(&img, &SequentialBackend).expect("apply_native should succeed"));
    for &v in &result {
        // In f32, exp(-50) < f32::EPSILON, so 1.0 + exp(-50) == 1.0 exactly.
        // The sigmoid is bounded in [0, 1] in f32; strict-open bound requires wider domain.
        assert!(
            (0.0..=1.0).contains(&v),
            "sigmoid output must be in [0, 1], got {}",
            v
        );
    }
}

#[test]
fn test_large_positive_input_approaches_max() {
    let img = make_native_image(vec![1e6_f32], [1, 1, 1]);
    let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
    let result = native_vals(&f.apply_native(&img, &SequentialBackend).expect("apply_native should succeed"));
    assert!(
        result[0] > 0.9999,
        "large positive input should approach max_output=1.0, got {}",
        result[0]
    );
}

#[test]
fn test_large_negative_input_approaches_min() {
    let img = make_native_image(vec![-1e6_f32], [1, 1, 1]);
    let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
    let result = native_vals(&f.apply_native(&img, &SequentialBackend).expect("apply_native should succeed"));
    assert!(
        result[0] < 0.0001,
        "large negative input should approach min_output=0.0, got {}",
        result[0]
    );
}

#[test]
fn native_sigmoid_preserves_values_and_metadata() {
    let image = make_native_image_with_metadata(
        vec![0.0, 2.0, 4.0],
        [1, 1, 3],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
    );
    let output = SigmoidImageFilter::new(2.0, 1.0, 0.0, 1.0)
        .apply_native(&image, &SequentialBackend)
        .expect("native sigmoid succeeds");
    let values = output.data_slice().expect("invariant: contiguous storage");
    assert!(values[0] < 0.12 && values[2] > 0.88);
    assert_eq!(values[1], 0.5);
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}

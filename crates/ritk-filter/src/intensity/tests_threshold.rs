use super::*;
use crate::native_support::{make_native_image, make_native_image_with_metadata, native_vals};
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};

#[test]
fn test_below_zeros_low_values() {
    // values 0..9, threshold=5 -> pixels 0,1,2,3,4 become 0.0; 5..9 unchanged
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_native_image(vals, [1, 1, 10]);
    let f = ThresholdImageFilter::below(5.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for i in 0..5usize {
        assert_eq!(result[i], 0.0, "pixel {} (value {}) should be zeroed", i, i);
    }
    for i in 5..10usize {
        assert!(
            (result[i] - i as f32).abs() < 1e-6,
            "pixel {} should be unchanged",
            i
        );
    }
}

#[test]
fn test_above_zeros_high_values() {
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_native_image(vals, [1, 1, 10]);
    let f = ThresholdImageFilter::above(5.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for i in 0..=5usize {
        assert!(
            (result[i] - i as f32).abs() < 1e-6,
            "pixel {} should be unchanged",
            i
        );
    }
    for i in 6..10usize {
        assert_eq!(result[i], 0.0, "pixel {} should be zeroed", i);
    }
}

#[test]
fn test_outside_keeps_interior() {
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_native_image(vals, [1, 1, 10]);
    let f = ThresholdImageFilter::outside(3.0, 6.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for i in 0..3usize {
        assert_eq!(result[i], 0.0, "pixel {} outside [3,6] should be zeroed", i);
    }
    for i in 3..=6usize {
        assert!(
            (result[i] - i as f32).abs() < 1e-6,
            "pixel {} inside [3,6] should be unchanged",
            i
        );
    }
    for i in 7..10usize {
        assert_eq!(result[i], 0.0, "pixel {} outside [3,6] should be zeroed", i);
    }
}

#[test]
fn test_below_no_change_if_all_above() {
    let vals = vec![10.0_f32, 20.0, 30.0];
    let img = make_native_image(vals.clone(), [1, 1, 3]);
    let f = ThresholdImageFilter::below(5.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for (a, b) in vals.iter().zip(result.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "all above threshold: no change expected"
        );
    }
}

#[test]
fn test_above_no_change_if_all_below() {
    let vals = vec![1.0_f32, 2.0, 3.0];
    let img = make_native_image(vals.clone(), [1, 1, 3]);
    let f = ThresholdImageFilter::above(10.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for (a, b) in vals.iter().zip(result.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "all below threshold: no change expected"
        );
    }
}

#[test]
fn test_outside_uniform_inside() {
    let vals = vec![5.0_f32, 5.5, 6.0, 6.5, 7.0];
    let img = make_native_image(vals.clone(), [1, 1, 5]);
    let f = ThresholdImageFilter::outside(5.0, 7.0, 0.0);
    let result = native_vals(
        &f.apply_native(&img, &SequentialBackend)
            .expect("apply_native should succeed"),
    );
    for (a, b) in vals.iter().zip(result.iter()) {
        assert!((a - b).abs() < 1e-6, "all inside [5,7]: unchanged");
    }
}

#[test]
fn native_outside_threshold_preserves_values_and_metadata() {
    let image = make_native_image_with_metadata(
        vec![1.0, 3.0, 5.0, 7.0],
        [1, 1, 4],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
    );
    let output = ThresholdImageFilter::outside(3.0, 5.0, 0.0)
        .apply_native(&image, &SequentialBackend)
        .expect("native threshold succeeds");

    assert_eq!(
        output.data_slice().expect("invariant: contiguous storage"),
        &[0.0, 3.0, 5.0, 0.0]
    );
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}

use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
type B = NdArray<f32>;

fn make_image(vals: Vec<f32>) -> Image<B, 3> {
    let n = vals.len();
    ts::make_image::<B, 3>(vals, [1, 1, n])
}

fn get_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

#[test]
fn test_below_zeros_low_values() {
    // values 0..9, threshold=5 -> pixels 0,1,2,3,4 become 0.0; 5..9 unchanged
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_image(vals);
    let f = ThresholdImageFilter::below(5.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
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
    let img = make_image(vals);
    let f = ThresholdImageFilter::above(5.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
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
    let img = make_image(vals);
    let f = ThresholdImageFilter::outside(3.0, 6.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
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
    let img = make_image(vals.clone());
    let f = ThresholdImageFilter::below(5.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
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
    let img = make_image(vals.clone());
    let f = ThresholdImageFilter::above(10.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
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
    let img = make_image(vals.clone());
    let f = ThresholdImageFilter::outside(5.0, 7.0, 0.0);
    let result = get_vals(&f.apply(&img).unwrap());
    for (a, b) in vals.iter().zip(result.iter()) {
        assert!((a - b).abs() < 1e-6, "all inside [5,7]: unchanged");
    }
}

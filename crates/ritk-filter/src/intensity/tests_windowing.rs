use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
type B = NdArray<f32>;

fn make_image(vals: Vec<f32>) -> Image<B, 3> {
    let n = vals.len();
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new([1, 1, n]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

fn get_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

#[test]
fn test_below_window_clamp_to_out_min() {
    // Values -10 are below window [0, 100] -> out_min = 0.0
    let img = make_image(vec![-10.0, -5.0, -1.0]);
    let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
    let result = get_vals(&f.apply(&img).unwrap());
    for &v in &result {
        assert!(
            (v - 0.0).abs() < 1e-6,
            "below window -> out_min=0.0, got {}",
            v
        );
    }
}

#[test]
fn test_above_window_clamp_to_out_max() {
    let img = make_image(vec![200.0, 300.0, 1000.0]);
    let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
    let result = get_vals(&f.apply(&img).unwrap());
    for &v in &result {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "above window -> out_max=1.0, got {}",
            v
        );
    }
}

#[test]
fn test_interior_linear_mapping() {
    // Value at midpoint of window -> midpoint of output
    let img = make_image(vec![50.0]); // midpoint of [0, 100]
    let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
    let result = get_vals(&f.apply(&img).unwrap());
    assert!(
        (result[0] - 0.5).abs() < 1e-5,
        "midpoint -> 0.5, got {}",
        result[0]
    );
}

#[test]
fn test_full_image_output_bounded() {
    let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let img = make_image(vals);
    let f = IntensityWindowingFilter::new(20.0, 80.0, 0.0, 255.0);
    let result = get_vals(&f.apply(&img).unwrap());
    let min_out = result.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_out = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (min_out - 0.0).abs() < 1e-4,
        "min output = 0.0, got {}",
        min_out
    );
    assert!(
        (max_out - 255.0).abs() < 1e-4,
        "max output = 255.0, got {}",
        max_out
    );
}

#[test]
fn test_equal_window_bounds_gives_out_min() {
    let img = make_image(vec![50.0, 100.0, 200.0]);
    // window_min == window_max -> all pixels -> out_min
    let f = IntensityWindowingFilter::new(100.0, 100.0, 3.0, 7.0);
    let result = get_vals(&f.apply(&img).unwrap());
    for &v in &result {
        assert!(
            (v - 3.0).abs() < 1e-6,
            "equal window -> out_min=3.0, got {}",
            v
        );
    }
}

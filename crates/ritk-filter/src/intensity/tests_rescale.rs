use super::*;
use burn_ndarray::NdArray;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
type B = NdArray<f32>;

fn make_image(vals: Vec<f32>) -> Image<B, 3> {
    let n = vals.len();
    let device = Default::default();
    let td = TensorData::new(vals, Shape::from([1, 1, n]));
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
fn test_uniform_image_gives_out_min() {
    let img = make_image(vec![5.0_f32; 8]);
    let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
    let vals = get_vals(&out);
    for &v in &vals {
        assert!(
            (v - 0.0).abs() < 1e-6,
            "uniform image -> out_min=0.0, got {}",
            v
        );
    }
}

#[test]
fn test_ramp_rescale_to_unit() {
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_image(vals);
    let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
    let result = get_vals(&out);
    assert!(
        (result[0] - 0.0).abs() < 1e-6,
        "min -> 0.0, got {}",
        result[0]
    );
    assert!(
        (result[9] - 1.0).abs() < 1e-6,
        "max -> 1.0, got {}",
        result[9]
    );
    // Intermediate: (5 - 0) / (9 - 0) = 5/9
    let expected_mid = 5.0_f32 / 9.0;
    assert!(
        (result[5] - expected_mid).abs() < 1e-5,
        "mid -> {}, got {}",
        expected_mid,
        result[5]
    );
}

#[test]
fn test_custom_output_range() {
    let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let img = make_image(vals);
    let out = RescaleIntensityFilter::new(2.0, 5.0).apply(&img).unwrap();
    let result = get_vals(&out);
    assert!(
        (result[0] - 2.0).abs() < 1e-5,
        "min -> 2.0, got {}",
        result[0]
    );
    assert!(
        (result[9] - 5.0).abs() < 1e-5,
        "max -> 5.0, got {}",
        result[9]
    );
}

#[test]
fn test_negative_values_rescaled() {
    let vals: Vec<f32> = (-5i32..=5).map(|i| i as f32).collect(); // -5..=5
    let img = make_image(vals);
    let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
    let result = get_vals(&out);
    assert!(
        (result[0] - 0.0).abs() < 1e-5,
        "min=-5 -> 0.0, got {}",
        result[0]
    );
    assert!(
        (result[10] - 1.0).abs() < 1e-5,
        "max=5 -> 1.0, got {}",
        result[10]
    );
}

#[test]
fn test_single_voxel_gives_out_min() {
    let img = make_image(vec![42.0_f32]);
    let out = RescaleIntensityFilter::new(3.0, 7.0).apply(&img).unwrap();
    let result = get_vals(&out);
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 3.0).abs() < 1e-6,
        "single voxel -> out_min=3.0, got {}",
        result[0]
    );
}

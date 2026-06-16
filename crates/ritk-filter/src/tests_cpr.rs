#![allow(clippy::identity_op, clippy::erasing_op)]

use super::*;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn make_image_with_metadata(
    vals: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<B, 3> {
    ts::make_image_with::<B, 3>(vals, dims, Some(origin), Some(spacing), Some(direction))
}

#[test]
fn constant_image_uniform_output() {
    let img = make_image(vec![1.0_f32; 8 * 8 * 8], [8, 8, 8]);
    let cpr = CprImageFilter::new(
        vec![[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        CprConfig {
            num_path_samples: 10,
            cross_section_half_width: 1.0,
            num_cross_samples: 10,
        },
    );
    let out = cpr.apply(&img).unwrap();
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v.len(), 100);
    for &x in &v {
        assert!(
            (x - 1.0_f32).abs() < 1e-4,
            "constant image must produce uniform CPR output: {x}"
        );
    }
}

#[test]
fn linear_path_along_z() {
    let mut vals = vec![0.0_f32; 10 * 10 * 10];
    for iz in 0..10 {
        for iy in 0..10 {
            for ix in 0..10 {
                vals[iz * 100 + iy * 10 + ix] = (iy as f32) + (ix as f32);
            }
        }
    }
    let img = make_image(vals, [10, 10, 10]);
    let cpr = CprImageFilter::new(
        vec![[2.0, 4.5, 4.5], [6.0, 4.5, 4.5]],
        CprConfig {
            num_path_samples: 5,
            cross_section_half_width: 1.5,
            num_cross_samples: 5,
        },
    );
    let out = cpr.apply(&img).unwrap();
    let (v, dims) = extract_vec_infallible(&out);
    assert_eq!(dims, [5, 5]);
    let center_val = v[2 * 5 + 2];
    assert!(
        (center_val - 9.0_f32).abs() < 0.5,
        "center of path should sample near (y=4.5, x=4.5): got {center_val}"
    );
}

#[test]
fn handles_non_zero_origin() {
    let mut vals = vec![0.0_f32; 6 * 6 * 6];
    vals[3 * 36 + 3 * 6 + 3] = 100.0_f32;
    let img = make_image_with_metadata(
        vals,
        [6, 6, 6],
        Point::new([5.0, 5.0, 5.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let cpr = CprImageFilter::new(
        vec![[8.0, 8.0, 8.0], [8.1, 8.0, 8.0]],
        CprConfig {
            num_path_samples: 2,
            cross_section_half_width: 0.5,
            num_cross_samples: 3,
        },
    );
    let out = cpr.apply(&img).unwrap();
    let (v, _) = extract_vec_infallible(&out);
    let center = v[1 * 2 + 0];
    assert!(
        center > 50.0,
        "marker voxel should be sampled near center: got {center}"
    );
}

#[test]
fn handles_non_unit_spacing() {
    let mut vals = vec![0.0_f32; 10 * 10 * 10];
    vals[2 * 100 + 5 * 10 + 5] = 100.0_f32;
    let img = make_image_with_metadata(
        vals,
        [10, 10, 10],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction::identity(),
    );
    let cpr = CprImageFilter::new(
        vec![[4.0, 15.0, 20.0], [4.1, 15.0, 20.0]],
        CprConfig {
            num_path_samples: 2,
            cross_section_half_width: 1.0,
            num_cross_samples: 3,
        },
    );
    let out = cpr.apply(&img).unwrap();
    let (v, _) = extract_vec_infallible(&out);
    let center = v[1 * 2 + 0];
    assert!(
        center > 50.0,
        "non-unit spacing: marker should be at expected physical position: got {center}"
    );
}

#[test]
fn insufficient_control_points_errors() {
    let img = make_image(vec![0.0_f32; 8], [2, 2, 2]);
    let cpr = CprImageFilter::new(vec![[1.0, 1.0, 1.0]], CprConfig::default());
    assert!(cpr.apply(&img).is_err());
}

#[test]
fn zero_length_path_errors() {
    let img = make_image(vec![0.0_f32; 8], [2, 2, 2]);
    let cpr = CprImageFilter::new(vec![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], CprConfig::default());
    assert!(cpr.apply(&img).is_err());
}

#[test]
fn output_shape_matches_config() {
    let img = make_image(vec![1.0_f32; 27], [3, 3, 3]);
    let cpr = CprImageFilter::new(
        vec![[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        CprConfig {
            num_path_samples: 7,
            cross_section_half_width: 0.5,
            num_cross_samples: 5,
        },
    );
    let out = cpr.apply(&img).unwrap();
    assert_eq!(out.shape(), [5, 7]);
}

#[test]
fn output_metadata_is_reasonable() {
    let img = make_image(vec![1.0_f32; 27], [3, 3, 3]);
    let cpr = CprImageFilter::new(
        vec![[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        CprConfig {
            num_path_samples: 10,
            cross_section_half_width: 5.0,
            num_cross_samples: 10,
        },
    );
    let out = cpr.apply(&img).unwrap();
    assert!((out.origin()[0] + 5.0).abs() < 1e-6);
    assert!((out.origin()[1] - 0.0).abs() < 1e-6);
    assert!((out.spacing()[0] - 10.0 / 9.0).abs() < 1e-6);
    assert!((out.spacing()[1] - 1.0 / 9.0).abs() < 1e-6);
}

#[test]
fn generate_path_batch_matches_scalar() {
    let control_points: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [3.0, 1.0, 4.0],
        [5.0, 5.0, 5.0],
        [7.0, 0.0, 3.0],
    ];
    let num_samples = 1000;
    let scalar = generate_path(&control_points, num_samples);
    let batch = generate_path_batch(&control_points, num_samples);
    assert_eq!(scalar.len(), batch.len());
    for (s, b) in scalar.iter().zip(batch.iter()) {
        for k in 0..3 {
            assert!(
                (s[k] - b[k]).abs() < 1e-12,
                "mismatch at component {k}: scalar={}, batch={}",
                s[k],
                b[k]
            );
        }
    }
}

#[test]
fn generate_path_batch_empty() {
    let control_points: Vec<[f64; 3]> = vec![];
    let scalar = generate_path(&control_points, 100);
    let batch = generate_path_batch(&control_points, 100);
    assert!(scalar.is_empty());
    assert!(batch.is_empty());
}

#[test]
fn generate_path_batch_single_point() {
    let control_points: Vec<[f64; 3]> = vec![[42.0, 7.0, 13.0]];
    let num_samples = 50;
    let scalar = generate_path(&control_points, num_samples);
    let batch = generate_path_batch(&control_points, num_samples);
    assert_eq!(scalar.len(), num_samples);
    assert_eq!(batch.len(), num_samples);
    for (s, b) in scalar.iter().zip(batch.iter()) {
        for k in 0..3 {
            assert!((s[k] - b[k]).abs() < 1e-12);
        }
    }
}

#[test]
fn physical_to_index_identity() {
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let idx = physical_to_index(&[5.0, 10.0, 15.0], &origin, &spacing, &direction);
    assert!((idx[0] - 5.0).abs() < 1e-10);
    assert!((idx[1] - 10.0).abs() < 1e-10);
    assert!((idx[2] - 15.0).abs() < 1e-10);
}

#[test]
fn physical_to_index_non_identity() {
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([2.0, 3.0, 4.0]);
    let direction = Direction::identity();
    let idx = physical_to_index(&[20.0, 50.0, 70.0], &origin, &spacing, &direction);
    assert!((idx[0] - 5.0).abs() < 1e-10);
    assert!((idx[1] - 10.0).abs() < 1e-10);
    assert!((idx[2] - 10.0).abs() < 1e-10);
}

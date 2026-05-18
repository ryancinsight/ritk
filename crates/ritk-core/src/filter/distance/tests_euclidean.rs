//! Tests for euclidean
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// --- edt_3d unit tests ---------------------------------------------------

#[test]
fn edt_3d_single_foreground_voxel_at_origin() {
    // 5x5x5 volume, single foreground at (0,0,0)
    let dims = [5usize, 5, 5];
    let mut fg = vec![false; 5 * 5 * 5];
    fg[0] = true; // iz=0, iy=0, ix=0
    let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
    // Voxel (0,0,0): distance 0
    assert!((dt[0] - 0.0).abs() < 1e-5);
    // Voxel (0,0,1): distance 1
    let idx = 0 * 25 + 0 * 5 + 1;
    assert!(
        (dt[idx] - 1.0).abs() < 1e-4,
        "expected 1.0, got {}",
        dt[idx]
    );
    // Voxel (0,1,0): distance 1
    let idx = 0 * 25 + 1 * 5 + 0;
    assert!(
        (dt[idx] - 1.0).abs() < 1e-4,
        "expected 1.0, got {}",
        dt[idx]
    );
    // Voxel (1,1,1): distance sqrt(3) ≈ 1.732
    let idx = 1 * 25 + 1 * 5 + 1;
    assert!(
        (dt[idx] - 3.0_f64.sqrt() as f32).abs() < 1e-4,
        "expected sqrt(3), got {}",
        dt[idx]
    );
}

#[test]
fn edt_3d_all_foreground_gives_zero_everywhere() {
    let dims = [4usize, 4, 4];
    let fg = vec![true; 64];
    let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
    for (i, &v) in dt.iter().enumerate() {
        assert!((v - 0.0).abs() < 1e-5, "voxel {} expected 0, got {}", i, v);
    }
}

#[test]
fn edt_3d_two_foreground_voxels_midpoint() {
    // 1×1×5 volume, foreground at ix=0 and ix=4
    let dims = [1usize, 1, 5];
    let fg = vec![true, false, false, false, true];
    let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
    // Distances: 0, 1, 2, 1, 0
    let expected = [0.0f32, 1.0, 2.0, 1.0, 0.0];
    for (i, (&d, &e)) in dt.iter().zip(expected.iter()).enumerate() {
        assert!((d - e).abs() < 1e-4, "ix={}: expected {}, got {}", i, e, d);
    }
}

#[test]
fn edt_3d_anisotropic_spacing_scales_distance() {
    // 1×1×3 with spacing sx=2.0; foreground at ix=0 only
    let dims = [1usize, 1, 3];
    let fg = vec![true, false, false];
    let dt = edt_3d(&fg, dims, [1.0, 1.0, 2.0]);
    // Distances: 0, 2, 4 (in mm with sx=2)
    assert!((dt[0] - 0.0).abs() < 1e-4);
    assert!((dt[1] - 2.0).abs() < 1e-4, "expected 2.0, got {}", dt[1]);
    assert!((dt[2] - 4.0).abs() < 1e-4, "expected 4.0, got {}", dt[2]);
}

// --- DistanceTransformImageFilter tests ----------------------------------

#[test]
fn unsigned_edt_filter_preserves_spatial_metadata() {
    let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2]);
    let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.origin(), img.origin());
}

#[test]
fn unsigned_edt_filter_foreground_voxel_receives_zero() {
    let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2]);
    let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    // iz=0,iy=0,ix=0 is foreground → distance 0
    assert!(
        (v[0] - 0.0).abs() < 1e-4,
        "foreground voxel expected 0, got {}",
        v[0]
    );
}

#[test]
fn unsigned_edt_filter_background_voxels_have_positive_distance() {
    // Single foreground at (0,0,0) in a 3×3×3 volume
    let mut vals = vec![0.0f32; 27];
    vals[0] = 1.0;
    let img = make_image(vals, [3, 3, 3]);
    let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    // All non-foreground voxels must have distance > 0
    for (i, &d) in v.iter().enumerate() {
        if i == 0 {
            assert!((d - 0.0).abs() < 1e-4);
        } else {
            assert!(d > 0.0, "voxel {} expected positive distance, got {}", i, d);
        }
    }
}

// --- SignedDistanceTransformImageFilter tests ----------------------------

#[test]
fn signed_edt_filter_inside_negative_outside_positive() {
    // 1×1×5: foreground is ix=[1,2,3], background is ix=[0,4]
    let vals = vec![0.0f32, 1.0, 1.0, 1.0, 0.0];
    let img = make_image(vals, [1, 1, 5]);
    let out = SignedDistanceTransformImageFilter::new()
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    // ix=0 (background): positive distance to nearest fg (ix=1) = 1
    assert!(v[0] > 0.0, "background expected positive, got {}", v[0]);
    assert!((v[0] - 1.0).abs() < 1e-4, "expected +1, got {}", v[0]);
    // ix=1 (foreground): negative distance to nearest bg (ix=0) = −1
    assert!(
        v[1] < 0.0,
        "foreground edge expected negative, got {}",
        v[1]
    );
    assert!((v[1] - (-1.0)).abs() < 1e-4, "expected -1, got {}", v[1]);
    // ix=2 (foreground center): distance to nearest bg is 2
    assert!(
        v[2] < 0.0,
        "foreground center expected negative, got {}",
        v[2]
    );
    assert!((v[2] - (-2.0)).abs() < 1e-4, "expected -2, got {}", v[2]);
    // ix=4 (background): positive 1
    assert!(v[4] > 0.0, "background expected positive, got {}", v[4]);
}

#[test]
fn signed_edt_filter_preserves_spatial_metadata() {
    let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2]);
    let out = SignedDistanceTransformImageFilter::new()
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
}

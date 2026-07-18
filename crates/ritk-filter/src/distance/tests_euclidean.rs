//! Tests for euclidean
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// --- edt_3d unit tests ---------------------------------------------------

#[test]
#[allow(clippy::erasing_op, clippy::identity_op)]
fn edt_3d_single_foreground_voxel_at_origin() {
    // 5x5x5 volume, single foreground at (0,0,0)
    let dims = [5usize, 5, 5];
    let mut fg = vec![false; 5 * 5 * 5];
    fg[0] = true; // iz=0, iy=0, ix=0
    let dt = euclidean_dt(&fg, dims, [1.0, 1.0, 1.0]);
    // Voxel (0,0,0): distance 0
    assert!((dt[0] - 0.0).abs() < 1e-5);
    // Voxel (0,0,1): distance 1 â€” index formula: iz * ny * nx + iy * nx + ix
    let ny = dims[1];
    let nx = dims[2];
    let idx = 0 * ny * nx + 0 * nx + 1;
    assert!(
        (dt[idx] - 1.0).abs() < 1e-4,
        "expected 1.0, got {}",
        dt[idx]
    );
    // Voxel (0,1,0): distance 1
    let idx = 0 * ny * nx + 1 * nx + 0;
    assert!(
        (dt[idx] - 1.0).abs() < 1e-4,
        "expected 1.0, got {}",
        dt[idx]
    );
    // Voxel (1,1,1): distance sqrt(3) â‰ˆ 1.732
    let idx = 25 + 5 + 1;
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
    let dt = euclidean_dt(&fg, dims, [1.0, 1.0, 1.0]);
    for (i, &v) in dt.iter().enumerate() {
        assert!((v - 0.0).abs() < 1e-5, "voxel {} expected 0, got {}", i, v);
    }
}

#[test]
fn edt_3d_two_foreground_voxels_midpoint() {
    // 1Ã—1Ã—5 volume, foreground at ix=0 and ix=4
    let dims = [1usize, 1, 5];
    let fg = vec![true, false, false, false, true];
    let dt = euclidean_dt(&fg, dims, [1.0, 1.0, 1.0]);
    // Distances: 0, 1, 2, 1, 0
    let expected = [0.0f32, 1.0, 2.0, 1.0, 0.0];
    for (i, (&d, &e)) in dt.iter().zip(expected.iter()).enumerate() {
        assert!((d - e).abs() < 1e-4, "ix={}: expected {}, got {}", i, e, d);
    }
}

#[test]
fn edt_3d_anisotropic_spacing_scales_distance() {
    // 1Ã—1Ã—3 with spacing sx=2.0; foreground at ix=0 only
    let dims = [1usize, 1, 3];
    let fg = vec![true, false, false];
    let dt = euclidean_dt(&fg, dims, [1.0, 1.0, 2.0]);
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
    // iz=0,iy=0,ix=0 is foreground â†’ distance 0
    assert!(
        (v[0] - 0.0).abs() < 1e-4,
        "foreground voxel expected 0, got {}",
        v[0]
    );
}

#[test]
fn unsigned_edt_filter_background_voxels_have_positive_distance() {
    // Single foreground at (0,0,0) in a 3Ã—3Ã—3 volume
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

#[test]
fn unsigned_squared_measure_has_exact_grid_values() {
    let img = make_image(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 4]);
    let out = DistanceTransformImageFilter::new()
        .with_measure(crate::distance::DistanceMeasure::Squared)
        .apply(&img)
        .expect("valid squared transform");
    assert_eq!(voxels(&out), vec![0.0, 1.0, 4.0, 9.0]);
}

#[test]
fn unsigned_empty_seed_set_is_zero() {
    let img = make_image(vec![0.0; 8], [2, 2, 2]);
    let out = DistanceTransformImageFilter::new()
        .apply(&img)
        .expect("empty seed set has a defined zero result");
    assert_eq!(voxels(&out), vec![0.0; 8]);
}

#[test]
fn unsigned_filter_rejects_invalid_numeric_inputs() {
    for (value, display) in [
        (f32::NAN, "NaN"),
        (f32::INFINITY, "inf"),
        (f32::NEG_INFINITY, "-inf"),
    ] {
        let img = make_image(vec![value; 8], [2, 2, 2]);
        let error = match DistanceTransformImageFilter::new().apply(&img) {
            Err(error) => error,
            Ok(_) => panic!("non-finite sample must be rejected"),
        };
        assert_eq!(
            error.to_string(),
            format!("distance-transform sample at flat index 0 must be finite, got {display}")
        );
    }
    for threshold in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            crate::distance::BinarizationThreshold::new(threshold),
            Err("BinarizationThreshold must be finite and non-negative")
        );
    }
}

#[test]
fn unsigned_validation_reports_invalid_shape_and_spacing_exactly() {
    let threshold = crate::distance::BinarizationThreshold::DEFAULT;
    let zero_dimension =
        super::unsigned::validate_input(&[], [0, 1, 1], [1.0, 1.0, 1.0], threshold)
            .expect_err("zero dimension must be rejected");
    assert_eq!(
        zero_dimension.to_string(),
        "distance-transform dimensions must be non-zero, got [0, 1, 1]"
    );
    for spacing in [
        [0.0, 1.0, 1.0],
        [f64::NAN, 1.0, 1.0],
        [f64::INFINITY, 1.0, 1.0],
    ] {
        let error = super::unsigned::validate_input(&[0.0], [1, 1, 1], spacing, threshold)
            .expect_err("invalid spacing must be rejected");
        assert_eq!(
            error.to_string(),
            format!("distance-transform spacing must be finite and positive, got {spacing:?}")
        );
    }
}

// --- SignedDistanceTransformImageFilter tests ----------------------------

#[test]
fn signed_edt_filter_inside_negative_outside_positive() {
    // 1Ã—1Ã—5: foreground is ix=[1,2,3], background is ix=[0,4]
    let vals = vec![0.0f32, 1.0, 1.0, 1.0, 0.0];
    let img = make_image(vals, [1, 1, 5]);
    let out = SignedDistanceTransformImageFilter::new()
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    // ix=0 (background): positive distance to nearest fg (ix=1) = 1
    assert!(v[0] > 0.0, "background expected positive, got {}", v[0]);
    assert!((v[0] - 1.0).abs() < 1e-4, "expected +1, got {}", v[0]);
    // ix=1 (foreground): negative distance to nearest bg (ix=0) = âˆ’1
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

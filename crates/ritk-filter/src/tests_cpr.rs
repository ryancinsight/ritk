#![allow(clippy::identity_op, clippy::erasing_op)]

use super::*;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn make_image_with_metadata(
    vals: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, B, 3> {
    ts::make_image_with::<f32, B, 3>(vals, dims, Some(origin), Some(spacing), Some(direction))
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
fn native_constant_image_preserves_values_and_cpr_geometry() {
    let image = NativeImage::from_flat_on(
        vec![1.0_f32; 8 * 8 * 8],
        [8, 8, 8],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = CprImageFilter::new(
        vec![[2.0, 2.0, 2.0], [4.0, 2.0, 2.0]],
        CprConfig {
            num_path_samples: 4,
            cross_section_half_width: 1.0,
            num_cross_samples: 3,
        },
    )
    .apply_native(&image, &SequentialBackend)
    .expect("native CPR succeeds");

    assert_eq!(output.shape(), [3, 4]);
    assert_eq!(
        output
            .data_slice()
            .expect("invariant: sequential storage is contiguous"),
        &[1.0_f32; 12]
    );
    assert_eq!(*output.origin(), Point::new([-1.0, 0.0]));
    assert_eq!(*output.spacing(), Spacing::new([1.0, 2.0 / 3.0]));
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

#[path = "tests_cpr_ref.rs"]
mod tests_cpr_ref;
use tests_cpr_ref::cpr_apply_reference;

#[test]
fn cpr_apply_matches_brute_force_reference() {
    // Deterministic, non-trivial volume: linear ramp + a bright voxel so the
    // cross-section trilinear interpolation exercises direction-matrix and
    // non-unit spacing paths.
    let dim: usize = 12;
    let mut vals = vec![0.0_f32; dim * dim * dim];
    for iz in 0..dim {
        for iy in 0..dim {
            for ix in 0..dim {
                vals[iz * dim * dim + iy * dim + ix] = (iz as f32) * 0.5 + (iy + ix) as f32 * 0.25;
            }
        }
    }
    // Bright marker to exercise non-axis-aligned trilinear mixing.
    let mid = dim / 2;
    vals[mid * dim * dim + mid * dim + mid] = 100.0_f32;

    let image = make_image_with_metadata(
        vals,
        [dim, dim, dim],
        Point::new([0.5, -1.0, 2.0]),
        Spacing::new([1.5, 2.0, 0.75]),
        Direction::identity(),
    );

    let cpr = CprImageFilter::new(
        vec![[3.0, 4.0, 5.0], [9.0, 14.0, 8.0], [15.0, 7.0, 4.0]],
        CprConfig {
            num_path_samples: 24,
            cross_section_half_width: 4.0,
            num_cross_samples: 16,
        },
    );

    let opt = cpr.apply(&image).expect("optimized apply");
    let ref_out = cpr_apply_reference(&cpr, &image).expect("reference apply");

    let (v_opt, d_opt) = extract_vec_infallible(&opt);
    let (v_ref, d_ref) = extract_vec_infallible(&ref_out);
    assert_eq!(d_opt, d_ref, "shape mismatch optimized vs reference");
    assert_eq!(v_opt.len(), v_ref.len(), "length mismatch");

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (a, b) in v_opt.iter().zip(v_ref.iter()) {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let denom = a.abs().max(b.abs()).max(f32::EPSILON);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_abs <= 1e-5,
        "max |Î”| between optimized and reference = {max_abs} exceeds 1e-5"
    );
    assert!(max_rel <= 1e-5, "max relative Î” = {max_rel} exceeds 1e-5");
}

#[test]
fn cpr_apply_matches_brute_force_reference_nonidentity_direction() {
    // Stress the direction matrix: 90Â° rotation about Z maps (x, y) â†’ (-y, x)
    // which forces inv_dir â‰  identity and exercises every matrix entry of
    // the hoisted transform.
    use ritk_core::image::Image as RitkImage;
    let dim: usize = 10;
    let mut vals = vec![0.0_f32; dim * dim * dim];
    for iz in 0..dim {
        for iy in 0..dim {
            for ix in 0..dim {
                vals[iz * dim * dim + iy * dim + ix] = (iz * 7 + iy * 11 + ix * 13) as f32;
            }
        }
    }

    // 90Â° rotation about Z in RITK [z, y, x] convention: direction columns
    // are the image-axis vectors expressed in physical space. We place the
    // first two columns as the rotated (y, -x) pair, third as (0, 0, 1).
    let data = ritk_image::tensor::Tensor::<f32, B>::from_slice([dim, dim, dim], &vals);
    let mut direction = Direction::identity();
    {
        let m = direction.inner_mut();
        m[(0, 0)] = 0.0;
        m[(0, 1)] = -1.0;
        m[(1, 0)] = 1.0;
        m[(1, 1)] = 0.0;
    }
    let image = RitkImage::new(
        data,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.5, 2.0]),
        direction,
    );

    let cpr = CprImageFilter::new(
        vec![[3.0, 2.0, 1.0], [5.0, 6.0, 9.0]],
        CprConfig {
            num_path_samples: 12,
            cross_section_half_width: 2.0,
            num_cross_samples: 8,
        },
    );

    let opt = cpr.apply(&image).expect("optimized apply");
    let ref_out = cpr_apply_reference(&cpr, &image).expect("reference apply");

    let (v_opt, _) = extract_vec_infallible(&opt);
    let (v_ref, _) = extract_vec_infallible(&ref_out);
    assert_eq!(v_opt.len(), v_ref.len());
    let mut max_abs = 0.0_f32;
    for (a, b) in v_opt.iter().zip(v_ref.iter()) {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs <= 1e-5,
        "non-identity direction: max |Î”| = {max_abs} exceeds 1e-5"
    );
}

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

/// Reference implementation of `CprImageFilter::apply` written in the
/// pre-optimization form (no hoisted inverse, no per-path-point basis).
/// Used by [`cpr_apply_matches_brute_force_reference`] to lock value
/// semantics of the optimized kernel against the explicit mathematical
/// formulation.
#[allow(clippy::too_many_lines)]
fn cpr_apply_reference<B: burn::tensor::backend::Backend>(
    cpr: &CprImageFilter,
    image: &ritk_core::image::Image<B, 3>,
) -> anyhow::Result<ritk_core::image::Image<B, 2>> {
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_tensor_ops::extract_vec;

    let (vals, dims) = extract_vec(image)?;
    let [nz, ny, nx] = dims;

    if cpr.control_points.len() < CPR_MIN_CONTROL_POINTS {
        anyhow::bail!(
            "CPR requires at least {CPR_MIN_CONTROL_POINTS} control points, got {}",
            cpr.control_points.len()
        );
    }

    let origin = *image.origin();
    let spacing = *image.spacing();
    let direction = *image.direction();

    let num_path = cpr.config.num_path_samples;
    let num_cross = cpr.config.num_cross_samples;
    let half_width = cpr.config.cross_section_half_width;

    let dense_pts = generate_path_batch(&cpr.control_points, num_path * CPR_DENSE_FACTOR);

    let mut arc_lengths = vec![0.0_f64; dense_pts.len()];
    for i in 1..dense_pts.len() {
        let (ax, ay, az) = (
            dense_pts[i][0] - dense_pts[i - 1][0],
            dense_pts[i][1] - dense_pts[i - 1][1],
            dense_pts[i][2] - dense_pts[i - 1][2],
        );
        arc_lengths[i] = arc_lengths[i - 1] + (ax * ax + ay * ay + az * az).sqrt();
    }
    let total_length = arc_lengths[dense_pts.len() - 1];
    if total_length < 1e-12 {
        anyhow::bail!("CPR path has zero total length — all control points coincident");
    }

    let mut path_pts = Vec::with_capacity(num_path);
    for i in 0..num_path {
        let target_arc = if num_path > 1 {
            (i as f64 / (num_path - 1) as f64) * total_length
        } else {
            0.0
        };
        let seg_idx = match arc_lengths.binary_search_by(|&a| {
            a.partial_cmp(&target_arc)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };
        let seg = seg_idx.max(1).min(dense_pts.len() - 1);
        let seg_prev = seg - 1;
        let seg_len = arc_lengths[seg] - arc_lengths[seg_prev];
        let frac = if seg_len > 0.0 {
            (target_arc - arc_lengths[seg_prev]) / seg_len
        } else {
            0.0
        };
        let p = [
            dense_pts[seg_prev][0] + frac * (dense_pts[seg][0] - dense_pts[seg_prev][0]),
            dense_pts[seg_prev][1] + frac * (dense_pts[seg][1] - dense_pts[seg_prev][1]),
            dense_pts[seg_prev][2] + frac * (dense_pts[seg][2] - dense_pts[seg_prev][2]),
        ];
        path_pts.push(p);
    }

    let mut output = vec![0.0_f32; num_cross * num_path];
    for i in 0..num_path {
        let p = &path_pts[i];
        let tangent = if num_path > 1 {
            let prev = if i > 0 {
                &path_pts[i - 1]
            } else {
                &path_pts[0]
            };
            let next = if i < num_path - 1 {
                &path_pts[i + 1]
            } else {
                &path_pts[num_path - 1]
            };
            [next[0] - prev[0], next[1] - prev[1], next[2] - prev[2]]
        } else {
            [0.0, 0.0, 1.0]
        };
        let (v_up, _v_right) = cross_section_basis(&tangent);
        for j in 0..num_cross {
            let offset = if num_cross > 1 {
                (j as f64 / (num_cross - 1) as f64 - 0.5) * 2.0 * half_width
            } else {
                0.0
            };
            let sample = [
                p[0] + v_up[0] * offset,
                p[1] + v_up[1] * offset,
                p[2] + v_up[2] * offset,
            ];
            let idx = j * num_path + i;
            output[idx] =
                trilinear_sample(&vals, [nz, ny, nx], &origin, &spacing, &direction, &sample);
        }
    }

    let device = image.data().device();
    let td_out = TensorData::new(output, Shape::new([num_cross, num_path]));
    let tensor = Tensor::<B, 2>::from_data(td_out, &device);

    let cs_step = if num_cross > 1 {
        2.0 * half_width / (num_cross - 1) as f64
    } else {
        1.0
    };
    let path_step = if num_path > 1 {
        total_length / (num_path - 1) as f64
    } else {
        1.0
    };

    Ok(ritk_core::image::Image::new(
        tensor,
        Point::new([-half_width, 0.0]),
        Spacing::new([cs_step, path_step]),
        Direction::identity(),
    ))
}

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
        "max |Δ| between optimized and reference = {max_abs} exceeds 1e-5"
    );
    assert!(max_rel <= 1e-5, "max relative Δ = {max_rel} exceeds 1e-5");
}

#[test]
fn cpr_apply_matches_brute_force_reference_nonidentity_direction() {
    // Stress the direction matrix: 90° rotation about Z maps (x, y) → (-y, x)
    // which forces inv_dir ≠ identity and exercises every matrix entry of
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

    // 90° rotation about Z in RITK [z, y, x] convention: direction columns
    // are the image-axis vectors expressed in physical space. We place the
    // first two columns as the rotated (y, -x) pair, third as (0, 0, 1).
    let device = Default::default();
    let data = burn::tensor::Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(vals, burn::tensor::Shape::new([dim, dim, dim])),
        &device,
    );
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
        "non-identity direction: max |Δ| = {max_abs} exceeds 1e-5"
    );
}

use super::*;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec;

/// Reference implementation of `CprImageFilter::apply` written in the
/// pre-optimization form (no hoisted inverse, no per-path-point basis).
/// Used by [`cpr_apply_matches_brute_force_reference`] to lock value
/// semantics of the optimized kernel against the explicit mathematical
/// formulation.
#[allow(clippy::too_many_lines)]
pub fn cpr_apply_reference<B: Backend>(
    cpr: &CprImageFilter,
    image: &Image<B, 3>,
) -> anyhow::Result<Image<B, 2>> {
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

    Ok(Image::new(
        tensor,
        Point::new([-half_width, 0.0]),
        Spacing::new([cs_step, path_step]),
        Direction::identity(),
    ))
}

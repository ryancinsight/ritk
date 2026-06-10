//! Curved Planar Reformation (CPR) filter.
//!
//! # Mathematical Specification
//!
//! CPR takes a 3-D image and a set of control points defining a curved path,
//! then generates a 2-D "straightened" view by extracting cross-sectional
//! planes perpendicular to the path tangent.
//!
//! ## Algorithm
//!
//! 1. **Path generation**: Control points are interpolated with a Catmull-Rom
//!    spline, oversampled at `10 × num_path_samples`, then resampled at
//!    evenly-spaced arc-length intervals to produce `num_path_samples` points.
//!
//! 2. **Cross-section basis**: At each path point, the tangent is estimated
//!    by central finite difference of adjacent path points. An orthonormal
//!    frame `{tangent, up, right}` is constructed via Gram-Schmidt against a
//!    reference axis (world Z or world X when tangent is near Z). The `up`
//!    and `right` vectors span the cross-section plane.
//!
//! 3. **Sampling**: For each path point, `num_cross_samples` points are
//!    sampled along the `up` direction within `[-half_width, +half_width]`
//!    physical units (mm). Each sample is evaluated via trilinear
//!    interpolation of the image voxel data with boundary clamping.
//!
//! ## Output geometry
//!
//! The output is a 2-D image where:
//! - **Rows** correspond to cross-section offset (perpendicular to path)
//! - **Columns** correspond to distance along path
//!
//! Spatial metadata:
//! - Origin: `Point2(-half_width, 0.0)` — start of cross-section, start of path
//! - Spacing: `(2·half_width / (num_cross-1), total_path_length / (num_path-1))`
//! - Direction: identity (CPR space is a straightened coordinate system)
//!
//! ## References
//! - Kanitsar, A. et al. (2002). "CPR — Curved Planar Reformation."
//!   *IEEE Visualization*, pp. 37–44.
//! - Catmull, E. and Rom, R. (1974). "A class of local interpolating splines."
//!   In *Computer Aided Geometric Design*, Academic Press.

use crate::filter::ops::extract_vec;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

#[path = "cpr_helpers.rs"]
pub(super) mod cpr_helpers;
pub(super) use cpr_helpers::*;
pub use cpr_helpers::{generate_path, generate_path_batch};

/// Minimum number of control points required for CPR.
pub const CPR_MIN_CONTROL_POINTS: usize = 2;

/// Default number of samples along the path (output columns).
pub const CPR_DEFAULT_PATH_SAMPLES: usize = 256;

/// Default cross-section half-width in physical units (mm).
pub const CPR_DEFAULT_HALF_WIDTH: f64 = 10.0;

/// Default number of cross-section samples (output rows).
pub const CPR_DEFAULT_CROSS_SAMPLES: usize = 64;

/// Dense oversampling factor for arc-length parameterisation.
const CPR_DENSE_FACTOR: usize = 10;

/// Configuration for [`CprImageFilter`].
#[derive(Debug, Clone, Copy)]
pub struct CprConfig {
    /// Number of samples along the path (output columns).
    pub num_path_samples: usize,
    /// Cross-section half-width in physical units (mm).
    pub cross_section_half_width: f64,
    /// Number of cross-section samples (output rows).
    pub num_cross_samples: usize,
}

impl Default for CprConfig {
    fn default() -> Self {
        Self {
            num_path_samples: CPR_DEFAULT_PATH_SAMPLES,
            cross_section_half_width: CPR_DEFAULT_HALF_WIDTH,
            num_cross_samples: CPR_DEFAULT_CROSS_SAMPLES,
        }
    }
}

/// Curved Planar Reformation (CPR) filter.
///
/// Takes a 3-D image and a set of 3-D control points (physical coordinates
/// `[z, y, x]`) defining a curved path, then generates a 2-D straightened
/// view by sampling cross-sectional planes perpendicular to the path tangent.
///
/// # Example
/// ```ignore
/// let filter = CprImageFilter::new(
///     vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 10.0, 0.0]],
///     CprConfig::default(),
/// );
/// let result: Image<B, 2> = filter.apply(&volume)?;
/// ```
#[derive(Debug, Clone)]
pub struct CprImageFilter {
    /// Control points in physical coordinates `[z, y, x]`.
    pub control_points: Vec<[f64; 3]>,
    /// Filter configuration.
    pub config: CprConfig,
}

impl CprImageFilter {
    /// Create a new CPR filter.
    pub fn new(control_points: Vec<[f64; 3]>, config: CprConfig) -> Self {
        Self {
            control_points,
            config,
        }
    }

    /// Apply the CPR filter.
    ///
    /// Returns a 2-D `Image` where rows = cross-section offset and columns = path position.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 2>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;

        if self.control_points.len() < CPR_MIN_CONTROL_POINTS {
            anyhow::bail!(
                "CPR requires at least {CPR_MIN_CONTROL_POINTS} control points, got {}",
                self.control_points.len()
            );
        }

        let origin = *image.origin();
        let spacing = *image.spacing();
        let direction = *image.direction();

        let num_path = self.config.num_path_samples;
        let num_cross = self.config.num_cross_samples;
        let half_width = self.config.cross_section_half_width;

        // ── 1. Generate arc-length-parameterised path ──────────────────────────
        let dense_pts = generate_path_batch(&self.control_points, num_path * CPR_DENSE_FACTOR);

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

        // ── 2. Sample cross-sections along the path ───────────────────────────
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

        // ── 3. Build 2-D output image ─────────────────────────────────────────
        let device = image.data().device();
        let td_out = TensorData::new(output, Shape::new([num_cross, num_path]));
        let tensor = Tensor::<B, 2>::from_data(td_out, &device);

        let cs_step = if num_cross > 1 {
            2.0 * half_width / (num_cross - 1) as f64
        } else {
            1.0 // degenerate 1-sample cross-section: use dummy spacing
        };
        let path_step = if num_path > 1 {
            total_length / (num_path - 1) as f64
        } else {
            1.0 // degenerate 1-sample path: use dummy spacing
        };

        Ok(Image::new(
            tensor,
            Point::new([-half_width, 0.0]),
            Spacing::new([cs_step, path_step]),
            Direction::identity(),
        ))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_cpr.rs"]
mod tests;

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
        let dense_pts = generate_path(&self.control_points, num_path * CPR_DENSE_FACTOR);

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
            0.0
        };
        let path_step = if num_path > 1 {
            total_length / (num_path - 1) as f64
        } else {
            0.0
        };

        Ok(Image::new(
            tensor,
            Point::new([-half_width, 0.0]),
            Spacing::new([cs_step, path_step]),
            Direction::identity(),
        ))
    }
}

// ── Catmull-Rom spline helpers ────────────────────────────────────────────

/// Evaluate the Catmull-Rom spline at parameter `t ∈ [0, 1]` for segment
/// bounded by control points `(p0, p1, p2, p3)`.
///
/// Standard Catmull-Rom basis (tension = 0.5, the uniform Catmull-Rom):
///
///   P(t) = 0.5 · (2·p₁ + (-p₀ + p₂)·t
///                 + (2·p₀ - 5·p₁ + 4·p₂ - p₃)·t²
///                 + (-p₀ + 3·p₁ - 3·p₂ + p₃)·t³)
fn catmull_rom_point(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
    t: f64,
) -> [f64; 3] {
    let t2 = t * t;
    let t3 = t2 * t;

    let a0 = 2.0 * p1[0];
    let a1 = -p0[0] + p2[0];
    let a2 = 2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0];
    let a3 = -p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0];
    let x = 0.5 * (a0 + a1 * t + a2 * t2 + a3 * t3);

    let b0 = 2.0 * p1[1];
    let b1 = -p0[1] + p2[1];
    let b2 = 2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1];
    let b3 = -p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1];
    let y = 0.5 * (b0 + b1 * t + b2 * t2 + b3 * t3);

    let c0 = 2.0 * p1[2];
    let c1 = -p0[2] + p2[2];
    let c2 = 2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2];
    let c3 = -p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2];
    let z = 0.5 * (c0 + c1 * t + c2 * t2 + c3 * t3);

    [x, y, z]
}

/// Generate `num_samples` evenly-parameterised points along a Catmull-Rom
/// path through the control points. End segments mirror the first / last
/// control point for boundary continuity.
fn generate_path(control_points: &[[f64; 3]], num_samples: usize) -> Vec<[f64; 3]> {
    let n = control_points.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![control_points[0]; num_samples];
    }

    let mut path = Vec::with_capacity(num_samples);
    let total_segments = (n - 1) as f64;

    for i in 0..num_samples {
        let t_total = if num_samples > 1 {
            i as f64 / (num_samples - 1) as f64
        } else {
            0.0
        };

        let seg_f = t_total * total_segments;
        let seg_idx = (seg_f as usize).min(n - 2);
        let t_local = seg_f - seg_idx as f64;

        let p0 = if seg_idx > 0 {
            &control_points[seg_idx - 1]
        } else {
            &control_points[0]
        };
        let p1 = &control_points[seg_idx];
        let p2 = &control_points[seg_idx + 1];
        let p3 = if seg_idx + 2 < n {
            &control_points[seg_idx + 2]
        } else {
            &control_points[n - 1]
        };

        path.push(catmull_rom_point(p0, p1, p2, p3, t_local));
    }

    path
}

// ── Cross-section basis ──────────────────────────────────────────────────

/// Construct an orthonormal basis `(up, right)` spanning the plane
/// perpendicular to `tangent`.
///
/// `up` is the component of the reference axis (world Z, falling back to
/// world X when tangent is near Z) orthogonal to the tangent, normalised.
/// `right = cross(tangent, up)`.  Degenerate (zero-length) tangents fall
/// back to the world-Z reference axis.
fn cross_section_basis(tangent: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let len = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let t = if len > 1e-12 {
        [tangent[0] / len, tangent[1] / len, tangent[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    };

    let ref_vec = if t[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    let dot = ref_vec[0] * t[0] + ref_vec[1] * t[1] + ref_vec[2] * t[2];
    let mut up = [
        ref_vec[0] - dot * t[0],
        ref_vec[1] - dot * t[1],
        ref_vec[2] - dot * t[2],
    ];
    let up_len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
    if up_len > 1e-12 {
        up = [up[0] / up_len, up[1] / up_len, up[2] / up_len];
    } else {
        up = [1.0, 0.0, 0.0];
    }

    let right = [
        t[1] * up[2] - t[2] * up[1],
        t[2] * up[0] - t[0] * up[2],
        t[0] * up[1] - t[1] * up[0],
    ];

    (up, right)
}

// ── Coordinate transforms ────────────────────────────────────────────────

/// Convert a physical-space point `[z, y, x]` to a continuous voxel index
/// using the image spatial metadata, following the convention:
///
///   index = D⁻¹ · (point − origin) / spacing   (element-wise)
fn physical_to_index(
    point: &[f64; 3],
    origin: &Point<3>,
    spacing: &Spacing<3>,
    direction: &Direction<3>,
) -> [f64; 3] {
    let pt = Point::new([point[0], point[1], point[2]]);
    let diff = pt - *origin;

    let inv_dir = direction
        .try_inverse()
        .expect("Direction matrix must be invertible");

    let rotated = inv_dir * diff;

    [
        rotated[0] / spacing[0],
        rotated[1] / spacing[1],
        rotated[2] / spacing[2],
    ]
}

// ── Trilinear interpolation ─────────────────────────────────────────────

/// Sample the image at `physical_point` using trilinear interpolation with
/// boundary clamping (edge-value extrapolation).
fn trilinear_sample(
    vals: &[f32],
    dims: [usize; 3],
    origin: &Point<3>,
    spacing: &Spacing<3>,
    direction: &Direction<3>,
    physical_point: &[f64; 3],
) -> f32 {
    let idx = physical_to_index(physical_point, origin, spacing, direction);
    let [nz, ny, nx] = dims;

    let iz = idx[0].clamp(0.0, (nz - 1) as f64);
    let iy = idx[1].clamp(0.0, (ny - 1) as f64);
    let ix = idx[2].clamp(0.0, (nx - 1) as f64);

    let iz0 = iz.floor() as usize;
    let iz1 = (iz0 + 1).min(nz - 1);
    let iy0 = iy.floor() as usize;
    let iy1 = (iy0 + 1).min(ny - 1);
    let ix0 = ix.floor() as usize;
    let ix1 = (ix0 + 1).min(nx - 1);

    let wz = iz - iz0 as f64;
    let wy = iy - iy0 as f64;
    let wx = ix - ix0 as f64;

    let idx000 = iz0 * ny * nx + iy0 * nx + ix0;
    let idx001 = iz0 * ny * nx + iy0 * nx + ix1;
    let idx010 = iz0 * ny * nx + iy1 * nx + ix0;
    let idx011 = iz0 * ny * nx + iy1 * nx + ix1;
    let idx100 = iz1 * ny * nx + iy0 * nx + ix0;
    let idx101 = iz1 * ny * nx + iy0 * nx + ix1;
    let idx110 = iz1 * ny * nx + iy1 * nx + ix0;
    let idx111 = iz1 * ny * nx + iy1 * nx + ix1;

    let v000 = vals[idx000] as f64;
    let v001 = vals[idx001] as f64;
    let v010 = vals[idx010] as f64;
    let v011 = vals[idx011] as f64;
    let v100 = vals[idx100] as f64;
    let v101 = vals[idx101] as f64;
    let v110 = vals[idx110] as f64;
    let v111 = vals[idx111] as f64;

    let v = (1.0 - wz) * (1.0 - wy) * (1.0 - wx) * v000
        + (1.0 - wz) * (1.0 - wy) * wx * v001
        + (1.0 - wz) * wy * (1.0 - wx) * v010
        + (1.0 - wz) * wy * wx * v011
        + wz * (1.0 - wy) * (1.0 - wx) * v100
        + wz * (1.0 - wy) * wx * v101
        + wz * wy * (1.0 - wx) * v110
        + wz * wy * wx * v111;

    v as f32
}

// ── Tests ────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_cpr.rs"]
mod tests;

//! Min/max curvature flow, matching `itk::MinMaxCurvatureFlowImageFilter`.
//!
//! # Mathematical Specification
//!
//! Min/max curvature flow is mean curvature flow whose speed is gated to suppress
//! the smoothing of features finer than a stencil of radius `R`: each voxel's
//! curvature-flow speed `v = |∇I|·κ` is replaced by `max(v, 0)` when the
//! `R`-sphere average of the neighbourhood is below a directional threshold, and
//! by `min(v, 0)` otherwise.  The threshold is the average of the two (2-D) or
//! four (3-D) neighbourhood samples perpendicular to the gradient at distance
//! `R` (ITK `MinMaxCurvatureFlowFunction::ComputeThreshold`).
//!
//! The base curvature-flow update reuses the exact discretization of
//! [`CurvatureFlowImageFilter`](super::CurvatureFlowImageFilter) (speed
//! `= N / |∇I|²`), and the effective time step is `time_step / (2·D)` where `D`
//! is the image dimension (2 for a `z = 1` slice, 3 otherwise) — the
//! ITK-internal CFL scaling that `MinMaxCurvatureFlowImageFilter` applies.
//!
//! # ITK / SimpleITK parity
//! Float-exact to `sitk.MinMaxCurvatureFlow` for `stencil_radius ≥ 2` (including
//! the SimpleITK default of 2), validated across many seeds, time steps, and
//! iteration counts (2-D `Dispatch<2>` for `z = 1`, 3-D `Dispatch<3>` otherwise).
//! `stencil_radius = 1` carries a residual ITK `R = 1` sampling subtlety and is
//! not bit-exact — a documented limit, not the common case.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Denominator floor below which curvature is treated as zero (flat region).
const GRAD_MAG_EPSILON: f64 = 1e-12;

/// Configuration for [`MinMaxCurvatureFlowImageFilter`].
#[derive(Debug, Clone, Copy)]
pub struct MinMaxCurvatureFlowConfig {
    /// Number of evolution iterations (ITK default 5).
    pub num_iterations: usize,
    /// User time step; the effective step is `time_step / (2·D)` (ITK default 0.05).
    pub time_step: f32,
    /// Stencil radius `R` for the min/max gate (ITK default 2).
    pub stencil_radius: usize,
}

impl Default for MinMaxCurvatureFlowConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            time_step: 0.05,
            stencil_radius: 2,
        }
    }
}

/// Min/max curvature flow filter.
#[derive(Debug, Clone, Copy)]
pub struct MinMaxCurvatureFlowImageFilter {
    /// Filter configuration.
    pub config: MinMaxCurvatureFlowConfig,
}

impl MinMaxCurvatureFlowImageFilter {
    /// Construct with the given configuration.
    pub fn new(config: MinMaxCurvatureFlowConfig) -> Self {
        Self { config }
    }

    /// Apply min/max curvature flow to a 3-D image (`z = 1` is treated as 2-D).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let slab = ny * nx;
        let r = self.config.stencil_radius as isize;
        let two_d = nz == 1;
        let d_dim = if two_d { 2.0 } else { 3.0 };
        let dt = (self.config.time_step as f64) / (2.0 * d_dim);

        let (sphere, sphere_w) = sphere_offsets(r, two_d);
        let rf = r as f64;

        let mut cur: Vec<f64> = vals_vec.iter().map(|&v| v as f64).collect();

        for _ in 0..self.config.num_iterations {
            let prev = cur.clone();
            let get = |zz: isize, yy: isize, xx: isize| -> f64 {
                let zc = zz.clamp(0, nz as isize - 1) as usize;
                let yc = yy.clamp(0, ny as isize - 1) as usize;
                let xc = xx.clamp(0, nx as isize - 1) as usize;
                prev[zc * slab + yc * nx + xc]
            };
            cur = (0..n)
                .map(|flat| {
                    let iz = flat / slab;
                    let rem = flat - iz * slab;
                    let iy = rem / nx;
                    let ix = rem - iy * nx;
                    let (z, y, x) = (iz as isize, iy as isize, ix as isize);
                    let c = prev[flat];
                    let (mut speed, dx_, dy_, dz_) = curvature_speed(&get, z, y, x, c);

                    if speed != 0.0 {
                        let threshold = if two_d {
                            threshold_2d(&get, z, y, x, dx_, dy_, rf)
                        } else {
                            threshold_3d(&get, z, y, x, dx_, dy_, dz_, rf)
                        };
                        // Sphere average (normalized inner product).
                        let mut avg = 0.0;
                        for o in &sphere {
                            avg += sphere_w * get(z + o[0], y + o[1], x + o[2]);
                        }
                        speed = if avg < threshold {
                            speed.max(0.0)
                        } else {
                            speed.min(0.0)
                        };
                    }
                    c + dt * speed
                })
                .collect();
        }

        Ok(rebuild(cur.iter().map(|&v| v as f32).collect(), dims, image))
    }
}

/// Build the normalized Euclidean-ball stencil offsets of radius `r` (`z`-axis
/// omitted when `two_d`).
fn sphere_offsets(r: isize, two_d: bool) -> (Vec<[isize; 3]>, f64) {
    let mut sphere = Vec::new();
    let zr = if two_d { 0 } else { r };
    for dz in -zr..=zr {
        for dy in -r..=r {
            for dx in -r..=r {
                if dz * dz + dy * dy + dx * dx <= r * r {
                    sphere.push([dz, dy, dx]);
                }
            }
        }
    }
    let w = 1.0 / sphere.len() as f64;
    (sphere, w)
}

/// Mean curvature-flow speed `N / |∇I|²` at a voxel, plus the first derivatives
/// `(dx, dy, dz)` (reused by the directional threshold).  Matches the covered
/// [`CurvatureFlowImageFilter`](super::CurvatureFlowImageFilter) discretization.
fn curvature_speed<F: Fn(isize, isize, isize) -> f64>(
    get: &F,
    z: isize,
    y: isize,
    x: isize,
    c: f64,
) -> (f64, f64, f64, f64) {
    let dx_ = (get(z, y, x + 1) - get(z, y, x - 1)) * 0.5;
    let dy_ = (get(z, y + 1, x) - get(z, y - 1, x)) * 0.5;
    let dz_ = (get(z + 1, y, x) - get(z - 1, y, x)) * 0.5;
    let dxx = get(z, y, x + 1) - 2.0 * c + get(z, y, x - 1);
    let dyy = get(z, y + 1, x) - 2.0 * c + get(z, y - 1, x);
    let dzz = get(z + 1, y, x) - 2.0 * c + get(z - 1, y, x);
    let dxy = (get(z, y + 1, x + 1) - get(z, y + 1, x - 1) - get(z, y - 1, x + 1)
        + get(z, y - 1, x - 1))
        * 0.25;
    let dxz = (get(z + 1, y, x + 1) - get(z + 1, y, x - 1) - get(z - 1, y, x + 1)
        + get(z - 1, y, x - 1))
        * 0.25;
    let dyz = (get(z + 1, y + 1, x) - get(z + 1, y - 1, x) - get(z - 1, y + 1, x)
        + get(z - 1, y - 1, x))
        * 0.25;
    let num = dxx * (dy_ * dy_ + dz_ * dz_)
        + dyy * (dx_ * dx_ + dz_ * dz_)
        + dzz * (dx_ * dx_ + dy_ * dy_)
        - 2.0 * dx_ * dy_ * dxy
        - 2.0 * dx_ * dz_ * dxz
        - 2.0 * dy_ * dz_ * dyz;
    let grad_sq = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
    let speed = if grad_sq > GRAD_MAG_EPSILON {
        num / grad_sq
    } else {
        0.0
    };
    (speed, dx_, dy_, dz_)
}

/// Configuration for [`BinaryMinMaxCurvatureFlowImageFilter`].
#[derive(Debug, Clone, Copy)]
pub struct BinaryMinMaxCurvatureFlowConfig {
    /// Number of evolution iterations (ITK default 5).
    pub num_iterations: usize,
    /// User time step; effective step is `time_step / (2·D)` (ITK default 0.05).
    pub time_step: f32,
    /// Stencil radius `R` (ITK default 2).
    pub stencil_radius: usize,
    /// Scalar gate threshold (ITK default 0.0).
    pub threshold: f64,
}

impl Default for BinaryMinMaxCurvatureFlowConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            time_step: 0.05,
            stencil_radius: 2,
            threshold: 0.0,
        }
    }
}

/// Binary min/max curvature flow (`itk::BinaryMinMaxCurvatureFlowImageFilter`).
///
/// Like [`MinMaxCurvatureFlowImageFilter`] but the speed gate compares the
/// `R`-sphere average to a fixed scalar `threshold`: `avg < threshold ⇒
/// min(v, 0)`, else `max(v, 0)` (note the direction is opposite to the
/// directional-threshold variant).  Float-exact to `sitk.BinaryMinMaxCurvatureFlow`.
#[derive(Debug, Clone, Copy)]
pub struct BinaryMinMaxCurvatureFlowImageFilter {
    /// Filter configuration.
    pub config: BinaryMinMaxCurvatureFlowConfig,
}

impl BinaryMinMaxCurvatureFlowImageFilter {
    /// Construct with the given configuration.
    pub fn new(config: BinaryMinMaxCurvatureFlowConfig) -> Self {
        Self { config }
    }

    /// Apply binary min/max curvature flow.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let slab = ny * nx;
        let r = self.config.stencil_radius as isize;
        let two_d = nz == 1;
        let d_dim = if two_d { 2.0 } else { 3.0 };
        let dt = (self.config.time_step as f64) / (2.0 * d_dim);
        let thr = self.config.threshold;
        let (sphere, sphere_w) = sphere_offsets(r, two_d);

        let mut cur: Vec<f64> = vals_vec.iter().map(|&v| v as f64).collect();
        for _ in 0..self.config.num_iterations {
            let prev = cur.clone();
            let get = |zz: isize, yy: isize, xx: isize| -> f64 {
                let zc = zz.clamp(0, nz as isize - 1) as usize;
                let yc = yy.clamp(0, ny as isize - 1) as usize;
                let xc = xx.clamp(0, nx as isize - 1) as usize;
                prev[zc * slab + yc * nx + xc]
            };
            cur = (0..n)
                .map(|flat| {
                    let iz = flat / slab;
                    let rem = flat - iz * slab;
                    let iy = rem / nx;
                    let ix = rem - iy * nx;
                    let (z, y, x) = (iz as isize, iy as isize, ix as isize);
                    let c = prev[flat];
                    let (mut speed, _, _, _) = curvature_speed(&get, z, y, x, c);
                    if speed != 0.0 {
                        let mut avg = 0.0;
                        for o in &sphere {
                            avg += sphere_w * get(z + o[0], y + o[1], x + o[2]);
                        }
                        speed = if avg < thr {
                            speed.min(0.0)
                        } else {
                            speed.max(0.0)
                        };
                    }
                    c + dt * speed
                })
                .collect();
        }
        Ok(rebuild(cur.iter().map(|&v| v as f32).collect(), dims, image))
    }
}

/// ITK `Math::Round` for non-negative arguments: round half up.
#[inline]
fn round_pos(v: f64) -> isize {
    (v + 0.5).floor() as isize
}

/// 2-D directional threshold (`Dispatch<2>`): average of the two neighbourhood
/// samples perpendicular to the gradient at distance `R`.
fn threshold_2d<F: Fn(isize, isize, isize) -> f64>(
    get: &F,
    z: isize,
    y: isize,
    x: isize,
    dx_: f64,
    dy_: f64,
    r: f64,
) -> f64 {
    let gm = (dx_ * dx_ + dy_ * dy_).sqrt();
    if gm == 0.0 {
        return 0.0;
    }
    let gm2 = gm / r;
    let gg0 = dx_ / gm2; // along x
    let gg1 = dy_ / gm2; // along y
    // Offsets are relative to the stencil centre R; subtract R to get image deltas.
    let p1y = round_pos(r + gg0) - r as isize;
    let p1x = round_pos(r - gg1) - r as isize;
    let p2y = round_pos(r - gg0) - r as isize;
    let p2x = round_pos(r + gg1) - r as isize;
    (get(z, y + p1y, x + p1x) + get(z, y + p2y, x + p2x)) * 0.5
}

/// 3-D directional threshold (`Dispatch<3>`): average of the four neighbourhood
/// samples perpendicular to the gradient at distance `R` (spherical angles).
#[allow(clippy::too_many_arguments)]
fn threshold_3d<F: Fn(isize, isize, isize) -> f64>(
    get: &F,
    z: isize,
    y: isize,
    x: isize,
    dx_: f64,
    dy_: f64,
    dz_: f64,
    r: f64,
) -> f64 {
    let gm = (dx_ * dx_ + dy_ * dy_ + dz_ * dz_).sqrt();
    if gm == 0.0 {
        return 0.0;
    }
    let gm2 = gm / r;
    let (gx, gy, mut gz) = (dx_ / gm2, dy_ / gm2, dz_ / gm2);
    // gradient is now length R; ITK normalizes by R again for the angle and
    // clamps the z-component (gradient[2]) to [-1, 1].
    gz = (gz / r).clamp(-1.0, 1.0);
    let theta = gz.acos();
    let phi = if gx == 0.0 {
        std::f64::consts::FRAC_PI_2
    } else {
        (gy / gx).atan()
    };
    let (ct, st) = (theta.cos(), theta.sin());
    let (cp, sp) = (phi.cos(), phi.sin());
    let r_st = r * st;
    let r_ctcp = r * ct * cp;
    let r_ctsp = r * ct * sp;
    let r_sp = r * sp;
    let r_cp = r * cp;
    let ri = r as isize;
    // position[0]=x, [1]=y, [2]=z; subtract R for image deltas.
    let pt = |px: f64, py: f64, pz: f64| -> f64 {
        let ox = round_pos(r + px) - ri;
        let oy = round_pos(r + py) - ri;
        let oz = round_pos(r + pz) - ri;
        get(z + oz, y + oy, x + ox)
    };
    let mut t = 0.0;
    t += pt(r_ctcp, r_ctsp, -r_st); // angle 0
    t += pt(-r_sp, r_cp, 0.0); // angle 90
    t += pt(-r_ctcp, -r_ctsp, r_st); // angle 180
    t += pt(r_sp, -r_cp, 0.0); // angle 270
    t * 0.25
}

#[cfg(test)]
#[path = "tests_min_max_curvature_flow.rs"]
mod tests;

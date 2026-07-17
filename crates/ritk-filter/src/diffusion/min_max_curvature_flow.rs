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
//! `= N / |∇I|²`), and the effective time step is `time_step / R²` where `R` is
//! the stencil radius — the ITK-internal CFL scaling that
//! `MinMaxCurvatureFlowImageFilter` applies, independent of image dimension
//! (recovered from sitk: `R=1 → /1`, `R=2 → /4`, `R=3 → /9`, in both 2-D and 3-D).
//!
//! # ITK / SimpleITK parity
//! Float-exact to `sitk.MinMaxCurvatureFlow` across stencil radii (1, 2, 3),
//! time steps, and iteration counts, in both 2-D (`z = 1`, `Dispatch<2>`) and
//! 3-D (`Dispatch<3>`).

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Denominator floor below which curvature is treated as zero (flat region).
const GRAD_MAG_EPSILON: f64 = 1e-9;

/// Configuration for [`MinMaxCurvatureFlowImageFilter`].
#[derive(Debug, Clone, Copy)]
pub struct MinMaxCurvatureFlowConfig {
    /// Number of evolution iterations (ITK default 5).
    pub num_iterations: usize,
    /// User time step; the effective step is `time_step / R²` (ITK default 0.05).
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
        let sp = image.spacing();
        let result =
            min_max_curvature_flow_evolve(&vals_vec, dims, [sp[0], sp[1], sp[2]], &self.config);
        Ok(rebuild(result, dims, image))
    }

    /// Coeus-native sister of [`MinMaxCurvatureFlowImageFilter::apply`].
    ///
    /// Runs the identical directional-threshold min/max curvature-flow evolution
    /// via the shared `min_max_curvature_flow_evolve` host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            min_max_curvature_flow_evolve(vals, dims, spacing, &self.config)
        })
    }
}

/// Substrate-agnostic host core for [`MinMaxCurvatureFlowImageFilter`]: the
/// directional-threshold min/max curvature-flow explicit-Euler evolution on a
/// flat z-major buffer (f64 double buffer, per-iteration f32 rounding to match
/// ITK's image-pixel-type accumulation). Single source of truth for the Burn
/// [`apply`](MinMaxCurvatureFlowImageFilter::apply) and Coeus-native
/// [`apply_native`](MinMaxCurvatureFlowImageFilter::apply_native) paths.
fn min_max_curvature_flow_evolve(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &MinMaxCurvatureFlowConfig,
) -> Vec<f32> {
    assert!(
        config.stencil_radius >= 1,
        "MinMaxCurvatureFlowImageFilter: stencil_radius must be >= 1, got {}",
        config.stencil_radius
    );
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let slab = ny * nx;
    let r = config.stencil_radius as isize;
    let two_d = nz == 1;
    let dt = (config.time_step as f64) / (r * r).max(1) as f64;

    let (sphere, sphere_w) = sphere_offsets(r, two_d);
    let rf = r as f64;

    let inv_sp = [1.0 / spacing[2], 1.0 / spacing[1], 1.0 / spacing[0]];

    let mut cur: Vec<f64> = vals.iter().map(|&v| v as f64).collect();

    // Reusable double buffer: each sweep reads `cur`, writes `next`, then
    // swaps — avoids the per-iteration `cur.clone()` + `collect()` (two
    // N-element f64 allocations per iteration). Bit-identical: reading `cur`
    // after the swap is exactly the previous sweep's `prev` clone.
    let mut next: Vec<f64> = vec![0.0f64; n];
    for _ in 0..config.num_iterations {
        {
            // Shared borrow of `cur` outlives the moirai scope; moirai
            // uses scoped threads so non-'static borrows are safe.
            let cur_ref: &[f64] = &cur;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut next,
                slab,
                |iz, iz_out| {
                    let get = |zz: isize, yy: isize, xx: isize| -> f64 {
                        let zc = zz.clamp(0, nz as isize - 1) as usize;
                        let yc = yy.clamp(0, ny as isize - 1) as usize;
                        let xc = xx.clamp(0, nx as isize - 1) as usize;
                        cur_ref[zc * slab + yc * nx + xc]
                    };
                    for rem in 0..slab {
                        let iy = rem / nx;
                        let ix = rem - iy * nx;
                        let (z, y, x) = (iz as isize, iy as isize, ix as isize);
                        let c = cur_ref[iz * slab + rem];
                        let (mut speed, dx_, dy_, dz_) = curvature_speed(&get, z, y, x, c, inv_sp);

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
                        // ITK accumulates in the image pixel type (f32); round each
                        // iteration to f32 so high-dynamic-range volumes match bit-exactly.
                        iz_out[rem] = (c + dt * speed) as f32 as f64;
                    }
                },
            );
        }
        std::mem::swap(&mut cur, &mut next);
    }

    cur.iter().map(|&v| v as f32).collect()
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
    inv_sp: [f64; 3],
) -> (f64, f64, f64, f64) {
    let dx_ = (get(z, y, x + 1) - get(z, y, x - 1)) * 0.5 * inv_sp[0];
    let dy_ = (get(z, y + 1, x) - get(z, y - 1, x)) * 0.5 * inv_sp[1];
    let dz_ = (get(z + 1, y, x) - get(z - 1, y, x)) * 0.5 * inv_sp[2];
    let dxx = (get(z, y, x + 1) - 2.0 * c + get(z, y, x - 1)) * inv_sp[0] * inv_sp[0];
    let dyy = (get(z, y + 1, x) - 2.0 * c + get(z, y - 1, x)) * inv_sp[1] * inv_sp[1];
    let dzz = (get(z + 1, y, x) - 2.0 * c + get(z - 1, y, x)) * inv_sp[2] * inv_sp[2];
    let dxy = (get(z, y + 1, x + 1) - get(z, y + 1, x - 1) - get(z, y - 1, x + 1)
        + get(z, y - 1, x - 1))
        * 0.25
        * inv_sp[0]
        * inv_sp[1];
    let dxz = (get(z + 1, y, x + 1) - get(z + 1, y, x - 1) - get(z - 1, y, x + 1)
        + get(z - 1, y, x - 1))
        * 0.25
        * inv_sp[0]
        * inv_sp[2];
    let dyz = (get(z + 1, y + 1, x) - get(z + 1, y - 1, x) - get(z - 1, y + 1, x)
        + get(z - 1, y - 1, x))
        * 0.25
        * inv_sp[1]
        * inv_sp[2];
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
        let sp = image.spacing();
        let result = binary_min_max_curvature_flow_evolve(
            &vals_vec,
            dims,
            [sp[0], sp[1], sp[2]],
            &self.config,
        );
        Ok(rebuild(result, dims, image))
    }

    /// Coeus-native sister of [`BinaryMinMaxCurvatureFlowImageFilter::apply`].
    ///
    /// Runs the identical scalar-threshold binary min/max curvature-flow
    /// evolution via the shared `binary_min_max_curvature_flow_evolve` host
    /// core on the image's contiguous host buffer, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            binary_min_max_curvature_flow_evolve(vals, dims, spacing, &self.config)
        })
    }
}

/// Substrate-agnostic host core for [`BinaryMinMaxCurvatureFlowImageFilter`]:
/// the scalar-threshold binary min/max curvature-flow explicit-Euler evolution
/// on a flat z-major buffer. Single source of truth for the Burn
/// [`apply`](BinaryMinMaxCurvatureFlowImageFilter::apply) and Coeus-native
/// [`apply_native`](BinaryMinMaxCurvatureFlowImageFilter::apply_native) paths.
fn binary_min_max_curvature_flow_evolve(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &BinaryMinMaxCurvatureFlowConfig,
) -> Vec<f32> {
    assert!(
        config.stencil_radius >= 1,
        "BinaryMinMaxCurvatureFlowImageFilter: stencil_radius must be >= 1, got {}",
        config.stencil_radius
    );
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let slab = ny * nx;
    let r = config.stencil_radius as isize;
    let two_d = nz == 1;
    let dt = (config.time_step as f64) / (r * r).max(1) as f64;
    let thr = config.threshold;
    let (sphere, sphere_w) = sphere_offsets(r, two_d);

    let inv_sp = [1.0 / spacing[2], 1.0 / spacing[1], 1.0 / spacing[0]];

    let mut cur: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    // Reusable double buffer (see the min/max path above): read `cur`, write
    // `next`, swap — drops the per-iteration clone + collect allocations.
    let mut next: Vec<f64> = vec![0.0f64; n];
    for _ in 0..config.num_iterations {
        {
            let cur_ref: &[f64] = &cur;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut next,
                slab,
                |iz, iz_out| {
                    let get = |zz: isize, yy: isize, xx: isize| -> f64 {
                        let zc = zz.clamp(0, nz as isize - 1) as usize;
                        let yc = yy.clamp(0, ny as isize - 1) as usize;
                        let xc = xx.clamp(0, nx as isize - 1) as usize;
                        cur_ref[zc * slab + yc * nx + xc]
                    };
                    for rem in 0..slab {
                        let iy = rem / nx;
                        let ix = rem - iy * nx;
                        let (z, y, x) = (iz as isize, iy as isize, ix as isize);
                        let c = cur_ref[iz * slab + rem];
                        let (mut speed, _, _, _) = curvature_speed(&get, z, y, x, c, inv_sp);
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
                        // ITK accumulates in the image pixel type (f32); round each
                        // iteration to f32 so high-dynamic-range volumes match bit-exactly.
                        iz_out[rem] = (c + dt * speed) as f32 as f64;
                    }
                },
            );
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur.iter().map(|&v| v as f32).collect()
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
    // gradient is now length R; ITK clamps the z-component (gradient[2]) to [-1, 1] without dividing by R (ITK bug)
    gz = gz.clamp(-1.0, 1.0);
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

#[cfg(test)]
mod tests_native {
    use super::{
        BinaryMinMaxCurvatureFlowConfig, BinaryMinMaxCurvatureFlowImageFilter,
        MinMaxCurvatureFlowConfig, MinMaxCurvatureFlowImageFilter,
    };
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    #[test]
    fn min_max_matches_burn() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 13) % 19) as f32).collect();
        assert_native_matches_burn(
            vals,
            [3, 4, 5],
            |img| {
                MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig::default())
                    .apply(img)
                    .expect("burn min/max curvature flow")
            },
            |img, backend| {
                MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig::default())
                    .apply_native(img, backend)
            },
        );
    }

    #[test]
    fn binary_min_max_matches_burn() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 5) % 7) as f32).collect();
        assert_native_matches_burn(
            vals,
            [3, 4, 5],
            |img| {
                BinaryMinMaxCurvatureFlowImageFilter::new(BinaryMinMaxCurvatureFlowConfig::default())
                    .apply(img)
                    .expect("burn binary min/max curvature flow")
            },
            |img, backend| {
                BinaryMinMaxCurvatureFlowImageFilter::new(BinaryMinMaxCurvatureFlowConfig::default())
                    .apply_native(img, backend)
            },
        );
    }

    #[test]
    fn oracle_constant_field_preserved() {
        // Both variants gate the curvature speed to 0 on a constant field
        // (all derivatives 0), so the field is a fixed point of the evolution.
        let img = make_native_image(vec![4.0f32; 64], [4, 4, 4]);
        let out = MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig::default())
            .apply_native(&img, &SequentialBackend)
            .expect("native min/max curvature flow");
        for &v in &native_vals(&out) {
            assert_eq!(v, 4.0, "constant field must be a fixed point, got {v}");
        }
    }
}

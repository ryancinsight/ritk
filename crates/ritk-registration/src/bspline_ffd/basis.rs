//! Cubic B-spline basis functions and control grid evaluation.
//!
//! Implements the Rueckert (1999) uniform cubic B-spline basis:
//!
//! ```text
//! β₃₀(t) = (1 − t)³ / 6
//! β₃₁(t) = (3t³ − 6t² + 4) / 6
//! β₃₂(t) = (−3t³ + 3t² + 3t + 1) / 6
//! β₃₃(t) = t³ / 6
//! ```
//!
//! ## Optimization (Sprint 308)
//!
//! The original per-voxel `cubic_bspline_1d` calls were redundant:
//! for a 256³ volume there are only 256 unique t values per axis, not 16.7M.
//! `BasisCache` pre-computes the 4 basis values + control-point indices
//! (`k`, `[b0,b1,b2,b3]`) once per axis coordinate, converting the hot path
//! from compute to lookup.
//!
//! Additionally, >90% of voxels are "interior" — their 64 control-point
//! neighbourhood is fully in-bounds. The fast path detects interior ranges
//! and skips all bounds checks, eliminating ~1B branch instructions for a
//! 256³ volume.

use crate::deformable_field_ops::{flat, VelocityField};

/// Evaluate the four cubic B-spline basis values at parameter `t ∈ [0, 1]`.
///
/// Returns `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`. These sum to 1.0 (partition
/// of unity) and are non-negative on `[0, 1]`.
#[inline]
pub(super) fn cubic_bspline_1d(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt3 = omt * omt * omt;

    [
        omt3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

// ── Pre-computed basis cache ────────────────────────────────────────────

/// Pre-computed B-spline basis data for one axis.
///
/// For each image coordinate `i ∈ [0, dim)`, stores:
/// - `k`: the first control-point index (kz, ky, or kx)
/// - `b`: the four cubic basis values `[β₃₀(t), β₃₁(t), β₃₂(t), β₃₃(t)]`
///
/// This eliminates per-voxel `cubic_bspline_1d` calls (hot path is
/// lookup-only) and hoists `k` computation out of the inner loop.
#[derive(Clone)]
pub struct AxisBasis {
    /// `k[i]` = first control-point index for image coordinate i.
    pub k: Vec<isize>,
    /// `b[i]` = `[β₃₀(t_i), β₃₁(t_i), β₃₂(t_i), β₃₃(t_i)]`.
    pub b: Vec<[f64; 4]>,
}

impl AxisBasis {
    /// Pre-compute basis data for `dim` coordinates with the given control spacing.
    pub fn new(dim: usize, ctrl_spacing: f64) -> Self {
        let mut k = Vec::with_capacity(dim);
        let mut b = Vec::with_capacity(dim);
        for i in 0..dim {
            let u = i as f64 / ctrl_spacing + 1.0;
            let ki = u.floor() as isize - 1;
            let t = u - (ki + 1) as f64;
            k.push(ki);
            b.push(cubic_bspline_1d(t));
        }
        Self { k, b }
    }
}

/// Pre-computed B-spline basis data for all three axes.
///
/// ### Performance
///
/// Building the cache is O(nz + ny + nx) — negligible (~0.01 ms for a
/// 256³ volume). Each `evaluate_bspline_displacement_fast` call saves
/// ~50M `cubic_bspline_1d` evaluations.
#[derive(Clone)]
pub struct BasisCache {
    pub z: AxisBasis,
    pub y: AxisBasis,
    pub x: AxisBasis,
}

impl BasisCache {
    /// Build the basis cache for the given image dimensions and control spacing.
    pub fn new(dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> Self {
        Self {
            z: AxisBasis::new(dims[0], ctrl_spacing[0]),
            y: AxisBasis::new(dims[1], ctrl_spacing[1]),
            x: AxisBasis::new(dims[2], ctrl_spacing[2]),
        }
    }

    /// Return interior z-range `[z_lo, z_hi)` where all 4 control points
    /// along z are in-bounds (i.e. `kz ∈ [0, cnz-4]`).
    pub(super) fn interior_z_range(&self, cnz: usize) -> (usize, usize) {
        let lo = self.z.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .z
            .k
            .iter()
            .rposition(|&k| k <= cnz as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }

    /// Return interior y-range `[y_lo, y_hi)` where all 4 control points
    /// along y are in-bounds.
    pub(super) fn interior_y_range(&self, cny: usize) -> (usize, usize) {
        let lo = self.y.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .y
            .k
            .iter()
            .rposition(|&k| k <= cny as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }

    /// Return interior x-range `[x_lo, x_hi)` where all 4 control points
    /// along x are in-bounds.
    pub(super) fn interior_x_range(&self, cnx: usize) -> (usize, usize) {
        let lo = self.x.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .x
            .k
            .iter()
            .rposition(|&k| k <= cnx as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }
}

/// Compute control-grid dimensions from image dimensions and control spacing.
///
/// The control lattice extends one extra control point beyond each boundary
/// to ensure full support coverage. Grid dimension along axis `d`:
///
/// ```text
/// n_ctrl[d] = ceil(dims[d] / spacing[d]) + 3
/// ```
///
/// The `+3` accounts for one point before the domain origin and two points
/// after the far boundary, providing the four-point support stencil at every
/// image voxel.
pub fn init_control_grid(dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> [usize; 3] {
    let mut ctrl_dims = [0usize; 3];
    for d in 0..3 {
        ctrl_dims[d] = (dims[d] as f64 / ctrl_spacing[d]).ceil() as usize + 3;
    }
    ctrl_dims
}

/// Evaluate the dense displacement field from B-spline control points.
///
/// For each image voxel `(iz, iy, ix)`, computes the displacement as the
/// tensor-product of 1D cubic B-spline bases evaluated over the 4×4×4
/// neighborhood of control points.
///
/// # Returns
/// `VelocityField` — displacement components in voxel units, each of length
/// `dims[0] * dims[1] * dims[2]`.
pub(super) fn evaluate_bspline_displacement(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: [usize; 3],
) -> VelocityField {
    let cache = BasisCache::new(dims, ctrl_spacing);
    evaluate_bspline_displacement_fast(cp_z, cp_y, cp_x, ctrl_dims, dims, &cache)
}

/// Evaluate the dense displacement field using a pre-computed [`BasisCache`].
///
/// This is the optimized path:
/// - Basis values and control-point indices are looked up from the cache
///   (no per-voxel `cubic_bspline_1d` calls).
/// - Interior voxels (where all 64 control points are in-bounds) skip all
///   bounds checks — ~1B branches saved for a 256³ volume.
/// - The inner 4×4×4 tensor-product loop is structured for auto-vectorization
///   (consecutive memory accesses, loop-invariant weights).
///
/// Allocates three `Vec<f32>` output buffers. Use
/// [`evaluate_bspline_displacement_fast_into`] to write into caller-owned
/// buffers without allocation.
///
/// # Returns
/// `VelocityField` — displacement components in voxel units, each of length
/// `dims[0] * dims[1] * dims[2]`.
pub fn evaluate_bspline_displacement_fast(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    dims: [usize; 3],
    cache: &BasisCache,
) -> VelocityField {
    let n = dims[0] * dims[1] * dims[2];
    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    evaluate_bspline_displacement_fast_into(
        cp_z, cp_y, cp_x, ctrl_dims, dims, cache, &mut dz, &mut dy, &mut dx,
    );
    VelocityField {
        z: dz,
        y: dy,
        x: dx,
    }
}

/// Zero-allocation variant of [`evaluate_bspline_displacement_fast`].
///
/// Writes displacement components directly into caller-provided buffers,
/// avoiding the three `Vec<f32>` allocations of the allocating version.
/// Buffers are zeroed on entry; any prior contents are overwritten.
///
/// # Panics
/// Panics if `dz`, `dy`, or `dx` are shorter than
/// `dims[0] * dims[1] * dims[2]`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_bspline_displacement_fast_into(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    dims: [usize; 3],
    cache: &BasisCache,
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [cnz, cny, cnx] = *ctrl_dims;

    // Zero-fill output buffers before accumulation.
    dz[..n].fill(0.0);
    dy[..n].fill(0.0);
    dx[..n].fill(0.0);

    // Determine interior ranges — voxels where ALL 64 control points are
    // in-bounds. For voxels outside these ranges, the per-voxel bounds-check
    // path is still used.
    let (iz_lo, iz_hi) = cache.interior_z_range(cnz);
    let (iy_lo, iy_hi) = cache.interior_y_range(cny);
    let (ix_lo, ix_hi) = cache.interior_x_range(cnx);

    // ── Bounds-check path (boundary voxels) ──────────────────────────
    #[inline]
    fn eval_bounds(
        dz: &mut [f32],
        dy: &mut [f32],
        dx: &mut [f32],
        iz: usize,
        iy: usize,
        ix: usize,
        kz: isize,
        ky: isize,
        kx: isize,
        bz: &[f64; 4],
        by: &[f64; 4],
        bx: &[f64; 4],
        cp_z: &[f32],
        cp_y: &[f32],
        cp_x: &[f32],
        cnz: usize,
        cny: usize,
        cnx: usize,
        ny: usize,
        nx: usize,
    ) {
        let fi = flat(iz, iy, ix, ny, nx);
        let mut sum_z = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_x = 0.0_f64;
        for az in 0..4isize {
            let ciz = kz + az;
            if ciz < 0 || ciz >= cnz as isize {
                continue;
            }
            let wz = bz[az as usize];
            for ay in 0..4isize {
                let ciy = ky + ay;
                if ciy < 0 || ciy >= cny as isize {
                    continue;
                }
                let wzy = wz * by[ay as usize];
                for ax in 0..4isize {
                    let cix = kx + ax;
                    if cix < 0 || cix >= cnx as isize {
                        continue;
                    }
                    let w = wzy * bx[ax as usize];
                    let ci = flat(ciz as usize, ciy as usize, cix as usize, cny, cnx);
                    sum_z += w * cp_z[ci] as f64;
                    sum_y += w * cp_y[ci] as f64;
                    sum_x += w * cp_x[ci] as f64;
                }
            }
        }
        dz[fi] = sum_z as f32;
        dy[fi] = sum_y as f32;
        dx[fi] = sum_x as f32;
    }

    // ── Interior fast path (no bounds checks) ────────────────────────
    #[inline]
    fn eval_interior(
        dz: &mut [f32],
        dy: &mut [f32],
        dx: &mut [f32],
        iz: usize,
        iy: usize,
        ix: usize,
        kz: isize,
        ky: isize,
        kx: isize,
        bz: &[f64; 4],
        by: &[f64; 4],
        bx: &[f64; 4],
        cp_z: &[f32],
        cp_y: &[f32],
        cp_x: &[f32],
        cny: usize,
        cnx: usize,
        ny: usize,
        nx: usize,
    ) {
        let fi = flat(iz, iy, ix, ny, nx);
        let mut sum_z = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_x = 0.0_f64;

        // Unrolled 4×4×4 tensor product — all control points guaranteed
        // in-bounds. Structured so the compiler can auto-vectorize the
        // 4-wide x-axis accumulation.
        #[allow(clippy::needless_range_loop)]
        for az in 0..4usize {
            let ciz = (kz + az as isize) as usize;
            let wz = bz[az];
            #[allow(clippy::needless_range_loop)]
            for ay in 0..4usize {
                let ciy = (ky + ay as isize) as usize;
                let wzy = wz * by[ay];
                let ci_base = flat(ciz, ciy, 0, cny, cnx);
                let w0 = wzy * bx[0];
                let w1 = wzy * bx[1];
                let w2 = wzy * bx[2];
                let w3 = wzy * bx[3];

                let kx_usize = kx as usize;
                let c0 = ci_base + kx_usize;
                let c1 = ci_base + kx_usize + 1;
                let c2 = ci_base + kx_usize + 2;
                let c3 = ci_base + kx_usize + 3;

                sum_z += w0 * cp_z[c0] as f64
                    + w1 * cp_z[c1] as f64
                    + w2 * cp_z[c2] as f64
                    + w3 * cp_z[c3] as f64;
                sum_y += w0 * cp_y[c0] as f64
                    + w1 * cp_y[c1] as f64
                    + w2 * cp_y[c2] as f64
                    + w3 * cp_y[c3] as f64;
                sum_x += w0 * cp_x[c0] as f64
                    + w1 * cp_x[c1] as f64
                    + w2 * cp_x[c2] as f64
                    + w3 * cp_x[c3] as f64;
            }
        }

        dz[fi] = sum_z as f32;
        dy[fi] = sum_y as f32;
        dx[fi] = sum_x as f32;
    }

    // ── Main evaluation loop ──────────────────────────────────────────
    for iz in 0..nz {
        let kz = cache.z.k[iz];
        let bz = &cache.z.b[iz];

        for iy in 0..ny {
            let ky = cache.y.k[iy];
            let by = &cache.y.b[iy];

            // Interior x-range: all 64 control points in-bounds → fast path.
            if iz >= iz_lo && iz < iz_hi && iy >= iy_lo && iy < iy_hi {
                for ix in ix_lo..ix_hi {
                    let kx = cache.x.k[ix];
                    let bx = &cache.x.b[ix];
                    eval_interior(
                        &mut *dz, &mut *dy, &mut *dx, iz, iy, ix, kz, ky, kx, bz, by, bx, cp_z,
                        cp_y, cp_x, cny, cnx, ny, nx,
                    );
                }
                // Boundary x-ranges: use bounds-check path.
                for ix in 0..ix_lo {
                    let kx = cache.x.k[ix];
                    let bx = &cache.x.b[ix];
                    eval_bounds(
                        &mut *dz, &mut *dy, &mut *dx, iz, iy, ix, kz, ky, kx, bz, by, bx, cp_z,
                        cp_y, cp_x, cnz, cny, cnx, ny, nx,
                    );
                }
                for ix in ix_hi..nx {
                    let kx = cache.x.k[ix];
                    let bx = &cache.x.b[ix];
                    eval_bounds(
                        &mut *dz, &mut *dy, &mut *dx, iz, iy, ix, kz, ky, kx, bz, by, bx, cp_z,
                        cp_y, cp_x, cnz, cny, cnx, ny, nx,
                    );
                }
            } else {
                // Boundary z/y ranges: always use bounds-check path.
                for ix in 0..nx {
                    let kx = cache.x.k[ix];
                    let bx = &cache.x.b[ix];
                    eval_bounds(
                        &mut *dz, &mut *dy, &mut *dx, iz, iy, ix, kz, ky, kx, bz, by, bx, cp_z,
                        cp_y, cp_x, cnz, cny, cnx, ny, nx,
                    );
                }
            }
        }
    }
}

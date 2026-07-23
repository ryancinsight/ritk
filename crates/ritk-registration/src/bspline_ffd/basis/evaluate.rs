//! B-spline displacement field evaluation from control-point grids.
//!
//! Implements two evaluation paths:
//! - The cache-based sparse path ([`evaluate_bspline_displacement_fast_into`])
//!   uses pre-computed per-axis basis tables; interior voxels skip all bounds
//!   checks via [`BasisCache`].
//! - The bounded dense support-matrix path
//!   ([`evaluate_bspline_displacement_dense_into`]) replaces the per-axis
//!   basis cache with a dense support table where control-point indices are
//!   clamped to in-bounds positions *in advance*. The inner 4³ FMA is
//!   branch-free: OOB cells contribute weight 0 via a precomputed `mask`,
//!   so the compiler auto-vectorises the floating-point loop without
//!   conditional hops per voxel.
//!
//! Selection between paths is done by [`should_use_dense_path`], dispatched
//! in `BSplineFFDRegistration::register`'s inner loop. The dense path is
//! preferred for small control lattices (`ctrl_dims.product() <=
//! DENSE_LATTICE_CUTOFF`); the cache path wins on bandwidth and arithmetic
//! reuse for larger ones.
//!
//! Both paths produce numerically identical outputs (verified by
//! `bspline_dense_matches_sparse_on_small_lattice` in `tests/basis.rs`).

use super::super::volume_dims::VolumeDims;
use super::cache::BasisCache;
use super::scalar::cubic_bspline_basis;
use crate::deformable_field_ops::{flat, VelocityField};

/// Upper bound on the control-lattice product below which the bounded
/// dense support-matrix path is advertised as preferable.
///
/// Empirical cut-over: with `dims=(64,64,64)` and `ctrl_spacing=(8,8,8)`
/// (`ctrl_n = 11³ = 1331`), the dense path is competitive with the cache
/// path; above this size the cache-led path wins on bandwidth.
///
/// Tune per-machine once `criterion`'s bspline_displacement bench is in
/// CI; the value was chosen conservatively below 1M to keep allocation
/// cost compatible with the L1-resident claim.
pub const DENSE_LATTICE_CUTOFF: usize = 1_000_000;

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
pub fn init_control_grid(dims: VolumeDims, ctrl_spacing: &[f64; 3]) -> [usize; 3] {
    let d = dims.as_array();
    let mut ctrl_dims = [0usize; 3];
    for axis in 0..3 {
        ctrl_dims[axis] = (d[axis] as f64 / ctrl_spacing[axis]).ceil() as usize + 3;
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
pub fn evaluate_bspline_displacement(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: VolumeDims,
) -> VelocityField {
    let cache = BasisCache::new(dims, ctrl_spacing);
    evaluate_bspline_displacement_fast(cp_z, cp_y, cp_x, ctrl_dims, dims.as_array(), &cache)
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
        cp_z,
        cp_y,
        cp_x,
        ctrl_dims,
        VolumeDims(dims),
        cache,
        &mut dz,
        &mut dy,
        &mut dx,
    );
    VelocityField {
        z: dz,
        y: dy,
        x: dx,
    }
}

/// True iff the dense support-matrix path should be selected for the
/// given control lattice. Pure dispatch predicate: no allocation, no
/// arithmetic beyond one `usize` product.
///
/// # Cost model
///
/// The dense path's resident memory is `O(nz + ny + nx)` (six per-axis
/// `Vec<[u32; 4]>`-or-similar vectors, indexed by axis coordinate). The
/// per-call inner-loop work is `O(dims.product() * 64)` branch-free FMA.
/// The binding constraint on qualification is the control-lattice product
/// (`ctrl_n <= DENSE_LATTICE_CUTOFF`), chosen empirically as the cutover
/// where the cache-hit avoidance of the dense inner loop outweighs the
/// cache-bandwidth reuse of the sparse interior path.
#[inline]
pub fn should_use_dense_path(ctrl_dims: &[usize; 3]) -> bool {
    let ctrl_n = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    ctrl_n <= DENSE_LATTICE_CUTOFF
}

// ──────────────────────────────────────────────────────────────────────────────
// Dense path implementation
// ──────────────────────────────────────────────────────────────────────────────
//
// For each axis coordinate, build:
//   - idx[axis][i]: usize = clamped control-point linear index (`0..ctrl_n`).
//     OOB cells clamp to 0 (or any in-bounds value — masked weight = 0).
//   - w[axis][i]:   [f64; 4] cubic-basis weights (unchanged).
//   - mask[axis][i]: [f64; 4] = 0.0 if any of the 4 cells is OOB, else 1.0.
//
// The dense inner loop is a fully branch-free 64-cell FMA. The OOB sentinel
// path is replaced by a multiplicative mask: cells with `mask[k] = 0.0`
// contribute weight=0 to the sum, and `idx[k].clamp(0, ctrl_n-1)` guarantees
// every `cp_z[ci]` access is in-bounds (no runtime OOB on release).

/// Bounded dense support-matrix variant of
/// [`evaluate_bspline_displacement_fast_into`].
///
/// Convenience wrapper that allocates one `DenseSupport` table per call
/// (resident cost `O(nz+ny+nx)`), builds it, and runs the branch-free 4³
/// FMA. Callers that call this on a hot path (e.g. every iteration of a
/// registration) should instead build a `DenseSupport` once at the level
/// boundary and call [`evaluate_bspline_displacement_dense_with`]. Both
/// paths produce numerically identical outputs to within `f32`
/// summation-order tolerance on a qualifying lattice — see
/// `bspline_dense_matches_sparse_on_small_lattice`.
///
/// # Panics
/// Panics if `dz`, `dy`, or `dx` are shorter than
/// `dims[0] * dims[1] * dims[2]`.
pub fn evaluate_bspline_displacement_dense_into(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: VolumeDims,
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    let support = DenseSupport::build(dims, *ctrl_dims, ctrl_spacing);
    support.evaluate(cp_z, cp_y, cp_x, ctrl_dims, dz, dy, dx);
}

/// Bounded dense support-matrix variant that reuses a pre-built
/// [`DenseSupport`] table.
///
/// The correct fast path for a registration loop: callers hoist
/// `DenseSupport::build` to the level boundary (where `dims` /
/// `ctrl_dims` / `ctrl_spacing` are constant) and call this once per
/// iteration. Skips the per-call allocation in
/// [`evaluate_bspline_displacement_dense_into`].
///
/// # Panics
/// Panics if `dz`, `dy`, or `dx` are shorter than
/// `dims[0] * dims[1] * dims[2]`.
#[inline]
pub fn evaluate_bspline_displacement_dense_with(
    support: &DenseSupport,
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    support.evaluate(cp_z, cp_y, cp_x, ctrl_dims, dz, dy, dx);
}

/// Dense support-matrix cache built once per lattice; reused across the
/// 4³ FMA inner loop. Same per-axis tables as the bare `evaluate_*_into`
/// path, but exposed for callers that amortise across many warps
/// (e.g. the registration inner loop calling this once per B-spline
/// warp). The tables are small (`O(nz + ny + nx)`); resident cost for
/// the documented 64³ case is ~9 KiB.
#[derive(Clone, Debug)]
pub struct DenseSupport {
    z_idx: Vec<[u32; 4]>,
    z_w: Vec<[f64; 4]>,
    z_mask: Vec<[f64; 4]>,
    y_idx: Vec<[u32; 4]>,
    y_w: Vec<[f64; 4]>,
    y_mask: Vec<[f64; 4]>,
    x_idx: Vec<[u32; 4]>,
    x_w: Vec<[f64; 4]>,
    x_mask: Vec<[f64; 4]>,
}

impl DenseSupport {
    /// Build the dense support table for `dims`/`ctrl_dims`/`ctrl_spacing`.
    /// The dispatch predicate guarantees `ctrl_n <= DENSE_LATTICE_CUTOFF`,
    /// so `u32` indices never overflow.
    pub fn build(dims: VolumeDims, ctrl_dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> Self {
        let [nz, ny, nx] = dims.as_array();
        let [cnz, cny, cnx] = ctrl_dims;

        let z_idx = build_axis_idx_table(nz, ctrl_spacing[0], cnz);
        let y_idx = build_axis_idx_table(ny, ctrl_spacing[1], cny);
        let x_idx = build_axis_idx_table(nx, ctrl_spacing[2], cnx);
        let z_w = build_axis_w_table(nz, ctrl_spacing[0]);
        let y_w = build_axis_w_table(ny, ctrl_spacing[1]);
        let x_w = build_axis_w_table(nx, ctrl_spacing[2]);
        let z_mask = build_axis_mask_table(nz, ctrl_spacing[0], cnz);
        let y_mask = build_axis_mask_table(ny, ctrl_spacing[1], cny);
        let x_mask = build_axis_mask_table(nx, ctrl_spacing[2], cnx);

        Self {
            z_idx,
            z_w,
            z_mask,
            y_idx,
            y_w,
            y_mask,
            x_idx,
            x_w,
            x_mask,
        }
    }

    /// Branch-free 4³ FMA inner loop using the cached support tables.
    /// `cp_z`/`cp_y`/`cp_x` and `ctrl_dims` are caller-owned and unchanged.
    /// `dz`/`dy`/`dx` are caller-owned and zeroed on entry.
    ///
    /// # Panics
    /// Panics if any output buffer is shorter than `dims.product()`.
    #[inline]
    pub fn evaluate(
        &self,
        cp_z: &[f32],
        cp_y: &[f32],
        cp_x: &[f32],
        ctrl_dims: &[usize; 3],
        dz: &mut [f32],
        dy: &mut [f32],
        dx: &mut [f32],
    ) {
        let [cnz, cny, cnx] = *ctrl_dims;
        let ctrl_n = cnz * cny * cnx;
        let cny_cnx = cny * cnx;
        let nz = self.z_idx.len();
        let ny = self.y_idx.len();
        let nx = self.x_idx.len();
        let n = nz * ny * nx;

        dz[..n].fill(0.0);
        dy[..n].fill(0.0);
        dx[..n].fill(0.0);

        // The dispatch predicate guarantees `ctrl_n <= DENSE_LATTICE_CUTOFF`
        // (≤ 1_000_000), so linear indices fit comfortably in `u32`.
        // Pre-computed masks ensure OOB cells contribute weight=0 without
        // branches, so the inner FMA is branch-free and SIMD-friendly.

        for iz in 0..nz {
            let z_idx_row = &self.z_idx[iz];
            let z_w_row = &self.z_w[iz];
            let z_mask_row = &self.z_mask[iz];
            for iy in 0..ny {
                let y_idx_row = &self.y_idx[iy];
                let y_w_row = &self.y_w[iy];
                let y_mask_row = &self.y_mask[iy];
                for ix in 0..nx {
                    let x_idx_row = &self.x_idx[ix];
                    let x_w_row = &self.x_w[ix];
                    let x_mask_row = &self.x_mask[ix];
                    let fi = flat(iz, iy, ix, ny, nx);

                    let mut sum_z = 0.0_f64;
                    let mut sum_y = 0.0_f64;
                    let mut sum_x = 0.0_f64;

                    for az in 0..4usize {
                        let row_base = (z_idx_row[az] as usize) * cny_cnx;
                        let wz = z_w_row[az];
                        let mz = z_mask_row[az];
                        for ay in 0..4usize {
                            let slice_base = row_base + (y_idx_row[ay] as usize) * cnx;
                            let wzy = wz * y_w_row[ay];
                            let mzy = mz * y_mask_row[ay];
                            for ax in 0..4usize {
                                let ci = slice_base + x_idx_row[ax] as usize;
                                let w = wzy * x_w_row[ax];
                                // OOB-safe: x_idx clamped to cnx-1 (max valid
                                // index) keeps `ci < ctrl_n`; multiplicative
                                // mask zeroes OOB cells without a branch.
                                debug_assert!(ci < ctrl_n, "dense support OOB");
                                sum_z += mzy * x_mask_row[ax] * w * cp_z[ci] as f64;
                                sum_y += mzy * x_mask_row[ax] * w * cp_y[ci] as f64;
                                sum_x += mzy * x_mask_row[ax] * w * cp_x[ci] as f64;
                            }
                        }
                    }

                    dz[fi] = sum_z as f32;
                    dy[fi] = sum_y as f32;
                    dx[fi] = sum_x as f32;
                }
            }
        }
    }
}

// ── DenseSupport private builders ────────────────────────────────────────────

/// Per-axis clamped control-point indices (`0..ctrl_axis`) and basis
/// weights `[f64; 4]`. Cells with `cidx < 0 || cidx >= ctrl_axis` clamp
/// the index to `0` and a paired mask is zeroed in
/// [`build_axis_mask_table`].
#[inline]
fn build_axis_idx_table(dim: usize, ctrl_spacing: f64, ctrl_axis: usize) -> Vec<[u32; 4]> {
    let mut table = vec![[0_u32; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        for (k, slot) in row.iter_mut().enumerate() {
            let cidx = ki + k as isize;
            // Clamp OOB indices to 0 (a safe valid index; the paired mask
            // zeroes the contribution so clamping has no numerical effect).
            *slot = if cidx < 0 || cidx >= ctrl_axis as isize {
                0
            } else {
                cidx as u32
            };
        }
    }
    table
}

/// Per-axis cubic B-spline basis weights `[f64; 4]` at every `i`.
/// Caller is responsible for OOB masking via the corresponding
/// `build_axis_mask_table` (this routine fills all four entries verbatim).
#[inline]
fn build_axis_w_table(dim: usize, ctrl_spacing: f64) -> Vec<[f64; 4]> {
    let mut table = vec![[0_f64; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        let t = u - (ki + 1) as f64;
        // All 4 weights are copied verbatim — the `mask` table zeroes OOB cells.
        *row = cubic_bspline_basis(t);
    }
    table
}

/// Per-axis multiplicative mask `[f64; 4]` (1.0 in range, 0.0 OOB).
#[inline]
fn build_axis_mask_table(dim: usize, ctrl_spacing: f64, ctrl_axis: usize) -> Vec<[f64; 4]> {
    let mut table = vec![[0_f64; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        for (k, slot) in row.iter_mut().enumerate() {
            let cidx = ki + k as isize;
            *slot = if cidx < 0 || cidx >= ctrl_axis as isize {
                0.0
            } else {
                1.0
            };
        }
    }
    table
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
    dims: VolumeDims,
    cache: &BasisCache,
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    let [nz, ny, nx] = dims.as_array();
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

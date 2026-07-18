//! B-spline displacement field evaluation from control-point grids.
//!
//! Implements the fast evaluation path using pre-computed [`BasisCache`]:
//! - Basis values and control-point indices are looked up (not recomputed).
//! - Interior voxels skip all bounds checks.
//! - The 4Ã—4Ã—4 tensor-product loop is structured for auto-vectorization.

use super::super::volume_dims::VolumeDims;
use super::cache::BasisCache;
use crate::deformable_field_ops::{flat, VelocityField};

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
/// tensor-product of 1D cubic B-spline bases evaluated over the 4Ã—4Ã—4
/// neighborhood of control points.
///
/// # Returns
/// `VelocityField` â€” displacement components in voxel units, each of length
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
///   bounds checks â€” ~1B branches saved for a 256Â³ volume.
/// - The inner 4Ã—4Ã—4 tensor-product loop is structured for auto-vectorization
///   (consecutive memory accesses, loop-invariant weights).
///
/// Allocates three `Vec<f32>` output buffers. Use
/// [`evaluate_bspline_displacement_fast_into`] to write into caller-owned
/// buffers without allocation.
///
/// # Returns
/// `VelocityField` â€” displacement components in voxel units, each of length
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
        x: dx }
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

    // Determine interior ranges â€” voxels where ALL 64 control points are
    // in-bounds. For voxels outside these ranges, the per-voxel bounds-check
    // path is still used.
    let (iz_lo, iz_hi) = cache.interior_z_range(cnz);
    let (iy_lo, iy_hi) = cache.interior_y_range(cny);
    let (ix_lo, ix_hi) = cache.interior_x_range(cnx);

    // â”€â”€ Bounds-check path (boundary voxels) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    // â”€â”€ Interior fast path (no bounds checks) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        // Unrolled 4Ã—4Ã—4 tensor product â€” all control points guaranteed
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

    // â”€â”€ Main evaluation loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for iz in 0..nz {
        let kz = cache.z.k[iz];
        let bz = &cache.z.b[iz];

        for iy in 0..ny {
            let ky = cache.y.k[iy];
            let by = &cache.y.b[iy];

            // Interior x-range: all 64 control points in-bounds â†’ fast path.
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

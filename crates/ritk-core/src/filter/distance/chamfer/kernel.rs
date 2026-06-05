//! Low-level chamfer distance transform kernel: two raster scans over a
//! 3×3×3 mask.

/// Chamfer distance metric.
///
/// `Chessboard` matches the L∞ norm; `Taxicab` matches the L1 norm. Both
/// produce an integer-valued distance map.
///
/// Note: this kernel implements the **interior distance transform** of
/// `scipy.ndimage.distance_transform_cdt`: each foreground voxel is
/// assigned the chamfer distance to the nearest background voxel;
/// background voxels receive 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChamferMetric {
    /// L∞ norm: distance is the maximum absolute voxel offset along any axis.
    /// The default — matches `scipy.ndimage.distance_transform_cdt` default.
    #[default]
    Chessboard,
    /// L1 norm: distance is the sum of absolute voxel offsets along all axes.
    Taxicab,
}

/// Sentinel value for unreachable voxels (no foreground in the volume).
/// `i32::MAX` exceeds any practical chamfer distance for real data.
pub const INF: i32 = i32::MAX;

/// Returns the 7-tap predecessor half-mask S⁻ = {−1, 0}³ ∖ {(0,0,0)} as
/// `(dz, dy, dx)` tuples. Each has all components in {−1, 0} with at least
/// one −1. Together with the 7-tap successor half S⁺, all 26 unique
/// neighbours of a voxel are covered.
#[inline]
const fn predecessor_offsets() -> [(i32, i32, i32); 7] {
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
        (-1, -1, -1),
    ]
}

/// Returns the 7-tap successor half-mask S⁺ = {0, +1}³ ∖ {(0,0,0)} as
/// `(dz, dy, dx)` tuples. Each has all components in {0, +1} with at least
/// one +1.
#[inline]
const fn successor_offsets() -> [(i32, i32, i32); 7] {
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
}

/// Returns the chamfer weight for an offset (dz, dy, dx) under the given
/// metric, in units of `s_min` (the minimum physical spacing). This is the
/// `L_p` length of the offset, with `p = ∞` for chessboard and `p = 1` for
/// taxicab, multiplied by `[wz, wy, wx]` (per-axis physical weights).
///
/// Caller must pass `(dz, dy, dx) ∈ {−1, 0, +1}³ ∖ {(0, 0, 0)}`; behaviour
/// is unspecified for any other input.
#[inline]
const fn weight(dz: i32, dy: i32, dx: i32, w: [i32; 3], metric: ChamferMetric) -> i32 {
    let wz = if dz != 0 { w[0] } else { 0 };
    let wy = if dy != 0 { w[1] } else { 0 };
    let wx = if dx != 0 { w[2] } else { 0 };
    match metric {
        ChamferMetric::Chessboard => {
            // L∞ — max weighted axis delta (manual ternary since `Ord::max`
            // is not yet stable in `const fn`).
            if wz >= wy && wz >= wx {
                wz
            } else if wy >= wx {
                wy
            } else {
                wx
            }
        }
        ChamferMetric::Taxicab => {
            // L1 — sum of weighted axis deltas.
            wz + wy + wx
        }
    }
}

/// Compute the chamfer distance transform via two 3-D raster scans.
///
/// `weights = [wz, wy, wx]` is the integer weight per axis (face-neighbour
/// step size). For a uniform grid, `weights = [1, 1, 1]`. For anisotropic
/// spacing, `weights = round(s / s_min)` per axis.
///
/// The two-pass algorithm uses the **full 7-tap half-mask** in both
/// directions: predecessor S⁻ = {−1, 0}³ ∖ {(0,0,0)} in pass 1 and
/// successor S⁺ = {0, +1}³ ∖ {(0,0,0)} in pass 2. With the L∞ weight
/// assignment, this gives the exact L∞ distance (chessboard); with the
/// L1 weight assignment, this gives the exact L1 distance (taxicab) on a
/// uniform grid.
pub fn cdt_3d(fg: &[bool], dims: [usize; 3], weights: [i32; 3], metric: ChamferMetric) -> Vec<i32> {
    let [nz, ny, nx] = dims;
    let stride = ny * nx;
    let pred = predecessor_offsets();
    let succ = successor_offsets();

    // Seed: **background → 0, foreground → INF** (scipy interior-distance
    // convention). Background voxels stay at 0 across both passes; only
    // foreground voxels are relaxed, accumulating the chamfer distance to
    // the nearest background seed.
    let mut buf: Vec<i32> = fg.iter().map(|&b| if b { INF } else { 0 }).collect();

    // Pass 1 (forward): minimise over the predecessor half.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * stride + iy * nx + ix;
                if buf[i] == 0 {
                    continue;
                }
                let mut best = buf[i];
                for (dz, dy, dx) in pred {
                    let pz = iz as i32 + dz;
                    let py = iy as i32 + dy;
                    let px = ix as i32 + dx;
                    if pz < 0 || py < 0 || px < 0 {
                        continue;
                    }
                    let p = pz as usize * stride + py as usize * nx + px as usize;
                    let w = weight(dz, dy, dx, weights, metric);
                    let c = buf[p].saturating_add(w);
                    if c < best {
                        best = c;
                    }
                }
                buf[i] = best;
            }
        }
    }

    // Pass 2 (backward): minimise over the successor half.
    for iz in (0..nz).rev() {
        for iy in (0..ny).rev() {
            for ix in (0..nx).rev() {
                let i = iz * stride + iy * nx + ix;
                if buf[i] == 0 {
                    continue;
                }
                let mut best = buf[i];
                for (dz, dy, dx) in succ {
                    let pz = iz as i32 + dz;
                    let py = iy as i32 + dy;
                    let px = ix as i32 + dx;
                    if pz >= nz as i32 || py >= ny as i32 || px >= nx as i32 {
                        continue;
                    }
                    let p = pz as usize * stride + py as usize * nx + px as usize;
                    let w = weight(dz, dy, dx, weights, metric);
                    let c = buf[p].saturating_add(w);
                    if c < best {
                        best = c;
                    }
                }
                buf[i] = best;
            }
        }
    }

    buf
}

/// Free-function form (no Image<B, 3> binding) for callers that already have
/// a binary mask in row-major order.
///
/// Returns the chamfer distance map as `Vec<i32>` in units of `s_min`.
/// Unreachable voxels (no foreground in the volume) carry `i32::MAX`.
pub fn chamfer_distance_transform_3d(
    fg: &[bool],
    dims: [usize; 3],
    spacing: [f64; 3],
    metric: ChamferMetric,
) -> Vec<i32> {
    let s_min = spacing.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let weights: [i32; 3] = [
        (spacing[0] / s_min).round() as i32,
        (spacing[1] / s_min).round() as i32,
        (spacing[2] / s_min).round() as i32,
    ];
    cdt_3d(fg, dims, weights, metric)
}

//! Euclidean Distance Transform via Meijster et al. (2000).
//!
//! # Algorithm
//! "A General Algorithm for Computing Distance Transforms in Linear Time"
//! — A. Meijster, J.B.T.M. Roerdink, W.H. Hesselink, *Mathematical Morphology
//! and its Applications to Image and Signal Processing*, Kluwer, 2000.
//!
//! # Mathematical Specification
//! Given a binary image B where B(p) ∈ {0, 1} (0 = background, 1 = foreground),
//! the squared Euclidean Distance Transform is:
//!
//!   EDT²(p) = min_{q : B(q)=1} ‖p − q‖₂²
//!
//! The distance is computed from each voxel to the nearest foreground (object) voxel.
//! Foreground voxels receive distance 0 (they are their own nearest seed).
//! Background voxels receive the squared distance to the nearest foreground voxel.
//!
//! # Separability
//! The algorithm decomposes the D-dimensional problem into D independent 1D passes.
//! For 3D with shape `[nz, ny, nx]`:
//!
//! **Phase 1** (X-axis): For each (z, y) row, compute `g[z][y][x] = min_{x' : B[z][y][x']=1} |x - x'|`.
//! Two-pass forward/backward scan, O(nx) per row.
//!
//! **Phase 2** (Y-axis): For each (z, x) column, compute
//! `dt2[z][y][x] = min_{y'} { (y - y')² + g[z][y'][x]² }` using lower-envelope parabolas, O(ny) per column.
//!
//! **Phase 3** (Z-axis): For each (y, x) position, compute
//! `edt²[z][y][x] = min_{z'} { (z - z')² + dt2[z'][y][x] }` using lower-envelope parabolas, O(nz) per column.
//!
//! Total complexity: O(nz · ny · nx), i.e., linear in the number of voxels.
//!
//! # Sentinel Value
//! When no foreground voxel exists (all-background image), `g` is initialized to
//! `INF_DIST = (nz + ny + nx)` per row dimension, and the final squared distance
//! saturates at `(nz + ny + nx)²`. This is a finite upper bound rather than
//! `f32::INFINITY` to preserve numerical stability in downstream arithmetic.

use ritk_core::filter::ops::{extract_vec_infallible, rebuild};
use ritk_image::Image;
use burn::tensor::backend::Backend;

/// Sentinel for "infinite" distance in integer grid units.
/// Set per-image to `nz + ny + nx` so that `INF_DIST²` never overflows `i64`.
fn inf_dist(shape: &[usize; 3]) -> i64 {
    (shape[0] + shape[1] + shape[2]) as i64
}

// ─── Phase 1: 1D nearest-background scan along X ──────────────────────────

/// For a single row of length `nx`, compute `g[x] = min_{x': row[x']=fg} |x - x'|`.
/// If no foreground voxel exists in the row, all entries are set to `inf`.
fn phase1_row(row: &[bool], nx: usize, inf: i64, out: &mut [i64]) {
    debug_assert_eq!(row.len(), nx);
    debug_assert_eq!(out.len(), nx);

    if nx == 0 {
        return;
    }

    // Forward pass: propagate distance from left.
    if row[0] {
        out[0] = 0; // foreground seed
    } else {
        out[0] = inf;
    }
    for x in 1..nx {
        if row[x] {
            out[x] = 0;
        } else {
            out[x] = out[x - 1].saturating_add(1).min(inf);
        }
    }

    // Backward pass: propagate distance from right, keep minimum.
    let mut d = if row[nx - 1] { 0i64 } else { inf };
    out[nx - 1] = out[nx - 1].min(d);
    for x in (0..nx - 1).rev() {
        if row[x] {
            d = 0;
        } else {
            d = d.saturating_add(1).min(inf);
        }
        out[x] = out[x].min(d);
    }
}

// ─── Lower-envelope parabola algorithm (shared by phases 2 and 3) ──────────

/// Given a 1D array `f` of length `n` representing parabola heights,
/// compute `dt[i] = min_j { (i - j)² + f[j] }` for all i in [0, n).
///
/// Uses the lower envelope of parabolas technique from Felzenszwalb & Huttenlocher (2012)
/// / Meijster et al. (2000). O(n) time and O(n) auxiliary space.
///
/// `v`, `z_buf` are caller-provided scratch buffers of length ≥ n+1.
fn lower_envelope_transform(
    f: &[i64],
    n: usize,
    dt: &mut [i64],
    v: &mut [usize],
    z_buf: &mut [i64],
) {
    debug_assert!(f.len() >= n);
    debug_assert!(dt.len() >= n);
    debug_assert!(v.len() >= n);
    debug_assert!(z_buf.len() > n);

    if n == 0 {
        return;
    }
    if n == 1 {
        dt[0] = f[0];
        return;
    }

    // k = index of rightmost parabola on the lower envelope.
    let mut k: usize = 0;
    v[0] = 0;
    z_buf[0] = i64::MIN;
    z_buf[1] = i64::MAX;

    for q in 1..n {
        // Intersection of parabola centered at q with parabola centered at v[k]:
        //   s = ((f[q] + q²) - (f[v[k]] + v[k]²)) / (2q - 2v[k])
        // We use integer arithmetic and compare with `2 * (q - v[k]) * z_buf[k]`
        // to avoid division.
        loop {
            let vk = v[k] as i64;
            let qq = q as i64;
            // Numerator of intersection: (f[q] + q²) - (f[v[k]] + v[k]²)
            let s_num = (f[q] + qq * qq) - (f[v[k]] + vk * vk);
            let s_den = 2 * (qq - vk); // always > 0 since q > v[k] when they differ

            // Compare s with z_buf[k]. Since z_buf[k] might be MIN/MAX we must be careful.
            // s = s_num / s_den. We want: s <= z_buf[k]?
            // Equivalent to: s_num <= z_buf[k] * s_den  (s_den > 0).
            let remove = if z_buf[k] == i64::MIN || z_buf[k] == i64::MAX {
                false
            } else {
                // Use i128 to avoid overflow in the multiplication.
                (s_num as i128) <= (z_buf[k] as i128) * (s_den as i128)
            };

            if remove {
                if k == 0 {
                    // Replace the sole parabola.
                    v[0] = q;
                    // z_buf[0] stays MIN, z_buf[1] stays MAX is reset below.
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                // Compute the actual intersection point for z_buf[k].
                let vk_prev = v[k - 1] as i64;
                let num = (f[q] + qq * qq) - (f[v[k - 1]] + vk_prev * vk_prev);
                let den = 2 * (qq - vk_prev);
                // Integer division rounding: we want ceil-like behavior for the
                // boundary, but floor is fine because we scan left-to-right and
                // check `q >= z_buf[k]` below.
                z_buf[k] = div_floor(num, den);
                z_buf[k + 1] = i64::MAX;
                break;
            }
        }
    }

    // Scan: assign each position to its minimum parabola.
    let mut j = 0;
    for (q, dt_elem) in dt[..n].iter_mut().enumerate() {
        while j < k && (q as i64) > z_buf[j + 1] {
            j += 1;
        }
        let diff = q as i64 - v[j] as i64;
        *dt_elem = diff * diff + f[v[j]];
    }
}

/// Integer floor division (towards negative infinity) for signed integers.
fn div_floor(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r ^ b) < 0) {
        d - 1
    } else {
        d
    }
}

// ─── Full 3D EDT ───────────────────────────────────────────────────────────

/// Binarize the image: voxels with intensity > `threshold` are foreground (true).
fn binarize<B: Backend>(image: &Image<B, 3>, threshold: f32) -> (Vec<bool>, [usize; 3]) {
    let (vec, shape) = extract_vec_infallible(image);
    let binary: Vec<bool> = vec.iter().map(|&v| v > threshold).collect();
    (binary, shape)
}

/// Index into a flat 3D array with shape `[nz, ny, nx]`.
#[inline(always)]
fn idx3(z: usize, y: usize, x: usize, ny: usize, nx: usize) -> usize {
    z * ny * nx + y * nx + x
}

/// Compute the squared Euclidean distance transform of a 3D binary image.
///
/// # Input
/// Binary mask where voxels with intensity > `foreground_threshold` are foreground (object).
/// The distance is computed from each voxel to the nearest foreground (object) voxel.
/// Foreground voxels receive distance 0 (they are their own nearest seed).
///
/// For the inverse (distance from each voxel to nearest background), invert the mask
/// before calling, or threshold with a value that inverts the sense.
///
/// # Output
/// `Image<B, 3>` with squared Euclidean distances in voxel-unit² (not physical units).
/// To obtain physical distances, multiply by spacing² per axis or apply spacing correction
/// after the transform.
///
/// # Edge Cases
/// - All-background: all output values are 0.0 (no foreground seeds; distance to empty set is defined as 0).
/// - All-foreground: all output values are 0 (all voxels are seeds).
/// - 1×1×1 image: output is sentinel² if background, 0 if foreground.
///
/// # Complexity
/// O(N) where N = nz · ny · nx.
///
/// # References
/// - Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2000).
///   "A General Algorithm for Computing Distance Transforms in Linear Time."
/// - Felzenszwalb, P.F., Huttenlocher, D.P. (2012).
///   "Distance Transforms of Sampled Functions." *Theory of Computing* 8:415–428.
pub fn distance_transform_squared<B: Backend>(
    image: &Image<B, 3>,
    foreground_threshold: f32,
) -> Image<B, 3> {
    let (binary, shape) = binarize(image, foreground_threshold);
    let [nz, ny, nx] = shape;
    let total = nz * ny * nx;
    let inf = inf_dist(&shape);

    // Short-circuit: EDT(p) = min_{q:B(q)=1}||p-q|| over empty set → return 0 everywhere.
    // Convention: when no foreground exists, all distances are defined as 0 (no seeds = no-op).
    if !binary.iter().any(|&b| b) {
        let zeros = vec![0.0f32; total];
        return rebuild(zeros, shape, image);
    }

    // ── Phase 1: scan along X for each (z, y) row ──
    let mut g = vec![0i64; total];
    for z in 0..nz {
        for y in 0..ny {
            let row_start = idx3(z, y, 0, ny, nx);
            let row_end = row_start + nx;
            phase1_row(
                &binary[row_start..row_end],
                nx,
                inf,
                &mut g[row_start..row_end],
            );
        }
    }

    // ── Phase 2: lower-envelope along Y for each (z, x) column ──
    // Convert g to g² for parabola input, then transform along Y.
    for v in g.iter_mut() {
        *v = (*v) * (*v);
    }

    let max_dim = ny.max(nz);
    let mut col_f = vec![0i64; max_dim];
    let mut col_dt = vec![0i64; max_dim];
    let mut scratch_v = vec![0usize; max_dim];
    let mut scratch_z = vec![0i64; max_dim + 1];

    let mut dt2 = vec![0i64; total];

    for z in 0..nz {
        for x in 0..nx {
            // Extract column g²[z][*][x] into col_f.
            for y in 0..ny {
                col_f[y] = g[idx3(z, y, x, ny, nx)];
            }
            lower_envelope_transform(&col_f, ny, &mut col_dt, &mut scratch_v, &mut scratch_z);
            for y in 0..ny {
                dt2[idx3(z, y, x, ny, nx)] = col_dt[y];
            }
        }
    }

    // ── Phase 3: lower-envelope along Z for each (y, x) column ──
    let mut result = vec![0i64; total];

    for y in 0..ny {
        for x in 0..nx {
            for z in 0..nz {
                col_f[z] = dt2[idx3(z, y, x, ny, nx)];
            }
            lower_envelope_transform(&col_f, nz, &mut col_dt, &mut scratch_v, &mut scratch_z);
            for z in 0..nz {
                result[idx3(z, y, x, ny, nx)] = col_dt[z];
            }
        }
    }

    // ── Convert to f32 tensor ──
    let float_result: Vec<f32> = result.iter().map(|&v| v as f32).collect();
    rebuild(float_result, shape, image)
}

/// Compute the Euclidean distance transform (square root of squared distances).
///
/// Equivalent to `sqrt(distance_transform_squared(image, foreground_threshold))` per voxel.
///
/// See [`distance_transform_squared`] for full documentation.
pub fn distance_transform<B: Backend>(
    image: &Image<B, 3>,
    foreground_threshold: f32,
) -> Image<B, 3> {
    let sq = distance_transform_squared(image, foreground_threshold);
    let sqrt_tensor = sq.data().clone().sqrt();
    Image::new(sqrt_tensor, *sq.origin(), *sq.spacing(), *sq.direction())
}

/// Unit struct providing associated-function API for distance transforms.
///
/// All methods delegate to the free functions [`distance_transform_squared`]
/// and [`distance_transform`].
pub struct DistanceTransform;

impl DistanceTransform {
    /// Compute the squared Euclidean distance transform.
    /// See [`distance_transform_squared`].
    pub fn squared<B: Backend>(image: &Image<B, 3>, foreground_threshold: f32) -> Image<B, 3> {
        distance_transform_squared(image, foreground_threshold)
    }

    /// Compute the Euclidean distance transform.
    /// See [`distance_transform`].
    pub fn transform<B: Backend>(image: &Image<B, 3>, foreground_threshold: f32) -> Image<B, 3> {
        distance_transform(image, foreground_threshold)
    }
}

#[cfg(test)]
#[path = "tests_distance_transform.rs"]
mod tests_distance_transform;

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
//!   EDT²(p) = min_{q : B(q)=0} ‖p − q‖₂²
//!
//! Foreground voxels receive the squared distance to the nearest background voxel.
//! Background voxels receive distance 0.
//!
//! # Separability
//! The algorithm decomposes the D-dimensional problem into D independent 1D passes.
//! For 3D with shape `[nz, ny, nx]`:
//!
//! **Phase 1** (X-axis): For each (z, y) row, compute `g[z][y][x] = min_{x' : B[z][y][x']=0} |x - x'|`.
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
//! When no background voxel exists (all-foreground image), `g` is initialized to
//! `INF_DIST = (nz + ny + nx)` per row dimension, and the final squared distance
//! saturates at `(nz + ny + nx)²`. This is a finite upper bound rather than
//! `f32::INFINITY` to preserve numerical stability in downstream arithmetic.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Sentinel for "infinite" distance in integer grid units.
/// Set per-image to `nz + ny + nx` so that `INF_DIST²` never overflows `i64`.
fn inf_dist(shape: &[usize; 3]) -> i64 {
    (shape[0] + shape[1] + shape[2]) as i64
}

// ─── Phase 1: 1D nearest-background scan along X ──────────────────────────

/// For a single row of length `nx`, compute `g[x] = min_{x': row[x']=bg} |x - x'|`.
/// If no background voxel exists in the row, all entries are set to `inf`.
fn phase1_row(row: &[bool], nx: usize, inf: i64, out: &mut [i64]) {
    debug_assert_eq!(row.len(), nx);
    debug_assert_eq!(out.len(), nx);

    if nx == 0 {
        return;
    }

    // Forward pass: propagate distance from left.
    if !row[0] {
        out[0] = 0; // background
    } else {
        out[0] = inf;
    }
    for x in 1..nx {
        if !row[x] {
            out[x] = 0;
        } else {
            out[x] = out[x - 1].saturating_add(1).min(inf);
        }
    }

    // Backward pass: propagate distance from right, keep minimum.
    let mut d = if !row[nx - 1] { 0i64 } else { inf };
    out[nx - 1] = out[nx - 1].min(d);
    for x in (0..nx - 1).rev() {
        if !row[x] {
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
    debug_assert!(z_buf.len() >= n + 1);

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
            let remove = if z_buf[k] == i64::MIN {
                false
            } else if z_buf[k] == i64::MAX {
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
                z_buf[k] = div_floor_i64(num, den);
                z_buf[k + 1] = i64::MAX;
                break;
            }
        }
    }

    // Scan: assign each position to its minimum parabola.
    let mut j = 0;
    for q in 0..n {
        while j < k && (q as i64) > z_buf[j + 1] {
            j += 1;
        }
        let diff = q as i64 - v[j] as i64;
        dt[q] = diff * diff + f[v[j]];
    }
}

/// Integer floor division (towards negative infinity) for signed integers.
fn div_floor_i64(a: i64, b: i64) -> i64 {
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
    let shape = image.shape();
    let tensor_data = image.data().clone().into_data();
    let slice = tensor_data.as_slice::<f32>().expect("f32 tensor data");
    let binary: Vec<bool> = slice.iter().map(|&v| v > threshold).collect();
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
/// The distance is computed FROM each foreground voxel TO the nearest background voxel.
/// Background voxels receive distance 0.
///
/// For the inverse (distance from background to nearest foreground), invert the mask
/// before calling, or threshold with a value that inverts the sense.
///
/// # Output
/// `Image<B, 3>` with squared Euclidean distances in voxel-unit² (not physical units).
/// To obtain physical distances, multiply by spacing² per axis or apply spacing correction
/// after the transform.
///
/// # Edge Cases
/// - All-background: all output values are 0.
/// - All-foreground: output values are `(nz + ny + nx)²` (finite sentinel upper bound).
/// - 1×1×1 image: output is 0 if background, sentinel² if foreground.
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
    let device = image.data().device();
    let tensor =
        Tensor::<B, 3>::from_data(TensorData::new(float_result, Shape::new(shape)), &device);

    Image::new(
        tensor,
        image.origin().clone(),
        image.spacing().clone(),
        image.direction().clone(),
    )
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
    Image::new(
        sqrt_tensor,
        sq.origin().clone(),
        sq.spacing().clone(),
        sq.direction().clone(),
    )
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Helper: index into flat 3D array.
    fn at(vals: &[f32], z: usize, y: usize, x: usize, ny: usize, nx: usize) -> f32 {
        vals[z * ny * nx + y * nx + x]
    }

    // ── Test 1: Single foreground voxel in center of 5×5×5 ────────────────

    #[test]
    fn test_single_foreground_voxel_center_5x5x5() {
        // 5×5×5 image, all background (0.0) except center voxel (2,2,2) = 1.0.
        // EDT²(2,2,2) = min over all background voxels of squared distance.
        // The nearest background voxels are the 6 face-adjacent neighbors at distance 1.
        // EDT²(2,2,2) = 1.
        //
        // Corner voxels at (0,0,0), (4,4,4), etc.:
        //   distance to nearest background = 0 (they ARE background) → EDT² = 0.
        //
        // Since only the center is foreground, only that voxel has nonzero distance.
        let dims = [5, 5, 5];
        let total = 125;
        let mut data = vec![0.0f32; total];
        data[idx3(2, 2, 2, 5, 5)] = 1.0; // foreground

        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        // Only the center voxel is foreground; nearest background is at distance 1.
        assert_eq!(
            at(&vals, 2, 2, 2, 5, 5),
            1.0,
            "center foreground voxel EDT² must be 1"
        );

        // All background voxels must have EDT² = 0.
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    if (z, y, x) != (2, 2, 2) {
                        let v = at(&vals, z, y, x, 5, 5);
                        assert_eq!(
                            v, 0.0,
                            "background voxel ({z},{y},{x}) must have EDT²=0, got {v}"
                        );
                    }
                }
            }
        }

        // Verify the non-squared transform: √1 = 1.0.
        let edt = distance_transform(&image, 0.5);
        let edt_vals = get_values(&edt);
        let center_dist = at(&edt_vals, 2, 2, 2, 5, 5);
        assert!(
            (center_dist - 1.0).abs() < 1e-6,
            "center EDT must be 1.0, got {center_dist}"
        );
    }

    // ── Test 2: All-foreground image → sentinel distance ──────────────────

    #[test]
    fn test_all_foreground_image() {
        // 3×3×3, all voxels = 1.0 (foreground). No background exists.
        // EDT² for every voxel = inf_dist² = (3+3+3)² = 81.
        let dims = [3, 3, 3];
        let data = vec![1.0f32; 27];
        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        let expected = (3 + 3 + 3) as f32;
        let expected_sq = expected * expected; // 81.0

        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(
                v, expected_sq,
                "all-foreground voxel {i} must have EDT²={expected_sq}, got {v}"
            );
        }
    }

    // ── Test 3: All-background image → all distances 0 ────────────────────

    #[test]
    fn test_all_background_image() {
        let dims = [4, 3, 5];
        let data = vec![0.0f32; 60];
        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(v, 0.0, "all-background voxel {i} must have EDT²=0, got {v}");
        }
    }

    // ── Test 4: Single background voxel at corner → analytical distances ──

    #[test]
    fn test_single_background_voxel_at_corner() {
        // 5×5×5, all foreground (1.0) except corner (0,0,0) = 0.0.
        // EDT²(z,y,x) = z² + y² + x²  (distance to the only background voxel).
        let dims = [5, 5, 5];
        let total = 125;
        let mut data = vec![1.0f32; total];
        data[idx3(0, 0, 0, 5, 5)] = 0.0; // background

        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        for z in 0..5usize {
            for y in 0..5usize {
                for x in 0..5usize {
                    let expected = (z * z + y * y + x * x) as f32;
                    let actual = at(&vals, z, y, x, 5, 5);
                    assert_eq!(
                        actual, expected,
                        "EDT²({z},{y},{x}) = {actual}, expected {expected}"
                    );
                }
            }
        }

        // Corner (4,4,4): EDT = √(16+16+16) = √48
        let edt = distance_transform(&image, 0.5);
        let edt_vals = get_values(&edt);
        let corner = at(&edt_vals, 4, 4, 4, 5, 5);
        let expected_corner = (48.0f32).sqrt();
        assert!(
            (corner - expected_corner).abs() < 1e-4,
            "EDT(4,4,4) = {corner}, expected {expected_corner}"
        );
    }

    // ── Test 5: 2D-equivalent test (nz=1 plane) with known geometry ───────

    #[test]
    fn test_2d_plane_known_geometry() {
        // A 1×5×5 "plane" (flat in Z). Background is a vertical stripe at x=0.
        // All voxels at x=0 are background (0.0), rest are foreground (1.0).
        //
        // For any foreground voxel at (0, y, x) with x > 0:
        //   EDT² = x²  (nearest background is at (0, y, 0), distance = x).
        let ny = 5;
        let nx = 5;
        let dims = [1, ny, nx];
        let mut data = vec![1.0f32; ny * nx];
        for y in 0..ny {
            data[y * nx + 0] = 0.0; // x=0 column is background
        }

        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        for y in 0..ny {
            for x in 0..nx {
                let expected = (x * x) as f32;
                let actual = at(&vals, 0, y, x, ny, nx);
                assert_eq!(
                    actual, expected,
                    "EDT²(0,{y},{x}) = {actual}, expected {expected}"
                );
            }
        }
    }

    // ── Test 6: Boundary test — 1×1×1 image ──────────────────────────────

    #[test]
    fn test_1x1x1_background() {
        let image = make_image_3d(vec![0.0], [1, 1, 1]);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0], 0.0, "single background voxel EDT² = 0");
    }

    #[test]
    fn test_1x1x1_foreground() {
        let image = make_image_3d(vec![1.0], [1, 1, 1]);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);
        assert_eq!(vals.len(), 1);
        // Sentinel: (1+1+1)² = 9
        assert_eq!(vals[0], 9.0, "single foreground voxel EDT² = sentinel²");
    }

    // ── Test 7: Two background voxels — verify minimum is chosen ──────────

    #[test]
    fn test_two_background_voxels_minimum_distance() {
        // 1×1×7 row. Background at x=0 and x=6. Foreground at x=1..5.
        // EDT²(0,0,x):
        //   x=0: 0 (bg), x=1: 1, x=2: 4, x=3: 9 (but also 9 from x=6 side),
        //   x=4: 4, x=5: 1, x=6: 0 (bg).
        let nx = 7;
        let mut data = vec![1.0f32; nx];
        data[0] = 0.0;
        data[6] = 0.0;

        let image = make_image_3d(data, [1, 1, nx]);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        let expected = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0];
        for x in 0..nx {
            assert_eq!(
                vals[x], expected[x],
                "EDT²(0,0,{x}) = {}, expected {}",
                vals[x], expected[x]
            );
        }
    }

    // ── Test 8: DistanceTransform unit struct API consistency ──────────────

    #[test]
    fn test_unit_struct_api_matches_free_functions() {
        let dims = [3, 3, 3];
        let mut data = vec![1.0f32; 27];
        data[0] = 0.0;

        let image = make_image_3d(data, dims);
        let free_sq = distance_transform_squared(&image, 0.5);
        let struct_sq = DistanceTransform::squared(&image, 0.5);

        let free_vals = get_values(&free_sq);
        let struct_vals = get_values(&struct_sq);
        assert_eq!(
            free_vals, struct_vals,
            "unit struct must match free function"
        );

        let free_edt = distance_transform(&image, 0.5);
        let struct_edt = DistanceTransform::transform(&image, 0.5);

        let free_edt_vals = get_values(&free_edt);
        let struct_edt_vals = get_values(&struct_edt);
        assert_eq!(
            free_edt_vals, struct_edt_vals,
            "unit struct transform must match free function"
        );
    }

    // ── Test 9: Spatial metadata preserved ────────────────────────────────

    #[test]
    fn test_preserves_spatial_metadata() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2])),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let result = distance_transform_squared(&image, 0.5);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
        assert_eq!(result.shape(), [2, 2, 2]);
    }

    // ── Test 10: Checkerboard pattern ─────────────────────────────────────

    #[test]
    fn test_checkerboard_3d() {
        // 1×2×2 checkerboard:
        //   (0,0,0)=bg, (0,0,1)=fg, (0,1,0)=fg, (0,1,1)=bg
        // Foreground voxels each have a face-adjacent background neighbor at distance 1.
        // EDT² for foreground = 1, EDT² for background = 0.
        let data = vec![0.0, 1.0, 1.0, 0.0];
        let image = make_image_3d(data, [1, 2, 2]);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        assert_eq!(vals[0], 0.0, "(0,0,0) bg");
        assert_eq!(vals[1], 1.0, "(0,0,1) fg, nearest bg at distance 1");
        assert_eq!(vals[2], 1.0, "(0,1,0) fg, nearest bg at distance 1");
        assert_eq!(vals[3], 0.0, "(0,1,1) bg");
    }

    // ── Test 11: Asymmetric shape ─────────────────────────────────────────

    #[test]
    fn test_asymmetric_shape_2x3x4() {
        // All foreground except (0,0,0) = background.
        // EDT²(z,y,x) = z² + y² + x².
        let dims = [2, 3, 4];
        let total = 24;
        let mut data = vec![1.0f32; total];
        data[0] = 0.0;

        let image = make_image_3d(data, dims);
        let result = distance_transform_squared(&image, 0.5);
        let vals = get_values(&result);

        for z in 0..2usize {
            for y in 0..3usize {
                for x in 0..4usize {
                    let expected = (z * z + y * y + x * x) as f32;
                    let actual = at(&vals, z, y, x, 3, 4);
                    assert_eq!(
                        actual, expected,
                        "EDT²({z},{y},{x}) = {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    // ── Internal: lower_envelope_transform correctness ────────────────────

    #[test]
    fn test_lower_envelope_single_element() {
        let f = [7i64];
        let mut dt = [0i64];
        let mut v = [0usize];
        let mut z = [0i64; 2];
        lower_envelope_transform(&f, 1, &mut dt, &mut v, &mut z);
        assert_eq!(dt[0], 7, "single element passthrough");
    }

    #[test]
    fn test_lower_envelope_uniform() {
        // f = [5, 5, 5, 5]. dt[i] = min_j { (i-j)² + 5 } = 5 (at j=i).
        let f = [5i64; 4];
        let mut dt = [0i64; 4];
        let mut v = [0usize; 4];
        let mut z = [0i64; 5];
        lower_envelope_transform(&f, 4, &mut dt, &mut v, &mut z);
        for i in 0..4 {
            assert_eq!(dt[i], 5, "uniform f: dt[{i}] = 5, got {}", dt[i]);
        }
    }

    #[test]
    fn test_lower_envelope_known_case() {
        // f = [0, INF, INF, INF, 0] where INF = 100.
        // dt[i] = min(i², (i-4)²) since f[0]=0, f[4]=0.
        // dt[0]=0, dt[1]=1, dt[2]=4, dt[3]=1, dt[4]=0.
        let inf = 100i64;
        let f = [0, inf * inf, inf * inf, inf * inf, 0];
        let mut dt = [0i64; 5];
        let mut v = [0usize; 5];
        let mut z = [0i64; 6];
        lower_envelope_transform(&f, 5, &mut dt, &mut v, &mut z);
        assert_eq!(dt[0], 0);
        assert_eq!(dt[1], 1);
        assert_eq!(dt[2], 4);
        assert_eq!(dt[3], 1);
        assert_eq!(dt[4], 0);
    }

    // ── Test: phase1_row ──────────────────────────────────────────────────

    #[test]
    fn test_phase1_row_all_background() {
        let row = [false, false, false, false];
        let mut out = [0i64; 4];
        phase1_row(&row, 4, 100, &mut out);
        assert_eq!(out, [0, 0, 0, 0]);
    }

    #[test]
    fn test_phase1_row_single_bg_at_start() {
        let row = [false, true, true, true, true];
        let mut out = [0i64; 5];
        phase1_row(&row, 5, 100, &mut out);
        assert_eq!(out, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_phase1_row_bg_at_both_ends() {
        let row = [false, true, true, true, false];
        let mut out = [0i64; 5];
        phase1_row(&row, 5, 100, &mut out);
        assert_eq!(out, [0, 1, 2, 1, 0]);
    }

    #[test]
    fn test_phase1_row_all_foreground() {
        let row = [true, true, true];
        let mut out = [0i64; 3];
        phase1_row(&row, 3, 100, &mut out);
        // All foreground, no background: distances saturate to inf=100.
        assert_eq!(out, [100, 100, 100]);
    }
}

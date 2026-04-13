//! Connected-threshold region growing for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given an intensity image I and a seed voxel s ∈ ℤ³, the connected-threshold
//! region R is the maximal connected set satisfying:
//!
//!   R = { p ∈ ℤ³ : p is reachable from s through voxels with lower ≤ I(p) ≤ upper }
//!
//! Reachability uses 6-connectivity (face-adjacent voxels only) to ensure that
//! the segmented region is always a topologically connected 3-D object.
//!
//! # Algorithm — Breadth-First Flood Fill
//!
//! 1. Validate that I(seed) ∈ [lower, upper].
//! 2. Push seed onto a FIFO queue; mark seed as visited.
//! 3. While queue is non-empty:
//!    a. Pop voxel p.
//!    b. Set output(p) = 1.
//!    c. For each 6-connected neighbour q of p:
//!       - Skip if already visited.
//!       - Skip if I(q) ∉ [lower, upper].
//!       - Otherwise mark visited and enqueue.
//! 4. Return binary mask.
//!
//! # Complexity
//! - Time:  O(|R|) — each voxel is visited at most once.
//! - Space: O(|R|) for the visited set and queue.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use std::collections::VecDeque;

// ── Public types ──────────────────────────────────────────────────────────────

/// Connected-threshold region-growing filter.
///
/// Segments the region reachable from a seed voxel whose intensities fall
/// within the closed interval [lower, upper].
pub struct ConnectedThresholdFilter {
    /// Seed voxel in [z, y, x] index space.
    pub seed: [usize; 3],
    /// Inclusive lower intensity bound.
    pub lower: f32,
    /// Inclusive upper intensity bound.
    pub upper: f32,
}

impl ConnectedThresholdFilter {
    /// Create a `ConnectedThresholdFilter`.
    ///
    /// # Panics
    /// Panics if `lower > upper`.
    pub fn new(seed: [usize; 3], lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "lower bound {lower} must be ≤ upper bound {upper}"
        );
        Self { seed, lower, upper }
    }

    /// Apply region growing to `image`.
    ///
    /// Returns a binary mask (values in {0.0, 1.0}) with the same shape and
    /// spatial metadata as `image`.  If the seed voxel's intensity does not
    /// satisfy `lower ≤ I(seed) ≤ upper`, the output is all-zero.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        connected_threshold(image, self.seed, self.lower, self.upper)
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Connected-threshold region growing starting from `seed`.
///
/// Returns a binary mask whose shape and spatial metadata match `image`.
/// Voxels included in the grown region are set to 1.0; all others to 0.0.
///
/// If the seed's intensity does not lie in `[lower, upper]`, the returned
/// mask is all zeros (empty region).
///
/// # Panics
/// Panics if `lower > upper` or if `seed` is out of bounds for `image`.
pub fn connected_threshold<B: Backend>(
    image: &Image<B, 3>,
    seed: [usize; 3],
    lower: f32,
    upper: f32,
) -> Image<B, 3> {
    assert!(
        lower <= upper,
        "lower bound {lower} must be ≤ upper bound {upper}"
    );

    let shape = image.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);

    assert!(
        seed[0] < nz && seed[1] < ny && seed[2] < nx,
        "seed {:?} is out of bounds for image shape {:?}",
        seed,
        shape
    );

    let device = image.data().device();
    let img_data = image.data().clone().into_data();
    let img_slice = img_data.as_slice::<f32>().expect("f32 image tensor data");

    let result = flood_fill(img_slice, shape, seed, lower, upper);

    let td = TensorData::new(result, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);

    Image::new(
        tensor,
        image.origin().clone(),
        image.spacing().clone(),
        image.direction().clone(),
    )
}

// ── Core BFS flood fill ───────────────────────────────────────────────────────

/// Perform BFS flood fill on a flat `[nz × ny × nx]` intensity slice.
///
/// Returns a flat binary Vec<f32> of the same length as `data`.
fn flood_fill(
    data: &[f32],
    dims: [usize; 3],
    seed: [usize; 3],
    lower: f32,
    upper: f32,
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Check seed intensity.
    let seed_val = data[flat(seed[0], seed[1], seed[2])];
    if seed_val < lower || seed_val > upper {
        return vec![0.0_f32; n];
    }

    let mut output = vec![0.0_f32; n];
    let mut visited = vec![false; n];

    let seed_flat = flat(seed[0], seed[1], seed[2]);
    visited[seed_flat] = true;

    // BFS queue stores flat voxel indices.
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(1024);
    queue.push_back(seed_flat);

    // 6-connectivity face offsets as signed (dz, dy, dx).
    let face_offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    while let Some(curr_flat) = queue.pop_front() {
        // Mark as foreground in output.
        output[curr_flat] = 1.0;

        // Decode flat index back to (z, y, x).
        let iz = curr_flat / (ny * nx);
        let rem = curr_flat % (ny * nx);
        let iy = rem / nx;
        let ix = rem % nx;

        // Expand to 6 face-adjacent neighbours.
        for &(dz, dy, dx) in &face_offsets {
            let nz_i = iz as isize + dz;
            let ny_i = iy as isize + dy;
            let nx_i = ix as isize + dx;

            // Bounds check.
            if nz_i < 0
                || nz_i >= nz as isize
                || ny_i < 0
                || ny_i >= ny as isize
                || nx_i < 0
                || nx_i >= nx as isize
            {
                continue;
            }

            let n_flat = flat(nz_i as usize, ny_i as usize, nx_i as usize);

            if visited[n_flat] {
                continue;
            }

            let intensity = data[n_flat];
            if intensity < lower || intensity > upper {
                continue;
            }

            visited[n_flat] = true;
            queue.push_back(n_flat);
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::Image,
        spatial::{Direction, Point, Spacing},
    };
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
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

    fn count_foreground(image: &Image<TestBackend, 3>) -> usize {
        get_values(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── Positive tests ────────────────────────────────────────────────────────

    #[test]
    fn test_uniform_image_grows_entire_volume() {
        // All voxels have intensity 100; lower=50, upper=150 → entire 4×4×4 grown.
        let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
        let result = connected_threshold(&image, [0, 0, 0], 50.0, 150.0);
        assert_eq!(count_foreground(&result), 64);
    }

    #[test]
    fn test_single_voxel_exactly_on_lower_bound() {
        // Seed has intensity exactly equal to lower → should be included.
        let image = make_image(vec![50.0_f32; 8], [2, 2, 2]);
        let result = connected_threshold(&image, [0, 0, 0], 50.0, 100.0);
        assert_eq!(count_foreground(&result), 8);
    }

    #[test]
    fn test_single_voxel_exactly_on_upper_bound() {
        let image = make_image(vec![100.0_f32; 8], [2, 2, 2]);
        let result = connected_threshold(&image, [1, 1, 1], 50.0, 100.0);
        assert_eq!(count_foreground(&result), 8);
    }

    #[test]
    fn test_two_regions_seed_selects_one() {
        // 1×1×6 volume: [100, 100, 100, 10, 10, 10].
        // Seed at (0,0,0) with lower=50, upper=200 → only first 3 voxels.
        let values = vec![100.0, 100.0, 100.0, 10.0, 10.0, 10.0];
        let image = make_image(values, [1, 1, 6]);
        let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
        let vals = get_values(&result);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 1.0);
        assert_eq!(vals[3], 0.0);
        assert_eq!(vals[4], 0.0);
        assert_eq!(vals[5], 0.0);
    }

    #[test]
    fn test_connectivity_is_6_not_diagonal() {
        // 3×3×1 slice:
        //   A 0 0
        //   0 0 0
        //   0 0 B
        // A and B are high-intensity; all other voxels are low.
        // With 6-connectivity, seeding from A cannot reach B.
        let mut values = vec![0.0_f32; 9];
        values[0] = 200.0; // A at (0,0,0)
        values[8] = 200.0; // B at (0,2,2)
        let image = make_image(values, [1, 3, 3]);
        let result = connected_threshold(&image, [0, 0, 0], 100.0, 255.0);
        let vals = get_values(&result);
        // Only A should be foreground; B is not 6-connected to A.
        assert_eq!(vals[0], 1.0, "seed voxel A must be foreground");
        assert_eq!(vals[8], 0.0, "diagonal voxel B must not be reached");
        assert_eq!(count_foreground(&result), 1);
    }

    #[test]
    fn test_filter_struct_matches_function() {
        let values: Vec<f32> = (0..27).map(|i| i as f32 * 10.0).collect();
        let image = make_image(values, [3, 3, 3]);

        let via_fn = connected_threshold(&image, [1, 1, 1], 50.0, 150.0);
        let via_struct = ConnectedThresholdFilter::new([1, 1, 1], 50.0, 150.0).apply(&image);

        let fn_vals = get_values(&via_fn);
        let struct_vals = get_values(&via_struct);
        assert_eq!(
            fn_vals, struct_vals,
            "function and struct must produce identical results"
        );
    }

    // ── Negative / boundary tests ─────────────────────────────────────────────

    #[test]
    fn test_seed_outside_range_returns_all_zero() {
        // Seed intensity = 5.0, range [50, 200] → seed excluded → empty mask.
        let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
        let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
        assert_eq!(count_foreground(&result), 0);
    }

    #[test]
    fn test_intensity_just_above_upper_bound_excluded() {
        // Single voxel with intensity 201 when upper = 200.
        let image = make_image(vec![201.0_f32; 1], [1, 1, 1]);
        let result = connected_threshold(&image, [0, 0, 0], 0.0, 200.0);
        assert_eq!(count_foreground(&result), 0);
    }

    #[test]
    fn test_intensity_just_below_lower_bound_excluded() {
        let image = make_image(vec![49.0_f32; 1], [1, 1, 1]);
        let result = connected_threshold(&image, [0, 0, 0], 50.0, 200.0);
        assert_eq!(count_foreground(&result), 0);
    }

    #[test]
    fn test_fully_grown_region_output_is_strictly_binary() {
        let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
        let result = connected_threshold(&image, [1, 1, 1], 50.0, 150.0);
        for &v in get_values(&result).iter() {
            assert!(
                v == 0.0 || v == 1.0,
                "output must be strictly binary, got {v}"
            );
        }
    }

    #[test]
    fn test_spatial_metadata_preserved() {
        use crate::spatial::{Direction, Point, Spacing};
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![100.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);

        let result = connected_threshold(&image, [0, 0, 0], 50.0, 150.0);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
    }

    // ── 3-D volumetric test ───────────────────────────────────────────────────

    #[test]
    fn test_3d_sphere_region_growing() {
        // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
        // background intensity 50; lower=150, upper=255.
        // Region growing from center should capture exactly the sphere.
        let (nz, ny, nx) = (9, 9, 9);
        let mut values = vec![50.0_f32; nz * ny * nx];
        let (cz, cy, cx) = (4isize, 4isize, 4isize);
        let r2 = 9isize; // radius 3

        let mut sphere_count = 0;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as isize - cz;
                    let dy = iy as isize - cy;
                    let dx = ix as isize - cx;
                    if dz * dz + dy * dy + dx * dx <= r2 {
                        values[iz * ny * nx + iy * nx + ix] = 200.0;
                        sphere_count += 1;
                    }
                }
            }
        }

        let image = make_image(values, [nz, ny, nx]);
        let result = connected_threshold(&image, [4, 4, 4], 150.0, 255.0);
        assert_eq!(
            count_foreground(&result),
            sphere_count,
            "grown region must match sphere voxel count exactly"
        );
    }
}

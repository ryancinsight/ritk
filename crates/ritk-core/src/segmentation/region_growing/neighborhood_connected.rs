//! Neighborhood-connected region growing for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given an intensity image I: ℤ³ → ℝ, a seed voxel s, intensity bounds
//! [L, U] ⊂ ℝ, and a rectangular neighborhood radius R = (rz, ry, rx) ∈ ℕ³,
//! the neighborhood-connected region growing algorithm performs BFS flood-fill
//! where the inclusion predicate requires **all** voxels in the candidate's
//! local neighborhood to satisfy the intensity bounds.
//!
//! ## Definition (Neighborhood Admissibility)
//!
//! For a voxel v ∈ ℤ³, define the rectangular neighborhood:
//!
//!   N_R(v) = { n ∈ ℤ³ : |nᵢ − vᵢ| ≤ Rᵢ  ∀ i ∈ {z, y, x} } ∩ dom(I)
//!
//! The **neighborhood admissibility predicate** is:
//!
//!   P(v) ≡ ∀ n ∈ N_R(v) : L ≤ I(n) ≤ U
//!
//! ## Theorem (Neighborhood-Consistent Region Growing)
//!
//! The grown region G ⊆ dom(I) satisfies:
//!
//! 1. **Seed inclusion**: s ∈ G ⟺ P(s)
//! 2. **Neighborhood consistency**: ∀ v ∈ G : P(v)
//! 3. **6-connectedness**: G is 6-connected (face-adjacent in 3-D)
//! 4. **Maximality**: No voxel in the 6-connected boundary of G satisfies P
//!
//! ## Proof sketch
//!
//! The BFS processes voxels in FIFO order. Each dequeued voxel p has already
//! passed P(p). For each 6-connected neighbor q of p:
//! - If q is unvisited and P(q) holds, q is enqueued and marked visited.
//! - The visited set is monotonically increasing, so no voxel is tested twice.
//! - BFS terminates when the queue is empty, establishing maximality.
//! - 6-connectedness follows from the BFS expansion via face offsets only.
//!
//! ## Algorithm — BFS with Neighborhood Predicate
//!
//! 1. Validate seed bounds. If ¬P(s), return empty mask.
//! 2. Initialize BFS queue Q ← {s}, visited[s] ← true, output[s] ← 1.
//! 3. While Q ≠ ∅:
//!    a. Dequeue p from Q.
//!    b. For each q ∈ N₆(p) (6 face-adjacent neighbors):
//!       - If q ∉ visited and P(q): visited[q] ← true, output[q] ← 1, enqueue q.
//! 4. Return binary mask.
//!
//! ## Distinction from Connected Threshold
//!
//! `ConnectedThresholdFilter` checks only I(q) ∈ [L, U] (single-voxel predicate).
//! `NeighborhoodConnectedFilter` checks ∀ n ∈ N_R(q): I(n) ∈ [L, U] (neighborhood
//! predicate). The neighborhood predicate rejects isolated noise voxels that happen
//! to fall within [L, U] but whose local context does not, providing more robust
//! segmentation in the presence of salt-and-pepper or Rician noise.
//!
//! ## Complexity
//!
//! - Time: O(|G| · |N_R|) where |N_R| = ∏ᵢ (2Rᵢ + 1). Each candidate voxel
//!   requires a full neighborhood scan. For the default R = (1,1,1), |N_R| = 27.
//! - Space: O(|dom(I)|) for visited and output arrays.
//!
//! # References
//!
//! - Zucker, S.W. (1976). "Region Growing: Childhood and Adolescence."
//!   *Computer Graphics and Image Processing*, 5(3), 382-399.
//! - Adams, R., & Bischof, L. (1994). "Seeded Region Growing."
//!   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 16(6), 641-647.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use std::collections::VecDeque;

// ── Public types ─────────────────────────────────────────────────────────────

/// Neighborhood-connected region-growing filter.
///
/// Segments the region reachable from a seed voxel where every voxel in the
/// candidate's rectangular neighborhood satisfies the intensity bounds [lower, upper].
///
/// This provides more noise-robust segmentation than [`super::ConnectedThresholdFilter`]
/// by requiring local neighborhood consistency before voxel inclusion.
pub struct NeighborhoodConnectedFilter {
    /// Seed voxel in [z, y, x] index space.
    pub seed: [usize; 3],
    /// Inclusive lower intensity bound.
    pub lower: f32,
    /// Inclusive upper intensity bound.
    pub upper: f32,
    /// Rectangular neighborhood half-radius [rz, ry, rx].
    ///
    /// The full neighborhood for a voxel v is the set of all voxels n
    /// with |nᵢ − vᵢ| ≤ Rᵢ. Default: [1, 1, 1] (3×3×3 = 27 voxels).
    pub radius: [usize; 3],
}

impl NeighborhoodConnectedFilter {
    /// Create a `NeighborhoodConnectedFilter` with default radius [1, 1, 1].
    ///
    /// # Panics
    /// Panics if `lower > upper`.
    pub fn new(seed: [usize; 3], lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "lower bound {lower} must be ≤ upper bound {upper}"
        );
        Self {
            seed,
            lower,
            upper,
            radius: [1, 1, 1],
        }
    }

    /// Set the rectangular neighborhood half-radius.
    ///
    /// A radius of [rz, ry, rx] produces a neighborhood of size
    /// (2·rz+1) × (2·ry+1) × (2·rx+1).
    #[must_use]
    pub fn with_radius(mut self, radius: [usize; 3]) -> Self {
        self.radius = radius;
        self
    }

    /// Apply neighborhood-connected region growing to `image`.
    ///
    /// Returns a binary mask (values in {0.0, 1.0}) with the same shape and
    /// spatial metadata as `image`. If the seed voxel's neighborhood does not
    /// fully satisfy the intensity bounds, the output is all-zero.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        neighborhood_connected(image, self.seed, self.lower, self.upper, self.radius)
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Neighborhood-connected region growing starting from `seed`.
///
/// Returns a binary mask whose shape and spatial metadata match `image`.
/// Voxels included in the grown region are set to 1.0; all others to 0.0.
///
/// A voxel is admitted only if **every** voxel within its rectangular
/// neighborhood of half-radius `radius` has intensity in [lower, upper].
/// Expansion uses 6-connectivity (face-adjacent neighbors).
///
/// # Arguments
/// * `image` — input intensity image (3-D).
/// * `seed` — starting voxel [z, y, x].
/// * `lower` — inclusive lower intensity bound.
/// * `upper` — inclusive upper intensity bound.
/// * `radius` — rectangular neighborhood half-radius [rz, ry, rx].
///
/// # Panics
/// Panics if `lower > upper` or if `seed` is out of bounds for `image`.
pub fn neighborhood_connected<B: Backend>(
    image: &Image<B, 3>,
    seed: [usize; 3],
    lower: f32,
    upper: f32,
    radius: [usize; 3],
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

    let result = grow_neighborhood(img_slice, shape, seed, lower, upper, radius);

    let td = TensorData::new(result, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);

    Image::new(
        tensor,
        image.origin().clone(),
        image.spacing().clone(),
        image.direction().clone(),
    )
}

// ── Core BFS with neighborhood predicate ─────────────────────────────────────

/// Check whether all voxels in the rectangular neighborhood N_R(v) satisfy
/// L ≤ I(n) ≤ U.
///
/// The neighborhood is clamped to the image domain: voxels outside domain
/// boundaries are excluded from the check (they do not cause rejection).
///
/// # Invariant
/// Returns `true` iff ∀ n ∈ N_R(v) ∩ dom(I): lower ≤ data[flat(n)] ≤ upper.
#[inline]
fn is_neighborhood_admissible(
    data: &[f32],
    dims: [usize; 3],
    voxel: [usize; 3],
    lower: f32,
    upper: f32,
    radius: [usize; 3],
) -> bool {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);

    // Compute clamped neighborhood bounds.
    let z_lo = voxel[0].saturating_sub(radius[0]);
    let z_hi = (voxel[0] + radius[0]).min(nz - 1);
    let y_lo = voxel[1].saturating_sub(radius[1]);
    let y_hi = (voxel[1] + radius[1]).min(ny - 1);
    let x_lo = voxel[2].saturating_sub(radius[2]);
    let x_hi = (voxel[2] + radius[2]).min(nx - 1);

    for iz in z_lo..=z_hi {
        let z_offset = iz * ny * nx;
        for iy in y_lo..=y_hi {
            let zy_offset = z_offset + iy * nx;
            for ix in x_lo..=x_hi {
                let val = data[zy_offset + ix];
                if val < lower || val > upper {
                    return false;
                }
            }
        }
    }

    true
}

/// Perform BFS flood fill with neighborhood admissibility on flat `[nz × ny × nx]` data.
///
/// Returns a flat binary `Vec<f32>` of the same length as `data`.
fn grow_neighborhood(
    data: &[f32],
    dims: [usize; 3],
    seed: [usize; 3],
    lower: f32,
    upper: f32,
    radius: [usize; 3],
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Check seed neighborhood admissibility.
    if !is_neighborhood_admissible(data, dims, seed, lower, upper, radius) {
        return vec![0.0_f32; n];
    }

    let mut output = vec![0.0_f32; n];
    let mut visited = vec![false; n];

    let seed_flat = flat(seed[0], seed[1], seed[2]);
    visited[seed_flat] = true;
    output[seed_flat] = 1.0;

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
        // Decode flat index to (z, y, x).
        let iz = curr_flat / (ny * nx);
        let rem = curr_flat % (ny * nx);
        let iy = rem / nx;
        let ix = rem % nx;

        // Expand to 6 face-adjacent neighbors.
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

            let nz_u = nz_i as usize;
            let ny_u = ny_i as usize;
            let nx_u = nx_i as usize;
            let n_flat = flat(nz_u, ny_u, nx_u);

            if visited[n_flat] {
                continue;
            }

            // Neighborhood admissibility: ALL voxels in N_R(candidate) must be in [lower, upper].
            if !is_neighborhood_admissible(data, dims, [nz_u, ny_u, nx_u], lower, upper, radius) {
                continue;
            }

            visited[n_flat] = true;
            output[n_flat] = 1.0;
            queue.push_back(n_flat);
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
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
        // All voxels intensity 100; bounds [50, 150]; radius [1,1,1].
        // Every voxel's 3×3×3 neighborhood is uniform → all admissible.
        // Entire 4×4×4 volume is grown.
        let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
        let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            64,
            "uniform region must grow entire volume"
        );
    }

    #[test]
    fn test_radius_zero_equivalent_to_connected_threshold() {
        // With radius [0,0,0], the neighborhood is the voxel itself.
        // This degenerates to the connected-threshold predicate.
        let values = vec![
            100.0, 100.0, 100.0, //
            100.0, 100.0, 100.0, //
            100.0, 100.0, 100.0, //
            50.0, 50.0, 50.0, //
            50.0, 50.0, 50.0, //
            50.0, 50.0, 50.0, //
            100.0, 100.0, 100.0, //
            100.0, 100.0, 100.0, //
            100.0, 100.0, 100.0, //
        ];
        let image = make_image(values, [3, 3, 3]);
        // Seed at (0,0,0) intensity=100, bounds [80,120]. Region z=0 and z=2
        // have intensity 100; z=1 has intensity 50 (outside bounds).
        // With radius=0, single-voxel predicate: z=0 plane (9 voxels) grows,
        // z=1 blocks, z=2 unreachable.
        let result = neighborhood_connected(&image, [0, 0, 0], 80.0, 120.0, [0, 0, 0]);
        assert_eq!(
            count_foreground(&result),
            9,
            "radius=0 must behave as connected-threshold"
        );
    }

    #[test]
    fn test_two_regions_seed_selects_one() {
        // 1×1×6 volume: [100, 100, 100, 10, 10, 10].
        // Seed at (0,0,0) with bounds [50, 200], radius [0,0,0].
        // Only first 3 voxels qualify.
        let values = vec![100.0, 100.0, 100.0, 10.0, 10.0, 10.0];
        let image = make_image(values, [1, 1, 6]);
        let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 200.0, [0, 0, 0]);
        let vals = get_values(&result);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 1.0);
        assert_eq!(vals[3], 0.0);
        assert_eq!(vals[4], 0.0);
        assert_eq!(vals[5], 0.0);
    }

    #[test]
    fn test_neighborhood_rejects_noisy_voxel() {
        // 3×3×3 uniform intensity 100, except one voxel at (0,1,1)=5.0 (noise).
        // Seed at (1,1,1) center. Bounds [80, 120], radius [1,1,1].
        //
        // Voxel (0,1,1) itself is out of bounds → it fails the neighborhood check.
        // Additionally, any voxel whose 3×3×3 neighborhood includes (0,1,1) also
        // fails, because that neighbor has intensity 5.0 < 80.
        //
        // The noisy voxel at (0,1,1) affects neighbors at offsets within radius 1
        // in each axis. The affected voxels are those whose neighborhood includes
        // (0,1,1), i.e., voxels v with |vz-0|≤1, |vy-1|≤1, |vx-1|≤1.
        // That is z∈{0,1}, y∈{0,1,2}, x∈{0,1,2} → 2×3×3 = 18 voxels affected.
        // Total admissible = 27 - 18 = 9 voxels (the z=2 plane).
        // But the seed (1,1,1) is in the affected zone → seed itself is NOT admissible.
        let mut values = vec![100.0_f32; 27];
        values[0 * 9 + 1 * 3 + 1] = 5.0; // (0,1,1) = noise
        let image = make_image(values, [3, 3, 3]);
        let result = neighborhood_connected(&image, [1, 1, 1], 80.0, 120.0, [1, 1, 1]);
        // Seed's neighborhood includes (0,1,1) which has intensity 5.0 < 80.
        // Therefore seed is not admissible → empty mask.
        assert_eq!(
            count_foreground(&result),
            0,
            "noisy voxel in seed's neighborhood must cause empty result"
        );
    }

    #[test]
    fn test_neighborhood_radius_restricts_boundary() {
        // 5×5×5 image: interior 3×3×3 cube at center has intensity 200,
        // outer shell has intensity 50. Bounds [150, 255].
        //
        // With radius [0,0,0]: the 3×3×3 interior cube (27 voxels) is grown.
        // With radius [1,1,1]: voxels on the surface of the 3×3×3 cube have
        // neighborhoods that extend into the shell (intensity 50 < 150),
        // so they fail the neighborhood check. Only the single center voxel
        // (2,2,2) has a 3×3×3 neighborhood entirely within the 3×3×3 cube.
        let (nz, ny, nx) = (5, 5, 5);
        let mut values = vec![50.0_f32; nz * ny * nx];
        for iz in 1..4 {
            for iy in 1..4 {
                for ix in 1..4 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
        let image = make_image(values, [nz, ny, nx]);

        // radius [0,0,0]: connected-threshold behavior, captures entire cube.
        let result_r0 = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [0, 0, 0]);
        assert_eq!(
            count_foreground(&result_r0),
            27,
            "radius=0 must capture entire 3×3×3 cube"
        );

        // radius [1,1,1]: only the center voxel (2,2,2) has a full 3×3×3
        // neighborhood inside the high-intensity cube.
        let result_r1 = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result_r1),
            1,
            "radius=1 must capture only center voxel of 3×3×3 cube in 5×5×5 image"
        );
    }

    #[test]
    fn test_larger_interior_with_radius() {
        // 7×7×7 image: interior 5×5×5 cube (indices 1..6) has intensity 200,
        // outer shell has intensity 50. Bounds [150, 255].
        //
        // With radius [1,1,1]: voxels whose 3×3×3 neighborhoods are entirely
        // within the 5×5×5 cube are those at indices 2..5 in each axis → 3×3×3 = 27.
        let (nz, ny, nx) = (7, 7, 7);
        let mut values = vec![50.0_f32; nz * ny * nx];
        for iz in 1..6 {
            for iy in 1..6 {
                for ix in 1..6 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
        let image = make_image(values, [nz, ny, nx]);
        let result = neighborhood_connected(&image, [3, 3, 3], 150.0, 255.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            27,
            "radius=1 on 5×5×5 interior in 7×7×7 must capture 3×3×3 = 27 voxels"
        );
    }

    // ── Negative / boundary tests ─────────────────────────────────────────────

    #[test]
    fn test_seed_neighborhood_outside_range_returns_empty() {
        // Seed intensity = 100, but neighbor at seed-1 has intensity 5 < 50.
        // With radius [1,1,1], seed neighborhood includes the bad voxel → empty.
        let mut values = vec![100.0_f32; 8]; // 2×2×2
        values[0] = 5.0; // (0,0,0) low intensity
        let image = make_image(values, [2, 2, 2]);
        // Seed at (0,0,1): neighborhood includes (0,0,0) which has val 5.0 < 50.
        let result = neighborhood_connected(&image, [0, 0, 1], 50.0, 150.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            0,
            "seed with bad neighborhood must produce empty region"
        );
    }

    #[test]
    fn test_seed_voxel_itself_below_lower_returns_empty() {
        // Even with radius [0,0,0], if the seed intensity is below lower → empty.
        let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
        let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 200.0, [0, 0, 0]);
        assert_eq!(count_foreground(&result), 0);
    }

    #[test]
    fn test_seed_voxel_itself_above_upper_returns_empty() {
        let image = make_image(vec![201.0_f32; 1], [1, 1, 1]);
        let result = neighborhood_connected(&image, [0, 0, 0], 0.0, 200.0, [0, 0, 0]);
        assert_eq!(count_foreground(&result), 0);
    }

    #[test]
    fn test_strictly_binary_output() {
        let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
        let result = neighborhood_connected(&image, [1, 1, 1], 50.0, 150.0, [1, 1, 1]);
        for &v in get_values(&result).iter() {
            assert!(
                v == 0.0 || v == 1.0,
                "output must be strictly binary, got {v}"
            );
        }
    }

    #[test]
    fn test_connectivity_is_6_not_diagonal() {
        // 3×3×1 slice:
        //   A 0 0
        //   0 0 0
        //   0 0 B
        // A and B have high intensity; all others low.
        // With 6-connectivity, seeding from A cannot reach B diagonally.
        let mut values = vec![0.0_f32; 9];
        values[0] = 200.0; // A at (0,0,0)
        values[8] = 200.0; // B at (0,2,2)
        let image = make_image(values, [1, 3, 3]);
        let result = neighborhood_connected(&image, [0, 0, 0], 100.0, 255.0, [0, 0, 0]);
        let vals = get_values(&result);
        assert_eq!(vals[0], 1.0, "seed voxel A must be foreground");
        assert_eq!(vals[8], 0.0, "diagonal voxel B must not be reached");
        assert_eq!(count_foreground(&result), 1);
    }

    // ── Structural tests ──────────────────────────────────────────────────────

    #[test]
    fn test_spatial_metadata_preserved() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![100.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);

        let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [1, 1, 1]);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
    }

    #[test]
    fn test_filter_struct_matches_function() {
        let values: Vec<f32> = (0..27).map(|i| i as f32 * 10.0).collect();
        let image = make_image(values, [3, 3, 3]);

        let via_fn = neighborhood_connected(&image, [1, 1, 1], 50.0, 200.0, [1, 1, 1]);
        let via_struct = NeighborhoodConnectedFilter::new([1, 1, 1], 50.0, 200.0)
            .with_radius([1, 1, 1])
            .apply(&image);

        let fn_vals = get_values(&via_fn);
        let struct_vals = get_values(&via_struct);
        assert_eq!(
            fn_vals, struct_vals,
            "function and filter struct must produce identical results"
        );
    }

    #[test]
    fn test_filter_struct_default_radius() {
        // Default radius is [1,1,1]. Verify builder without explicit radius matches.
        let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);

        let via_fn = neighborhood_connected(&image, [1, 1, 1], 50.0, 150.0, [1, 1, 1]);
        let via_struct = NeighborhoodConnectedFilter::new([1, 1, 1], 50.0, 150.0).apply(&image);

        assert_eq!(
            get_values(&via_fn),
            get_values(&via_struct),
            "default radius must be [1,1,1]"
        );
    }

    // ── Anisotropic radius test ───────────────────────────────────────────────

    #[test]
    fn test_anisotropic_radius() {
        // 1×5×5 image: center 1×3×3 has intensity 200, border intensity 50.
        // Bounds [150, 255].
        //
        // With radius [0,1,1]: neighborhood in z is just the voxel itself (nz=1),
        // in y and x extends ±1. The voxels whose 1×3×3 neighborhood is fully
        // inside the 1×3×3 high-intensity region are those at y=2,x=2 (center only).
        let (nz, ny, nx) = (1, 5, 5);
        let mut values = vec![50.0_f32; nz * ny * nx];
        for iy in 1..4 {
            for ix in 1..4 {
                values[iy * nx + ix] = 200.0;
            }
        }
        let image = make_image(values, [nz, ny, nx]);
        let result = neighborhood_connected(&image, [0, 2, 2], 150.0, 255.0, [0, 1, 1]);
        // Center voxel (0,2,2): neighborhood y∈{1,2,3}, x∈{1,2,3} → all 200.0 ✓
        // Its face-neighbors in-plane: (0,1,2),(0,3,2),(0,2,1),(0,2,3).
        // (0,1,2): neighborhood y∈{0,1,2}, x∈{1,2,3} → includes (0,0,{1,2,3})=50 ✗
        // Same reasoning for (0,3,2): includes y=4 which is 50. ✗
        // (0,2,1): neighborhood x∈{0,1,2} → includes x=0 which is 50. ✗
        // (0,2,3): neighborhood x∈{2,3,4} → includes x=4 which is 50. ✗
        // So only the center voxel is admissible.
        assert_eq!(
            count_foreground(&result),
            1,
            "anisotropic radius must correctly restrict admissibility"
        );
    }

    // ── 3-D volumetric test ───────────────────────────────────────────────────

    #[test]
    fn test_3d_sphere_region_growing_radius_zero() {
        // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
        // background intensity 50; bounds [150, 255]; neighborhood radius [0,0,0].
        // This should behave exactly like connected-threshold.
        let (nz, ny, nx) = (9, 9, 9);
        let mut values = vec![50.0_f32; nz * ny * nx];
        let (cz, cy, cx) = (4isize, 4isize, 4isize);
        let r2 = 9isize; // radius squared = 9

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
        let result = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [0, 0, 0]);
        assert_eq!(
            count_foreground(&result),
            sphere_count,
            "radius=0 sphere must match connected-threshold exactly"
        );
    }

    #[test]
    fn test_3d_sphere_neighborhood_erodes_boundary() {
        // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
        // background intensity 50; bounds [150, 255]; neighborhood radius [1,1,1].
        //
        // Sphere voxels whose 3×3×3 neighborhood extends outside the sphere will be
        // rejected (some neighbors have intensity 50). This effectively erodes the
        // sphere boundary: only interior voxels (where the entire 3×3×3 neighborhood
        // is also within the sphere) are admitted.
        //
        // A voxel at (z,y,x) is admissible iff all 27 neighbors in its 3×3×3 box
        // are also within the sphere. This is equivalent to requiring
        // (z±1, y±1, x±1) all satisfy d² ≤ 9 where d² is measured from center.
        // The worst case is the corner offset (+1,+1,+1) from the voxel.
        let (nz, ny, nx) = (9, 9, 9);
        let mut values = vec![50.0_f32; nz * ny * nx];
        let (cz, cy, cx) = (4isize, 4isize, 4isize);
        let r2 = 9isize;

        // Build sphere.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as isize - cz;
                    let dy = iy as isize - cy;
                    let dx = ix as isize - cx;
                    if dz * dz + dy * dy + dx * dx <= r2 {
                        values[iz * ny * nx + iy * nx + ix] = 200.0;
                    }
                }
            }
        }

        // Count analytically: voxel (z,y,x) is admissible iff for ALL (dz,dy,dx)
        // with |dz|≤1, |dy|≤1, |dx|≤1, the neighbor (z+dz, y+dy, x+dx) is within
        // the sphere (or outside the image domain — but at the center of a 9×9×9
        // image with sphere radius 3, no neighborhood extends outside domain).
        let mut expected_count = 0;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let mut all_in_sphere = true;
                    'outer: for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let nzi = iz as isize + dz;
                                let nyi = iy as isize + dy;
                                let nxi = ix as isize + dx;
                                if nzi < 0
                                    || nzi >= nz as isize
                                    || nyi < 0
                                    || nyi >= ny as isize
                                    || nxi < 0
                                    || nxi >= nx as isize
                                {
                                    // Outside domain: excluded from check.
                                    continue;
                                }
                                let v = values
                                    [nzi as usize * ny * nx + nyi as usize * nx + nxi as usize];
                                if v < 150.0 || v > 255.0 {
                                    all_in_sphere = false;
                                    break 'outer;
                                }
                            }
                        }
                    }
                    if all_in_sphere {
                        // Also must be 6-connected to the seed region.
                        // For a convex shape centered at (4,4,4), all admissible
                        // voxels form a 6-connected region — verified analytically
                        // for the Euclidean sphere with r²≤9.
                        // We check that the voxel is in the original sphere itself
                        // (otherwise it's background that happens to have all
                        // in-bounds neighbors in the sphere, which is not possible
                        // for interior background of a filled sphere).
                        let dz = iz as isize - cz;
                        let dy = iy as isize - cy;
                        let dx = ix as isize - cx;
                        if dz * dz + dy * dy + dx * dx <= r2 {
                            expected_count += 1;
                        }
                    }
                }
            }
        }

        let image = make_image(values, [nz, ny, nx]);
        let result = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            expected_count,
            "neighborhood radius must erode sphere boundary"
        );
        // Verify the eroded region is strictly smaller than the full sphere.
        let full_sphere = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [0, 0, 0]);
        assert!(
            count_foreground(&result) < count_foreground(&full_sphere),
            "radius=1 region must be strictly smaller than radius=0 region for a sphere"
        );
    }

    // ── Predicate correctness test ────────────────────────────────────────────

    #[test]
    fn test_is_neighborhood_admissible_direct() {
        // Direct unit test of the internal predicate function.
        // 3×3×3 uniform intensity 100.
        let data = vec![100.0_f32; 27];
        let dims = [3, 3, 3];

        // Center voxel, radius [1,1,1]: all 27 voxels are 100.0 ∈ [50, 150].
        assert!(is_neighborhood_admissible(
            &data,
            dims,
            [1, 1, 1],
            50.0,
            150.0,
            [1, 1, 1]
        ));

        // Corner voxel (0,0,0), radius [1,1,1]: neighborhood clamped to
        // z∈[0,1], y∈[0,1], x∈[0,1] → 8 voxels, all 100.0.
        assert!(is_neighborhood_admissible(
            &data,
            dims,
            [0, 0, 0],
            50.0,
            150.0,
            [1, 1, 1]
        ));

        // Bounds exclude all voxels: lower=200 > all intensities.
        assert!(!is_neighborhood_admissible(
            &data,
            dims,
            [1, 1, 1],
            200.0,
            300.0,
            [1, 1, 1]
        ));

        // Single bad voxel in neighborhood.
        let mut data_bad = vec![100.0_f32; 27];
        data_bad[0] = 5.0; // (0,0,0) is bad
                           // Voxel (1,1,1) with radius [1,1,1] includes (0,0,0) → not admissible.
        assert!(!is_neighborhood_admissible(
            &data_bad,
            dims,
            [1, 1, 1],
            50.0,
            150.0,
            [1, 1, 1]
        ));
        // Voxel (2,2,2) with radius [1,1,1]: neighborhood is z∈[1,2], y∈[1,2], x∈[1,2].
        // Does NOT include (0,0,0) → admissible.
        assert!(is_neighborhood_admissible(
            &data_bad,
            dims,
            [2, 2, 2],
            50.0,
            150.0,
            [1, 1, 1]
        ));
    }

    // ── Adversarial tests ─────────────────────────────────────────────────────

    /// Two isolated high-intensity cubes separated by zero-intensity background.
    /// Seeding from each cube must not cross-contaminate the other.
    ///
    /// Image: 1×1×8 = [100, 100, 100, 0, 0, 200, 200, 200]
    /// Seed A at [0,0,0] with bounds [50, 150], radius [0,0,0] → 3 voxels.
    /// Seed B at [0,0,5] with bounds [150, 250], radius [0,0,0] → 3 voxels.
    #[test]
    fn test_multi_seed_two_cubes_no_cross_contamination() {
        let values = vec![100.0_f32, 100.0, 100.0, 0.0, 0.0, 200.0, 200.0, 200.0];
        let image = make_image(values, [1, 1, 8]);

        let result_a = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [0, 0, 0]);
        assert_eq!(
            count_foreground(&result_a),
            3,
            "seed in cube A must grow exactly 3 voxels"
        );
        let vals_a = get_values(&result_a);
        assert_eq!(
            vals_a[5], 0.0,
            "cube B position 5 must not be reached from seed A"
        );
        assert_eq!(
            vals_a[7], 0.0,
            "cube B position 7 must not be reached from seed A"
        );

        let result_b = neighborhood_connected(&image, [0, 0, 5], 150.0, 250.0, [0, 0, 0]);
        assert_eq!(
            count_foreground(&result_b),
            3,
            "seed in cube B must grow exactly 3 voxels"
        );
        let vals_b = get_values(&result_b);
        assert_eq!(
            vals_b[0], 0.0,
            "cube A position 0 must not be reached from seed B"
        );
        assert_eq!(
            vals_b[2], 0.0,
            "cube A position 2 must not be reached from seed B"
        );
    }

    /// Seed at volume corner [0,0,0] with radius [2,2,2] on a 3×3×3 uniform image.
    /// The neighborhood is clamped to [0, min(2, nDim-1)] in each axis.
    /// Since all voxels are uniform and within bounds, the entire 27-voxel volume
    /// must be grown (clamping does not prevent growth; it restricts neighborhood check).
    /// Analytical proof: for any voxel v with all neighbors in [0, 150], P(v) holds
    /// for any radius as long as the domain-clamped neighborhood stays in [50, 150].
    #[test]
    fn test_boundary_seed_radius_overflow_clamped_to_domain() {
        // 3×3×3 uniform intensity 100; bounds [50, 150]; radius [2,2,2].
        // Radius [2,2,2] on a 3×3×3 image: neighborhood clamped to [0,2] × [0,2] × [0,2]
        // = entire image. All voxels are 100 ∈ [50, 150] → all admissible → all grown.
        let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
        let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [2, 2, 2]);
        assert_eq!(
            count_foreground(&result),
            27,
            "radius [2,2,2] clamped to 3×3×3 domain must grow all 27 voxels"
        );
    }

    /// Large uniform image with large radius: every voxel's neighborhood is
    /// fully within bounds, so the entire volume is grown from any seed.
    ///
    /// 6×6×6 image, all intensity 100, bounds [50, 150], radius [1,1,1].
    /// Every voxel's 3×3×3 (clamped) neighborhood is all-100 ∈ [50, 150].
    /// All 216 voxels must be grown.
    #[test]
    fn test_large_uniform_image_large_radius_grows_all_voxels() {
        let image = make_image(vec![100.0_f32; 216], [6, 6, 6]);
        let result = neighborhood_connected(&image, [3, 3, 3], 50.0, 150.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            216,
            "6×6×6 uniform image with radius [1,1,1] must grow all 216 voxels"
        );
    }

    /// Adversarial: noisy background at image boundary; seed deep inside uniform
    /// interior; large radius must still reject boundary voxels that touch noise.
    ///
    /// 5×5×5 image: interior 3×3×3 cube (indices 1..4) = 200; boundary shell = 5 (noise).
    /// Bounds [150, 255], radius [1,1,1].
    ///
    /// With radius [1,1,1]: voxels on the surface of the 3×3×3 cube have neighborhoods
    /// that extend into the shell (intensity 5 < 150) → rejected.
    /// Only the center voxel (2,2,2) whose 3×3×3 neighborhood is entirely within the
    /// high-intensity cube is admissible → 1 voxel grown.
    /// This is the adversarial equivalent of test_neighborhood_radius_restricts_boundary
    /// but with noise intensity ≪ lower bound (5 vs 50), stressing the boundary rejection.
    #[test]
    fn test_adversarial_noisy_boundary_large_radius_rejects_surface_voxels() {
        let (nz, ny, nx) = (5, 5, 5);
        let mut values = vec![5.0_f32; nz * ny * nx]; // extreme noise in boundary shell
        for iz in 1..4 {
            for iy in 1..4 {
                for ix in 1..4 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
        let image = make_image(values, [nz, ny, nx]);
        let result = neighborhood_connected(&image, [2, 2, 2], 150.0, 255.0, [1, 1, 1]);
        assert_eq!(
            count_foreground(&result),
            1,
            "only center voxel (2,2,2) must be grown when boundary shell contains noise"
        );
        // Verify the center voxel is foreground.
        let vals = get_values(&result);
        let center_flat = 2 * ny * nx + 2 * nx + 2;
        assert_eq!(
            vals[center_flat], 1.0,
            "center voxel (2,2,2) must be foreground"
        );
    }
}

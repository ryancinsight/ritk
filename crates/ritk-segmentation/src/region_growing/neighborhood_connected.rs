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
//! 2. Initialize BFS queue Q ← {s}, visited\[s\] ← true, output\[s\] ← 1.
//! 3. While Q ≠ ∅:
//! a. Dequeue p from Q.
//! b. For each q ∈ N₆(p) (6 face-adjacent neighbors):
//! - If q ∉ visited and P(q): visited\[q\] ← true, output\[q\] ← 1, enqueue q.
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

use ritk_core::spatial::VoxelIndex;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
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
    seed: VoxelIndex,
    /// Inclusive lower intensity bound.
    lower: f32,
    /// Inclusive upper intensity bound.
    upper: f32,
    /// Rectangular neighborhood half-radius [rz, ry, rx].
    ///
    /// The full neighborhood for a voxel v is the set of all voxels n
    /// with |nᵢ − vᵢ| ≤ Rᵢ. Default: [1, 1, 1] (3×3×3 = 27 voxels).
    radius: [usize; 3],
}

impl NeighborhoodConnectedFilter {
    /// Create a `NeighborhoodConnectedFilter` with default radius [1, 1, 1].
    ///
    /// # Panics
    /// Panics if either bound is NaN or `lower > upper`.
    pub fn new(seed: impl Into<VoxelIndex>, lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "lower bound {lower} must be ≤ upper bound {upper}"
        );
        Self {
            seed: seed.into(),
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

    /// Return the seed voxel.
    pub fn seed(&self) -> VoxelIndex {
        self.seed
    }

    /// Return the inclusive lower bound.
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Return the inclusive upper bound.
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Return the rectangular neighborhood half-radius.
    pub fn radius(&self) -> [usize; 3] {
        self.radius
    }

    /// Apply neighborhood-connected growth to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the seed is outside the image, backend storage is
    /// not host-addressable, or the output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let shape = image.shape();
        anyhow::ensure!(
            self.seed[0] < shape[0] && self.seed[1] < shape[1] && self.seed[2] < shape[2],
            "seed {:?} is out of bounds for image shape {:?}",
            self.seed.as_array(),
            shape
        );
        crate::native_output::from_values(
            image,
            grow_neighborhood(
                image.data_slice()?,
                shape,
                self.seed,
                self.lower,
                self.upper,
                self.radius,
            ),
            backend,
        )
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
    seed: impl Into<VoxelIndex>,
    lower: f32,
    upper: f32,
    radius: [usize; 3],
) -> Image<B, 3> {
    assert!(
        lower <= upper,
        "lower bound {lower} must be ≤ upper bound {upper}"
    );

    let seed = seed.into();
    let shape = image.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);

    assert!(
        seed[0] < nz && seed[1] < ny && seed[2] < nx,
        "seed {:?} is out of bounds for image shape {:?}",
        seed.as_array(),
        shape
    );

    let (img_vals, shape) = extract_vec_infallible(image);
    let device = image.data().device();

    let result = grow_neighborhood(&img_vals, shape, seed, lower, upper, radius);

    let td = TensorData::new(result, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);

    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
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
    let z_hi = voxel[0].saturating_add(radius[0]).min(nz - 1);
    let y_lo = voxel[1].saturating_sub(radius[1]);
    let y_hi = voxel[1].saturating_add(radius[1]).min(ny - 1);
    let x_lo = voxel[2].saturating_sub(radius[2]);
    let x_hi = voxel[2].saturating_add(radius[2]).min(nx - 1);

    for iz in z_lo..=z_hi {
        let z_offset = iz * ny * nx;
        for iy in y_lo..=y_hi {
            let zy_offset = z_offset + iy * nx;
            for ix in x_lo..=x_hi {
                let val = data[zy_offset + ix];
                if !super::intensity::within_finite_bounds(val, lower, upper) {
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
pub(crate) fn grow_neighborhood(
    data: &[f32],
    dims: [usize; 3],
    seed: VoxelIndex,
    lower: f32,
    upper: f32,
    radius: [usize; 3],
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Check seed neighborhood admissibility.
    if !is_neighborhood_admissible(data, dims, seed.into(), lower, upper, radius) {
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
#[path = "tests_neighborhood_connected/mod.rs"]
mod tests_neighborhood_connected;

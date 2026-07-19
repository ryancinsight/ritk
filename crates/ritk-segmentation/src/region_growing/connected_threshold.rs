//! Connected-threshold region growing for 3-D images.
//!
//! Implements fixed-intensity-bounds BFS flood-fill region growing seeded
//! from a single voxel.

use ritk_core::spatial::VoxelIndex;
use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::VecDeque;

// ── Public type ───────────────────────────────────────────────────────────────

/// Connected-threshold region-growing filter.
///
/// Segments the region reachable from a seed voxel whose intensities fall
/// within the closed interval [lower, upper].
pub struct ConnectedThresholdFilter {
    /// Seed voxel in [z, y, x] index space.
    seed: VoxelIndex,
    /// Inclusive lower intensity bound.
    lower: f32,
    /// Inclusive upper intensity bound.
    upper: f32,
}

impl ConnectedThresholdFilter {
    /// Create a `ConnectedThresholdFilter`.
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
        }
    }

    /// Apply region growing to `image`.
    ///
    /// Returns a binary mask (values in {0.0, 1.0}) with the same shape and
    /// spatial metadata as `image`.  If the seed voxel's intensity does not
    /// satisfy `lower ≤ I(seed) ≤ upper`, the output is all-zero.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        connected_threshold(image, self.seed, self.lower, self.upper)
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

    /// Apply region growing to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the seed is outside the image, backend storage is
    /// not host-addressable, or the output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
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
            flood_fill(
                image.data_slice()?,
                shape,
                self.seed,
                self.lower,
                self.upper,
            ),
            backend,
        )
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
    image: &Image<f32, B, 3>,
    seed: impl Into<VoxelIndex>,
    lower: f32,
    upper: f32,
) -> Image<f32, B, 3> {
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

    let (img_slice_vec, _) = extract_vec_infallible(image);
    let img_slice: &[f32] = &img_slice_vec;

    let result = flood_fill(img_slice, shape, seed, lower, upper);

    let tensor = Tensor::<f32, B>::from_slice(shape, &result);

    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
    .expect("invariant: segmentation output tensor preserves the image rank")
}

// ── Core BFS flood fill ───────────────────────────────────────────────────────

/// Perform BFS flood fill on a flat `[nz × ny × nx]` intensity slice.
///
/// Returns a flat binary `Vec<f32>` of the same length as `data`.
pub(crate) fn flood_fill(
    data: &[f32],
    dims: [usize; 3],
    seed: VoxelIndex,
    lower: f32,
    upper: f32,
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Check seed intensity.
    let seed_val = data[flat(seed[0], seed[1], seed[2])];
    if !super::intensity::within_finite_bounds(seed_val, lower, upper) {
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
            if !super::intensity::within_finite_bounds(intensity, lower, upper) {
                continue;
            }

            visited[n_flat] = true;
            queue.push_back(n_flat);
        }
    }

    output
}

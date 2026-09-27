//! Sliding-window median filter for 3-D volumes.
//!
//! # Algorithm
//! For each voxel `(iz, iy, ix)`, collect all values in the axis-aligned cube
//! `[iz ± r, iy ± r, ix ± r]` using replicate (clamp) boundary conditions,
//! sort them, and take the middle element (lower median for even-length
//! neighbourhoods, consistent with standard medical-imaging toolkits).
//!
//! # Complexity
//! O(n * (2r+1)^3) average where n is the total voxel count and r is
//! the neighbourhood half-width, using introselect (select_nth_unstable_by)
//! rather than a full sort. Parallelised over z-slices via moirai's
//! `for_each_chunk_mut_enumerated_with` on the `Adaptive` execution policy.

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Sliding-window median filter for 3-D volumes.
///
/// Replaces each voxel with the median of its `(2r+1)³` axis-aligned
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
pub struct MedianFilter {
    /// Neighbourhood half-width in voxels (default 1 → 3×3×3 cube).
    pub radius: usize,
}

impl MedianFilter {
    /// Create a new median filter with the given neighbourhood half-width.
    ///
    /// A radius of 0 yields identity (each voxel is its own sole neighbour).
    /// A radius of 1 produces a 3×3×3 kernel (27 samples per voxel).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply the median filter to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, shape) = extract_vec(image)?;

        let filtered = median_3d(&vals, shape, self.radius);

        Ok(rebuild(filtered, shape, image))
    }

    /// Coeus-native sister of [`MedianFilter::apply`].
    ///
    /// Runs the identical sliding-window lower-median (replicate boundary) via
    /// the shared `median_3d` host core on the image's contiguous host buffer,
    /// so the result is bitwise-identical to the Coeus path. No tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let filtered = median_3d(&vals, dims, self.radius);
        ritk_tensor_ops::native::rebuild_image(filtered, dims, image, &B::default())
    }
}

/// Sliding-window median on a 3-D volume stored in flat Z×Y×X order.
///
/// # Arguments
/// * `data`   — flat voxel values in row-major (Z-major) order.
/// * `dims`   — `[nz, ny, nx]`.
/// * `radius` — neighbourhood half-width in voxels.
///
/// # Boundary handling
/// Replicate (clamp) padding: out-of-bounds indices are clamped to the
/// nearest valid index along each axis.
fn median_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let r = radius as isize;
    let cap = (2 * radius + 1).pow(3);
    let nz_isize = nz as isize;
    let ny_isize = ny as isize;
    let nx_isize = nx as isize;
    let stride_yx = ny * nx;
    let mut output = vec![0.0_f32; nz * ny * nx];

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        ny * nx,
        |iz, out_slice| {
            // Allocate once per z-slice (per parallel worker); reused across
            // all voxels in the slice via clear() to avoid per-voxel heap
            // allocation.
            let mut neighbors: Vec<f32> = Vec::with_capacity(cap);

            // Pre-clamp the Z-plane once per voxel-row: each `dz` maps to a
            // single `zz` regardless of `iy, ix`. Hoisting removes a
            // `(2r+1)²`-fold redundant branch per voxel (PERF-377-01 partial).
            const BUF_CAP: usize = 64;
            // The clamp-buffer holds `2 * radius + 1` clamped indices per axis.
            // Capacity 64 supports radii up to 31 (test-only envelope — production
            // radii are ≫ 8). A larger radius panics here to keep the hot path
            // stack-allocated.
            assert!(
                2 * radius < BUF_CAP,
                "MedianFilter::median_3d: radius {radius} exceeds buffer cap (max 31)"
            );
            let mut zz_buf: [usize; BUF_CAP] = [0; BUF_CAP];
            #[expect(clippy::needless_range_loop, reason = "ratchet RITK-LINT-1")]
            for dz in 0..=(2 * radius) {
                let z_raw = (iz as isize + dz as isize - r).clamp(0, nz_isize - 1);
                zz_buf[dz] = z_raw as usize;
            }

            if radius == 0 {
                // Single sample per voxel — neighborhood is just the voxel.
                for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                    for (ix, cell) in out_row.iter_mut().enumerate() {
                        let idx = iz * stride_yx + iy * nx + ix;
                        *cell = data[idx];
                    }
                }
                return;
            }

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                // Pre-clamp the Y-row once per iy: `(2r+1)²`-fold redundant
                // branch elimination when paired with the dz-hoist above.
                let mut yy_buf: [usize; BUF_CAP] = [0; BUF_CAP];
                #[expect(clippy::needless_range_loop, reason = "ratchet RITK-LINT-1")]
                for dy in 0..=(2 * radius) {
                    let y_raw = (iy as isize + dy as isize - r).clamp(0, ny_isize - 1);
                    yy_buf[dy] = y_raw as usize;
                }

                for (ix, cell) in out_row.iter_mut().enumerate() {
                    neighbors.clear();

                    // Collect (2r+1)^3 neighbourhood with replicate (clamp)
                    // padding. The per-axis clamp results are pre-baked in
                    // `zz_buf` / `yy_buf`; only the X-axis clamp is computed
                    // per inner-tick (it depends on `ix`).
                    //
                    // The triple-nested `dz`, `dy`, `dx` loops use indices
                    // to drive the geometry of the cubic neighbourhood;
                    // the iterator-with-enumerate transformation adds a
                    // bounds-check on every tick without paying for the
                    // inner clamp hoist. Single block-level allow per the
                    // precedent set by morphology::window_1d.
                    #[expect(clippy::needless_range_loop, reason = "ratchet RITK-LINT-1")]
                    for dz in 0..=(2 * radius) {
                        let zz_base = zz_buf[dz] * stride_yx;
                        #[expect(clippy::needless_range_loop, reason = "ratchet RITK-LINT-1")]
                        for dy in 0..=(2 * radius) {
                            let yy_base = yy_buf[dy] * nx;
                            let base = zz_base + yy_base;
                            for dx in 0..=(2 * radius) {
                                let xx =
                                    (ix as isize + dx as isize - r).clamp(0, nx_isize - 1) as usize;
                                neighbors.push(data[base + xx]);
                            }
                        }
                    }

                    // Lower median for even-length neighbourhoods.
                    // select_nth_unstable_by is O(N) average (introselect) versus
                    // O(N log N) for a full sort; the value at neighbors[mid] after
                    // the call is identical to sort-then-index.
                    let mid = neighbors.len() / 2;
                    neighbors.select_nth_unstable_by(mid, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *cell = neighbors[mid];
                }
            }
        },
    );

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_median_native.rs"]
mod tests_native;

#[cfg(test)]
mod tests;

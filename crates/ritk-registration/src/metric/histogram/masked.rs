use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

use super::parzen::ParzenJointHistogram;

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute joint histogram using a pre-selected set of fixed-image world coordinates.
    ///
    /// This is the brain-masked variant: instead of uniform random sampling, only
    /// foreground voxels (supplied by the caller in world space) contribute to the
    /// histogram. This prevents background intensities from dominating the MI
    /// landscape during CT↔MRI registration.
    ///
    /// # Arguments
    /// * `fixed` — Fixed reference image.
    /// * `fixed_world_points` — `[N, D]` world-space coordinates of the foreground voxels.
    /// * `moving` — Moving image.
    /// * `transform` — Current candidate transform (moving → fixed space mapping).
    /// * `interpolator` — Interpolator for sampling the moving image (zero-pad recommended).
    pub fn compute_masked_joint_histogram<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        fixed_world_points: Tensor<B, 2>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
    ) -> Tensor<B, 2> {
        let n = fixed_world_points.dims()[0];
        let device = fixed_world_points.device();

        if n == 0 {
            // Degenerate: empty mask — return zero histogram.
            return Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);
        }

        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
            // ── Non-chunked path ──────────────────────────────────────────────
            // Convert fixed world coords → fixed voxel indices, then sample.
            let fixed_voxel_indices = fixed.world_to_index_tensor(fixed_world_points.clone());
            let fixed_values = interpolator.interpolate(fixed.data(), fixed_voxel_indices);

            // Apply transform to get moving world coords, then sample moving image.
            let moving_world_points = transform.transform_points(fixed_world_points);
            let moving_voxel_indices = moving.world_to_index_tensor(moving_world_points);
            // Compute OOB mask before consuming moving_voxel_indices (uses immutable borrow).
            let oob_mask: Option<Tensor<B, 1>> = if D == 3 {
                let shape_arr = moving.shape();
                Some(super::parzen::compute_oob_mask_3d(&moving_voxel_indices, shape_arr.as_ref()))
            } else {
                None
            };
            let moving_values = interpolator.interpolate(moving.data(), moving_voxel_indices);

            self.compute_joint_histogram(&fixed_values, &moving_values, oob_mask.as_ref())
        } else {
            // ── Chunked path ──────────────────────────────────────────────────
            let num_chunks = n.div_ceil(CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);

                let chunk_fixed_world = fixed_world_points.clone().slice([start..end]);
                let chunk_fixed_idx = fixed.world_to_index_tensor(chunk_fixed_world.clone());
                let chunk_fixed_vals = interpolator.interpolate(fixed.data(), chunk_fixed_idx);

                let chunk_moving_world = transform.transform_points(chunk_fixed_world);
                let chunk_moving_idx = moving.world_to_index_tensor(chunk_moving_world);
                // Compute per-chunk OOB mask before consuming chunk_moving_idx.
                let chunk_oob: Option<Tensor<B, 1>> = if D == 3 {
                    let shape_arr = moving.shape();
                    Some(super::parzen::compute_oob_mask_3d(&chunk_moving_idx, shape_arr.as_ref()))
                } else {
                    None
                };
                let chunk_moving_vals = interpolator.interpolate(moving.data(), chunk_moving_idx);

                joint_hist_acc = joint_hist_acc
                    + self.compute_joint_histogram(&chunk_fixed_vals, &chunk_moving_vals, chunk_oob.as_ref());
            }

            joint_hist_acc
        }
    }
}

//! Chunked helper methods for the masked joint histogram path.
//!
//! Extracted from `masked.rs` to keep the main file under the 500-line
//! structural limit. Contains the chunked dense-cache and sparse-cache
//! histogram accumulation loops.

use crate::metric::histogram::parzen::ParzenJointHistogram;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_interpolation::{Interpolator, LinearInterpolator};

#[cfg(feature = "direct-parzen")]
use crate::metric::histogram::parzen::direct::SparseSampleCache;

impl<B: Backend> ParzenJointHistogram<B> {
    /// Chunked histogram computation using a cached dense W_fixed^T.
    ///
    /// Slices the cached `[num_bins, N]` tensor per chunk to avoid recomputing
    /// fixed-image Parzen weights.
    pub(in crate::metric::histogram) fn compute_masked_chunked_from_dense_cache<const D: usize>(
        &self,
        full_w_fixed_t: &Tensor<B, 2>,
        _fixed: &Image<B, D>,
        fixed_world_points: &Tensor<B, 2>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
    ) -> Tensor<B, 2> {
        let n = fixed_world_points.dims()[0];
        let device = fixed_world_points.device();
        let num_chunks = n.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
        let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

        for i in 0..num_chunks {
            let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
            let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n);
            let chunk_w_fixed_t = full_w_fixed_t.clone().slice([0..self.num_bins, start..end]);
            #[allow(clippy::single_range_in_vec_init)]
            let chunk_fixed_world = fixed_world_points.clone().slice([start..end]);
            let (chunk_moving_vals, chunk_oob) =
                sample_moving_chunk(&chunk_fixed_world, moving, transform, interpolator);
            joint_hist_acc = joint_hist_acc
                + self.compute_joint_histogram_from_cache_dispatch(
                    &chunk_w_fixed_t,
                    &chunk_moving_vals,
                    chunk_oob.as_ref(),
                );
        }
        joint_hist_acc
    }

    /// Chunked histogram computation using a cached sparse W_fixed^T.
    ///
    /// Slices the sparse cache per chunk (sub-Vec of entries) to avoid
    /// recomputing fixed-image Parzen weights.
    #[cfg(feature = "direct-parzen")]
    pub(in crate::metric::histogram) fn compute_masked_chunked_from_sparse_cache<const D: usize>(
        &self,
        sparse_w_fixed: &[(SparseSampleCache, f32)],
        _fixed: &Image<B, D>,
        fixed_world_points: &Tensor<B, 2>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
    ) -> Tensor<B, 2> {
        let n = fixed_world_points.dims()[0];
        let device = fixed_world_points.device();
        let num_chunks = n.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
        let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

        for i in 0..num_chunks {
            let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
            let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n);
            let chunk_sparse = &sparse_w_fixed[start..end];
            #[allow(clippy::single_range_in_vec_init)]
            let chunk_fixed_world = fixed_world_points.clone().slice([start..end]);
            let (chunk_moving_vals, chunk_oob) =
                sample_moving_chunk(&chunk_fixed_world, moving, transform, interpolator);
            joint_hist_acc = joint_hist_acc
                + self.compute_joint_histogram_from_cache_sparse_dispatch(
                    chunk_sparse,
                    &chunk_moving_vals,
                    chunk_oob.as_ref(),
                );
        }
        joint_hist_acc
    }
}

/// Sample moving image values and OOB mask for a chunk of world-space points.
fn sample_moving_chunk<B: Backend, const D: usize>(
    chunk_fixed_world: &Tensor<B, 2>,
    moving: &Image<B, D>,
    transform: &impl Transform<B, D>,
    interpolator: &LinearInterpolator,
) -> (Tensor<B, 1>, Option<Tensor<B, 1>>) {
    let chunk_moving_world = transform.transform_points(chunk_fixed_world.clone());
    let chunk_moving_idx = moving.world_to_index_tensor(chunk_moving_world);
    let oob = if D == 3 {
        Some(crate::metric::histogram::parzen::compute_oob_mask(
            &chunk_moving_idx,
            moving.shape().as_ref(),
        ))
    } else {
        None
    };
    let vals = interpolator.interpolate(moving.data(), chunk_moving_idx);
    (vals, oob)
}

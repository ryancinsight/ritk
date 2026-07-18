//! Cached masked-histogram dispatch.
//!
//! Cache population and cache reuse converge here so sparse eligibility and
//! dense fallback remain one policy per chunking regime.

use crate::metric::histogram::parzen::ParzenJointHistogram;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_image::tensor::{Backend, Tensor};
use ritk_interpolation::LinearInterpolator;

#[cfg(feature = "direct-parzen")]
use super::get_masked_cached_sparse_w_fixed;
use super::get_masked_cached_w_fixed_t;

impl<B: Backend> ParzenJointHistogram<B> {
    pub(super) fn compute_masked_from_cache(
        &self,
        cache_key: u64,
        n: usize,
        w_fixed_transposed: &Tensor<f32, B>,
        moving_values: &Tensor<f32, B>,
        oob_mask: Option<&Tensor<f32, B>>,
    ) -> Tensor<f32, B> {
        debug_assert!(self
            .masked_cache
            .with_ref(|cache| { get_masked_cached_w_fixed_t(cache, cache_key, n).is_some() }));

        #[cfg(feature = "direct-parzen")]
        {
            let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
            let sparse = self.masked_cache.with_mut(|cache| {
                get_masked_cached_sparse_w_fixed(cache, cache_key, n, self.num_bins, sigma_sq_fix)
            });
            if let Some(sparse) = sparse {
                return self.compute_joint_histogram_from_cache_sparse_dispatch(
                    &sparse,
                    moving_values,
                    oob_mask,
                );
            }
        }

        self.compute_joint_histogram_from_cache_dispatch(
            w_fixed_transposed,
            moving_values,
            oob_mask,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the dispatch boundary preserves the typed image, transform, and interpolation inputs"
    )]
    pub(super) fn compute_masked_chunked_from_cache<const D: usize>(
        &self,
        cache_key: u64,
        n: usize,
        w_fixed_transposed: &Tensor<f32, B>,
        fixed: &Image<f32, B, D>,
        fixed_world_points: &Tensor<f32, B>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
    ) -> Tensor<f32, B> {
        debug_assert!(self
            .masked_cache
            .with_ref(|cache| { get_masked_cached_w_fixed_t(cache, cache_key, n).is_some() }));

        #[cfg(feature = "direct-parzen")]
        {
            let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
            let sparse = self.masked_cache.with_mut(|cache| {
                get_masked_cached_sparse_w_fixed(cache, cache_key, n, self.num_bins, sigma_sq_fix)
            });
            if let Some(sparse) = sparse {
                return self.compute_masked_chunked_from_sparse_cache(
                    &sparse,
                    fixed,
                    fixed_world_points,
                    moving,
                    transform,
                    interpolator,
                );
            }
        }

        self.compute_masked_chunked_from_dense_cache(
            w_fixed_transposed,
            fixed,
            fixed_world_points,
            moving,
            transform,
            interpolator,
        )
    }
}

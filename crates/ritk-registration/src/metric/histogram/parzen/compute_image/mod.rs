//! Image-level joint histogram computation with spatial transform and caching.
//!
//! Extracted from `compute.rs` to keep the 500-line structural limit.
//! Cache/normalization helpers live in [`super::image_cache_helpers`].
//! Chunked iteration strategy lives in [`chunked`].

pub(super) mod chunked;

/// Whether stochastic subsampling is active for a histogram computation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SamplingMode {
    /// A random subset of voxels is used (stochastic sampling).
    Sampled,
    /// All voxels are used (dense evaluation).
    Dense }

use super::super::cache::WFixedCache;
use super::image_cache_helpers::{
    cache_matches_image, extract_cached_points, get_cached_w_fixed_t };
#[cfg(feature = "direct-parzen")]
use super::image_cache_helpers::{get_cached_sparse_w_fixed, normalize_fixed_values};
use super::ParzenJointHistogram;
use crate::metric::sampling::{resolve_n_points, SamplingConfig};
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_image::{generate_grid, generate_random_points};
use ritk_interpolation::{Interpolator, LinearInterpolator};

// `make_cache` was moved to `super::super::cache` in Sprint 354 (DRY-354-03).
// Both cfg-gated overloads (`#[cfg(feature = "direct-parzen")]` and
// `#[cfg(not(feature = "direct-parzen"))]`) are consolidated there.
// Call sites below use `super::super::cache::make_cache`.

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute joint histogram from images with transform and sampling.
    /// Handles chunking of the spatial domain to respect memory and dispatch limits.
    pub fn compute_image_joint_histogram<const D: usize>(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        sampling: SamplingConfig,
    ) -> Tensor<f32, B> {
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();

        // 1. Determine n and points strategy
        let total_voxels = fixed_shape.iter().product::<usize>();
        let (fixed_indices, n, use_sampling, cached_points) = if sampling.is_active() {
            let num_samples = resolve_n_points(&sampling, total_voxels);
            let indices = generate_random_points(fixed_shape, num_samples, &device);
            (Some(indices), num_samples, SamplingMode::Sampled, None)
        } else {
            let cached_points = self.cache.with_ref(|cache| {
                cache
                    .as_ref()
                    .filter(|c| cache_matches_image(c, fixed))
                    .map(|c| c.points.clone())
            });
            if let Some(pts) = cached_points {
                (None, total_voxels, SamplingMode::Dense, Some(pts))
            } else {
                let indices = generate_grid(fixed_shape, &device);
                (Some(indices), total_voxels, SamplingMode::Dense, None)
            }
        };

        if n <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            // â”€â”€ Non-chunked path â”€â”€
            let cached_w_fixed_t = (use_sampling == SamplingMode::Dense)
                .then(|| {
                    self.cache
                        .with_ref(|cache| get_cached_w_fixed_t(cache, fixed))
                })
                .flatten();

            #[cfg(feature = "direct-parzen")]
            let cached_sparse = (use_sampling == SamplingMode::Dense)
                .then(|| {
                    let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                    self.cache.with_mut(|cache| {
                        get_cached_sparse_w_fixed(cache, fixed, self.num_bins, sigma_sq_fix)
                    })
                })
                .flatten();
            let fixed_points = if let Some(pts) = cached_points {
                pts
            } else {
                fixed.index_to_world_tensor(
                    fixed_indices
                        .as_ref()
                        .expect("fixed_indices must be Some when no cached points exist")
                        .clone(),
                )
            };

            let (moving_values, oob_mask): (Tensor<f32, B>, Option<Tensor<f32, B>>) = {
                let moving_points = transform.transform_points(fixed_points);
                let moving_indices = moving.world_to_index_tensor(moving_points);
                let oob = if D == 3 {
                    Some(super::compute_oob_mask(
                        &moving_indices,
                        moving.shape().as_ref(),
                    ))
                } else {
                    None
                };
                let values = interpolator.interpolate(moving.data(), moving_indices);
                (values, oob)
            };

            // â”€â”€ Sparse cache path (CMA-ES, direct-parzen) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Prefer the sparse dispatch when available: it eliminates the
            // 0..num_bins inner scan and the `if w_f > 0.0` branch (~3Ã— faster
            // than the dense cache path on CPU). Only safe for derivative-free
            // backends (CMA-ES uses B::InnerBackend).
            #[cfg(feature = "direct-parzen")]
            if let Some(sparse) = cached_sparse {
                if !moving_values.is_require_grad() {
                    return self.compute_joint_histogram_from_cache_sparse_dispatch(
                        &sparse,
                        &moving_values,
                        oob_mask.as_ref(),
                    );
                }
            }

            if let Some(w_fixed_t) = cached_w_fixed_t {
                self.compute_joint_histogram_from_cache_dispatch(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            } else {
                let fixed_values = if use_sampling == SamplingMode::Sampled {
                    interpolator.interpolate(
                        fixed.data(),
                        fixed_indices
                            .clone()
                            .expect("fixed_indices must be Some in sampling mode"),
                    )
                } else {
                    fixed.data().clone().reshape([n])
                };

                let w_fixed_t = self.compute_w_fixed_transposed(&fixed_values, n);

                if use_sampling == SamplingMode::Dense {
                    #[cfg(feature = "direct-parzen")]
                    let fixed_norm = Some(normalize_fixed_values::<B>(
                        &fixed_values,
                        self.min_intensity,
                        self.max_intensity,
                        self.num_bins,
                    ));
                    #[cfg(not(feature = "direct-parzen"))]
                    let fixed_norm: Option<()> = None;

                    self.cache.with_mut(|cache| {
                        *cache = Some(super::super::cache::make_cache(
                            fixed.index_to_world_tensor(
                                fixed_indices
                                    .as_ref()
                                    .expect("fixed_indices must be Some when storing cache")
                                    .clone(),
                            ),
                            w_fixed_t.clone(),
                            fixed,
                            fixed_norm,
                        ));
                    });
                }

                self.compute_joint_histogram_from_cache_dispatch(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            }
        } else {
            // â”€â”€ Chunked path â”€â”€
            self.compute_image_joint_histogram_chunked(
                fixed,
                moving,
                transform,
                interpolator,
                n,
                use_sampling,
                fixed_indices,
                cached_points,
            )
        }
    }

    /// Compute joint histogram from images with transform, reusing a
    /// caller-supplied `W_fixed^T [num_bins, N]` matrix (350-P1-03).
    ///
    /// This is the public cache-hit fast path: callers that have precomputed
    /// the fixed-image Parzen weight matrix once (e.g. via the first iteration
    /// of a registration level, then `extract_w_fixed_t_cache`) can pass the
    /// matrix directly to avoid the O(N Ã— num_bins) Parzen weight
    /// recomputation on every iteration.
    ///
    /// # Arguments
    /// * `fixed` â€” Fixed reference image (used for spatial transform, OOB mask, and cache key).
    /// * `moving` â€” Moving image.
    /// * `transform` â€” Current candidate transform.
    /// * `interpolator` â€” Interpolator for sampling `moving`.
    /// * `w_fixed_transposed` â€” Precomputed Parzen weight matrix `[num_bins, N]`.
    ///   Treated as a constant â€” the autodiff path goes through the moving
    ///   image's interpolation only.
    /// * `n` â€” Number of points `N` in `w_fixed_transposed`. Must match the
    ///   second dimension of `w_fixed_transposed`.
    ///
    /// # Returns
    /// Joint histogram `[num_bins, num_bins]` on autodiff graph.
    ///
    /// # Performance
    /// For a 256Â³ volume with Mattes MI (50 bins), the full-grid
    /// `W_fixed^T` matrix is `[50, 16M]` = ~3.2 GB. Recomputing it every
    /// iteration costs ~400 ms on a 16-core CPU; reusing it across iterations
    /// of one level saves ~99 % of that on iteration 2+. See
    /// `docs/audit_optimization_sprint_350.md` Â§2.3 for the breakdown.
    pub fn compute_image_joint_histogram_with_w_fixed<const D: usize>(
        &self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        w_fixed_transposed: &Tensor<f32, B>,
        n: usize,
    ) -> Tensor<f32, B> {
        let device = fixed.data().device();

        // Reuse cached fixed_points if available, else build them on demand.
        // We do NOT populate the internal HistogramCache here â€” the caller
        // owns the W_fixed^T.
        let fixed_points = if let Some(pts) = extract_cached_points(fixed, &self.cache) {
            pts
        } else {
            let indices = generate_grid(fixed.shape(), &device);
            fixed.index_to_world_tensor(indices)
        };

        if n <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            // â”€â”€ Non-chunked path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            let (moving_values, oob_mask): (Tensor<f32, B>, Option<Tensor<f32, B>>) = {
                let moving_points = transform.transform_points(fixed_points);
                let moving_indices = moving.world_to_index_tensor(moving_points);
                let oob = if D == 3 {
                    Some(super::compute_oob_mask(
                        &moving_indices,
                        moving.shape().as_ref(),
                    ))
                } else {
                    None
                };
                let values = interpolator.interpolate(moving.data(), moving_indices);
                (values, oob)
            };

            // Skip the cache-miss path: use the caller's W_fixed^T directly.
            self.compute_joint_histogram_from_cache_dispatch(
                w_fixed_transposed,
                &moving_values,
                oob_mask.as_ref(),
            )
        } else {
            // â”€â”€ Chunked path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self.compute_image_joint_histogram_with_w_fixed_chunked(
                fixed,
                moving,
                transform,
                interpolator,
                w_fixed_transposed,
                &fixed_points,
                n,
            )
        }
    }

    /// Extract a `WFixedCache` entry from the internal `HistogramCache` for
    /// use by external callers (e.g. `MutualInformation`'s per-instance cache,
    /// 350-P1-03). Returns `None` if the internal cache is empty, the cached
    /// entry's spatial metadata does not match `fixed`, or the stored
    /// W_fixed^T tensor is `None`.
    ///
    /// This is the public entry point to the per-call W_fixed^T reuse pattern:
    /// `MutualInformation::forward` calls this once per registration level to
    /// populate its per-instance cache after the first `compute_image_joint_histogram`
    /// call, then `MutualInformation::forward_with_cache` reuses the cache
    /// across subsequent iterations.
    pub(crate) fn extract_w_fixed_t_cache<const D: usize>(
        &self,
        fixed: &Image<f32, B, D>,
        n: usize,
    ) -> Option<WFixedCache<B>> {
        self.cache.with_ref(|cache_opt| {
            let cache = cache_opt.as_ref()?;
            if !cache_matches_image(cache, fixed) {
                return None;
            }
            let w_fixed_t = cache.w_fixed_transposed.clone()?;
            Some(WFixedCache::from_image(fixed, n, w_fixed_t))
        })
    }
}

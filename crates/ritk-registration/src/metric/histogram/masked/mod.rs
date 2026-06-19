use super::cache::MaskedHistogramCache;
use super::parzen::ParzenJointHistogram;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_interpolation::{Interpolator, LinearInterpolator};

mod masked_chunked;

#[cfg(feature = "direct-parzen")]
use super::cache::SparseWFixedCache;
#[cfg(feature = "direct-parzen")]
use super::parzen::direct::SparseWFixedT;

/// Helper: read the dense W_fixed^T from the masked cache if the key matches.
///
/// Returns `Some(cached_tensor)` when the cache entry exists, its `cache_key`
/// matches the supplied key, and the stored point count `n` matches the
/// current number of foreground points. Otherwise returns `None`.
fn get_masked_cached_w_fixed_t<B: Backend>(
    cache_guard: &Option<MaskedHistogramCache<B>>,
    cache_key: u64,
    n: usize,
) -> Option<Tensor<B, 2>> {
    cache_guard.as_ref().and_then(|c| {
        (c.cache_key == cache_key && c.n == n)
            .then(|| c.w_fixed_transposed.clone())
            .flatten()
    })
}

/// Helper: read or lazily build the sparse W_fixed^T from the masked cache.
///
/// If the sparse cache already exists, returns a clone. Otherwise, if the
/// cache contains `fixed_norm` (the normalized fixed-image values), builds
/// the sparse cache from it, stores it in the cache for future use, and
/// returns it. This lazy construction reduces peak memory.
#[cfg(feature = "direct-parzen")]
fn get_masked_cached_sparse_w_fixed<B: Backend>(
    cache_guard: &mut Option<MaskedHistogramCache<B>>,
    cache_key: u64,
    n: usize,
    num_bins: usize,
    sigma_sq_fix: f32,
) -> Option<SparseWFixedT> {
    if B::ad_enabled() {
        return None;
    }
    let cache = cache_guard.as_mut()?;
    if cache.cache_key != cache_key || cache.n != n {
        return None;
    }
    cache.get_or_build_sparse_w_fixed(num_bins, sigma_sq_fix)
}

// `compute_fingerprint` and `make_masked_cache` live in `super::cache`
// (histogram/cache.rs). Call sites below use `super::cache::make_masked_cache`.

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
    /// * `cache_key` — Optional caller-supplied key identifying the mask/point-set.
    ///   When `Some(key)`, the fixed-image Parzen weights (`w_fixed_transposed`) are
    ///   cached and reused across calls with the same key and point count, eliminating
    ///   the O(N × num_bins) fixed-weight computation on every iteration after the first.
    ///   The caller should provide a generation counter or hash that changes only when
    ///   the mask changes (e.g., CMA-ES optimizer generation counter). Pass `None` to
    ///   disable caching (weights are recomputed on every call).
    pub fn compute_masked_joint_histogram<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        fixed_world_points: &Tensor<B, 2>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        cache_key: Option<u64>,
    ) -> Tensor<B, 2> {
        let n = fixed_world_points.dims()[0];
        let device = fixed_world_points.device();

        if n == 0 {
            // Degenerate: empty mask — return zero histogram.
            return Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);
        }

        if n <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            // ── Non-chunked path ──────────────────────────────────────────────

            // Convert fixed world coords → fixed voxel indices, then sample.
            let fixed_voxel_indices = fixed.world_to_index_tensor(fixed_world_points.clone());
            let fixed_values = interpolator.interpolate(fixed.data(), fixed_voxel_indices);

            // Apply transform to get moving world coords, then sample moving image.
            let (moving_values, oob_mask): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                let moving_world_points = transform.transform_points(fixed_world_points.clone());
                let moving_voxel_indices = moving.world_to_index_tensor(moving_world_points);
                let oob = if D == 3 {
                    Some(super::parzen::compute_oob_mask(
                        &moving_voxel_indices,
                        moving.shape().as_ref(),
                    ))
                } else {
                    None
                };
                let values = interpolator.interpolate(moving.data(), moving_voxel_indices);
                (values, oob)
            };

            // Attempt to use cached W_fixed^T for the fixed-image Parzen weights.
            if let Some(key) = cache_key {
                let cached_w_fixed_t = self
                    .masked_cache
                    .with_ref(|cache| get_masked_cached_w_fixed_t(cache, key, n));

                if let Some(w_fixed_t) = cached_w_fixed_t {
                    // Cache hit — use the cached dense W_fixed^T.
                    #[cfg(feature = "direct-parzen")]
                    {
                        // Also try sparse cache for derivative-free backends.
                        let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                        let cached_sparse = self.masked_cache.with_mut(|cache| {
                            get_masked_cached_sparse_w_fixed(
                                cache,
                                key,
                                n,
                                self.num_bins,
                                sigma_sq_fix,
                            )
                        });
                        if let Some(sparse) = cached_sparse {
                            return self.compute_joint_histogram_from_cache_sparse_dispatch(
                                &sparse,
                                &moving_values,
                                oob_mask.as_ref(),
                            );
                        }
                    }
                    return self.compute_joint_histogram_from_cache_dispatch(
                        &w_fixed_t,
                        &moving_values,
                        oob_mask.as_ref(),
                    );
                }

                // Cache miss — compute W_fixed^T and store in cache.
                let w_fixed_transposed = self.compute_w_fixed_transposed(&fixed_values, n);

                #[cfg(feature = "direct-parzen")]
                let fixed_norm = Some(
                    super::parzen::dispatch::normalize_and_extract(
                        &fixed_values,
                        self.min_intensity,
                        self.max_intensity,
                        self.num_bins,
                    )
                    .into_owned(),
                );
                #[cfg(not(feature = "direct-parzen"))]
                let fixed_norm: Option<()> = None;

                let new_cache =
                    super::cache::make_masked_cache(key, w_fixed_transposed.clone(), n, fixed_norm);
                self.masked_cache.with_mut(|cache| *cache = Some(new_cache));
                #[cfg(feature = "direct-parzen")]
                {
                    let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                    let cached_sparse = self.masked_cache.with_mut(|cache| {
                        get_masked_cached_sparse_w_fixed(cache, key, n, self.num_bins, sigma_sq_fix)
                    });
                    if let Some(sparse) = cached_sparse {
                        return self.compute_joint_histogram_from_cache_sparse_dispatch(
                            &sparse,
                            &moving_values,
                            oob_mask.as_ref(),
                        );
                    }
                }
                return self.compute_joint_histogram_from_cache_dispatch(
                    &w_fixed_transposed,
                    &moving_values,
                    oob_mask.as_ref(),
                );
            }

            // No cache key provided — fall through to uncached dispatch.
            self.compute_joint_histogram_dispatch(&fixed_values, &moving_values, oob_mask.as_ref())
        } else {
            // ── Chunked path ──────────────────────────────────────────────────

            // If a cache key is provided, try to get the full cached W_fixed^T
            // and slice it per chunk instead of recomputing.
            if let Some(key) = cache_key {
                let cached_w_fixed_t = self
                    .masked_cache
                    .with_ref(|cache| get_masked_cached_w_fixed_t(cache, key, n));

                if let Some(full_w_fixed_t) = cached_w_fixed_t {
                    // Cache hit — slice the cached W_fixed^T per chunk.
                    #[cfg(feature = "direct-parzen")]
                    {
                        let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                        let cached_sparse = self.masked_cache.with_mut(|cache| {
                            get_masked_cached_sparse_w_fixed(
                                cache,
                                key,
                                n,
                                self.num_bins,
                                sigma_sq_fix,
                            )
                        });
                        if let Some(sparse) = cached_sparse {
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
                    return self.compute_masked_chunked_from_dense_cache(
                        &full_w_fixed_t,
                        fixed,
                        fixed_world_points,
                        moving,
                        transform,
                        interpolator,
                    );
                }

                // Cache miss — compute full W_fixed^T, cache it, then use it.
                // We need fixed_values for the full point set to compute W_fixed^T.
                let fixed_voxel_indices = fixed.world_to_index_tensor(fixed_world_points.clone());
                let fixed_values = interpolator.interpolate(fixed.data(), fixed_voxel_indices);
                let w_fixed_transposed = self.compute_w_fixed_transposed(&fixed_values, n);

                #[cfg(feature = "direct-parzen")]
                let fixed_norm = Some(
                    super::parzen::dispatch::normalize_and_extract(
                        &fixed_values,
                        self.min_intensity,
                        self.max_intensity,
                        self.num_bins,
                    )
                    .into_owned(),
                );
                #[cfg(not(feature = "direct-parzen"))]
                let fixed_norm: Option<()> = None;
                let new_cache =
                    super::cache::make_masked_cache(key, w_fixed_transposed.clone(), n, fixed_norm);
                self.masked_cache.with_mut(|cache| *cache = Some(new_cache));
                #[cfg(feature = "direct-parzen")]
                {
                    let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                    let cached_sparse = self.masked_cache.with_mut(|cache| {
                        get_masked_cached_sparse_w_fixed(cache, key, n, self.num_bins, sigma_sq_fix)
                    });
                    if let Some(sparse) = cached_sparse {
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
                return self.compute_masked_chunked_from_dense_cache(
                    &w_fixed_transposed,
                    fixed,
                    fixed_world_points,
                    moving,
                    transform,
                    interpolator,
                );
            }

            // No cache key — fall through to the original uncached chunked path.
            let num_chunks = n.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n);
                #[allow(clippy::single_range_in_vec_init)]
                let chunk_fixed_world = fixed_world_points.clone().slice([start..end]);
                let chunk_fixed_idx = fixed.world_to_index_tensor(chunk_fixed_world.clone());
                let chunk_fixed_vals = interpolator.interpolate(fixed.data(), chunk_fixed_idx);

                let (chunk_moving_vals, chunk_oob): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                    let chunk_moving_world = transform.transform_points(chunk_fixed_world);
                    let chunk_moving_idx = moving.world_to_index_tensor(chunk_moving_world);
                    let oob = if D == 3 {
                        Some(super::parzen::compute_oob_mask(
                            &chunk_moving_idx,
                            moving.shape().as_ref(),
                        ))
                    } else {
                        None
                    };
                    let vals = interpolator.interpolate(moving.data(), chunk_moving_idx);
                    (vals, oob)
                };

                joint_hist_acc = joint_hist_acc
                    + self.compute_joint_histogram_dispatch(
                        &chunk_fixed_vals,
                        &chunk_moving_vals,
                        chunk_oob.as_ref(),
                    );
            }

            joint_hist_acc
        }
    }
}

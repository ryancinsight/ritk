//! Image-level joint histogram computation with spatial transform and caching.
//!
//! Extracted from `compute.rs` to keep the 500-line structural limit.
//! Cache/normalization helpers live in [`super::image_cache_helpers`].

use super::super::cache::WFixedCache;
use super::image_cache_helpers::{
    cache_matches_image, extract_cached_points, get_cached_w_fixed_t,
};
#[cfg(feature = "direct-parzen")]
use super::image_cache_helpers::{get_cached_sparse_w_fixed, normalize_fixed_values};
use super::ParzenJointHistogram;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

// `make_cache` was moved to `super::super::cache` in Sprint 354 (DRY-354-03).
// Both cfg-gated overloads (`#[cfg(feature = "direct-parzen")]` and
// `#[cfg(not(feature = "direct-parzen"))]`) are consolidated there.
// Call sites below use `super::super::cache::make_cache`.

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute joint histogram from images with transform and sampling.
    /// Handles chunking of the spatial domain to respect memory and dispatch limits.
    pub fn compute_image_joint_histogram<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        sampling_percentage: Option<f32>,
    ) -> Tensor<B, 2> {
        use ritk_core::image::grid;

        let fixed_shape = fixed.shape();
        let device = fixed.data().device();

        // 1. Determine n and points strategy
        let (fixed_indices, n, use_sampling, cached_points) = if let Some(p) = sampling_percentage {
            let total_voxels = fixed_shape.iter().product::<usize>();
            let num_samples = (total_voxels as f32 * p) as usize;
            let indices = grid::generate_random_points(fixed_shape, num_samples, &device);
            (Some(indices), num_samples, true, None)
        } else {
            let total_voxels = fixed_shape.iter().product::<usize>();
            let cached_points = {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                cache
                    .as_ref()
                    .filter(|c| cache_matches_image(c, fixed))
                    .map(|c| c.points.clone())
            };
            if let Some(pts) = cached_points {
                (None, total_voxels, false, Some(pts))
            } else {
                let indices = grid::generate_grid(fixed_shape, &device);
                (Some(indices), total_voxels, false, None)
            }
        };

        if n <= crate::wgpu_compat::WGPU_CHUNK_SIZE {
            // ── Non-chunked path ──
            let cached_w_fixed_t = (!use_sampling)
                .then(|| {
                    let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    get_cached_w_fixed_t(&cache, fixed)
                })
                .flatten();

            #[cfg(feature = "direct-parzen")]
            let cached_sparse = (!use_sampling)
                .then(|| {
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                    get_cached_sparse_w_fixed(&mut cache, fixed, self.num_bins, sigma_sq_fix)
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

            let (moving_values, oob_mask): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                let moving_points = transform.transform_points(fixed_points);
                let moving_indices = moving.world_to_index_tensor(moving_points);
                let oob = if D == 3 {
                    Some(super::compute_oob_mask_3d(
                        &moving_indices,
                        moving.shape().as_ref(),
                    ))
                } else {
                    None
                };
                let values = interpolator.interpolate(moving.data(), moving_indices);
                (values, oob)
            };

            // ── Sparse cache path (CMA-ES, direct-parzen) ──────────────────────
            // Prefer the sparse dispatch when available: it eliminates the
            // 0..num_bins inner scan and the `if w_f > 0.0` branch (~3× faster
            // than the dense cache path on CPU). Only safe for derivative-free
            // backends (CMA-ES uses B::InnerBackend).
            #[cfg(feature = "direct-parzen")]
            if let Some(sparse) = cached_sparse {
                return self.compute_joint_histogram_from_cache_sparse_dispatch(
                    &sparse,
                    &moving_values,
                    oob_mask.as_ref(),
                );
            }

            if let Some(w_fixed_t) = cached_w_fixed_t {
                self.compute_joint_histogram_from_cache_dispatch(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            } else {
                let fixed_values = if use_sampling {
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

                if !use_sampling {
                    #[cfg(feature = "direct-parzen")]
                    let fixed_norm = Some(normalize_fixed_values::<B>(
                        &fixed_values,
                        self.min_intensity,
                        self.max_intensity,
                        self.num_bins,
                    ));
                    #[cfg(not(feature = "direct-parzen"))]
                    let fixed_norm: Option<()> = None;

                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
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
                }

                self.compute_joint_histogram_from_cache_dispatch(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            }
        } else {
            // ── Chunked path ──
            let num_chunks = n.div_ceil(crate::wgpu_compat::WGPU_CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            let cached_w_fixed_t = (!use_sampling)
                .then(|| {
                    let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    get_cached_w_fixed_t(&cache, fixed)
                })
                .flatten();

            #[cfg(feature = "direct-parzen")]
            let cached_sparse = (!use_sampling)
                .then(|| {
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                    get_cached_sparse_w_fixed(&mut cache, fixed, self.num_bins, sigma_sq_fix)
                })
                .flatten();
            let all_fixed_points = if let Some(pts) = cached_points {
                pts
            } else if !use_sampling {
                let pts = fixed.index_to_world_tensor(
                    fixed_indices
                        .as_ref()
                        .expect("fixed_indices must be Some when no cached points exist")
                        .clone(),
                );
                if let Some(w_fixed_t) = &cached_w_fixed_t {
                    // Cache hit for W_fixed^T — only update cache if points are missing or
                    // the cached tensor dimensions are stale.  The sparse cache is not
                    // passed here; it is built lazily from `fixed_norm` on first access.
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    if cache.as_ref().is_none_or(|c| c.points.dims() != [n, D]) {
                        #[cfg(feature = "direct-parzen")]
                        let fixed_norm_for_cache: Option<Vec<f32>> = None;
                        #[cfg(not(feature = "direct-parzen"))]
                        let fixed_norm_for_cache: Option<()> = None;
                        *cache = Some(super::super::cache::make_cache(
                            pts.clone(),
                            w_fixed_t.clone(),
                            fixed,
                            fixed_norm_for_cache,
                        ));
                    }
                } else {
                    let fixed_data_flat = fixed.data().clone().reshape([n]);
                    let w_fixed_t = self.compute_w_fixed_transposed(&fixed_data_flat, n);

                    #[cfg(feature = "direct-parzen")]
                    let fixed_norm = Some(normalize_fixed_values::<B>(
                        &fixed_data_flat,
                        self.min_intensity,
                        self.max_intensity,
                        self.num_bins,
                    ));
                    #[cfg(not(feature = "direct-parzen"))]
                    let fixed_norm: Option<()> = None;

                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    *cache = Some(super::super::cache::make_cache(
                        pts.clone(),
                        w_fixed_t,
                        fixed,
                        fixed_norm,
                    ));
                }
                pts
            } else {
                Tensor::zeros([1, 1], &device)
            };

            let have_all_points = !use_sampling;

            for i in 0..num_chunks {
                let start = i * crate::wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + crate::wgpu_compat::WGPU_CHUNK_SIZE, n);
                let chunk_range = start..end;

                let chunk_fixed_points = if have_all_points {
                    all_fixed_points.clone().slice([chunk_range.clone()])
                } else {
                    let chunk_indices = fixed_indices
                        .as_ref()
                        .expect("fixed_indices must be Some in sampling chunk path")
                        .clone()
                        .slice([chunk_range.clone()]);
                    fixed.index_to_world_tensor(chunk_indices)
                };

                let (chunk_moving_values, chunk_oob): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                    let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                    let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                    let oob = if D == 3 {
                        Some(super::compute_oob_mask_3d(
                            &chunk_moving_indices,
                            moving.shape().as_ref(),
                        ))
                    } else {
                        None
                    };
                    let values = interpolator.interpolate(moving.data(), chunk_moving_indices);
                    (values, oob)
                };

                // ── Chunked histogram computation ──────────────────────────────────
                //
                // Three dispatch paths, in order of preference:
                //   1. Sparse cache hit — iterate only ~7 non-zero fixed bins per sample.
                //      Slicing the sparse cache is trivial (just `sparse[start..end].to_vec()`)
                //      and avoids the old `w_fixed_t.clone().slice()` pattern that cloned
                //      the entire [num_bins × N] dense tensor (~4 MB for N=32K) just to
                //      extract one chunk. The sparse slice copies only ~56 bytes per sample.
                //   2. Dense cache hit — slice [num_bins, start..end] and use the tensor
                //      matmul path (autodiff-safe, needed for RSGD).
                //   3. Cache miss (sampling) — compute both W_fixed and W_moving from scratch.

                #[cfg(feature = "direct-parzen")]
                let chunk_hist = if let Some(ref sparse) = cached_sparse {
                    let chunk_sparse: super::direct::SparseWFixedT = sparse[start..end].to_vec();
                    self.compute_joint_histogram_from_cache_sparse_dispatch(
                        &chunk_sparse,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                } else if let Some(ref w_fixed_t) = cached_w_fixed_t {
                    let chunk_w_fixed_t = w_fixed_t.clone().slice([0..self.num_bins, chunk_range]);
                    self.compute_joint_histogram_from_cache_dispatch(
                        &chunk_w_fixed_t,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                } else {
                    let chunk_fixed_values = if use_sampling {
                        let chunk_indices = fixed_indices
                            .as_ref()
                            .expect(
                                "fixed_indices must be Some in sampling chunk path (direct-parzen)",
                            )
                            .clone()
                            .slice([chunk_range.clone()]);
                        interpolator.interpolate(fixed.data(), chunk_indices)
                    } else {
                        fixed.data().clone().reshape([n]).slice([chunk_range])
                    };
                    self.compute_joint_histogram_dispatch(
                        &chunk_fixed_values,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                };

                #[cfg(not(feature = "direct-parzen"))]
                let chunk_hist = if let Some(ref w_fixed_t) = cached_w_fixed_t {
                    let chunk_w_fixed_t = w_fixed_t.clone().slice([0..self.num_bins, chunk_range]);
                    self.compute_joint_histogram_from_cache_dispatch(
                        &chunk_w_fixed_t,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                } else {
                    let chunk_fixed_values = if use_sampling {
                        let chunk_indices = fixed_indices
                            .as_ref()
                            .expect("fixed_indices must be Some in sampling chunk path (no direct-parzen)")
                            .clone()
                            .slice([chunk_range.clone()]);
                        interpolator.interpolate(fixed.data(), chunk_indices)
                    } else {
                        fixed.data().clone().reshape([n]).slice([chunk_range])
                    };
                    self.compute_joint_histogram_dispatch(
                        &chunk_fixed_values,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                };

                joint_hist_acc = joint_hist_acc + chunk_hist;
            }

            joint_hist_acc
        }
    }

    /// Compute joint histogram from images with transform, reusing a
    /// caller-supplied `W_fixed^T [num_bins, N]` matrix (350-P1-03).
    ///
    /// This is the public cache-hit fast path: callers that have precomputed
    /// the fixed-image Parzen weight matrix once (e.g. via the first iteration
    /// of a registration level, then `extract_w_fixed_t_cache`) can pass the
    /// matrix directly to avoid the O(N × num_bins) Parzen weight
    /// recomputation on every iteration.
    ///
    /// # Arguments
    /// * `fixed` — Fixed reference image (used for spatial transform, OOB mask, and cache key).
    /// * `moving` — Moving image.
    /// * `transform` — Current candidate transform.
    /// * `interpolator` — Interpolator for sampling `moving`.
    /// * `w_fixed_transposed` — Precomputed Parzen weight matrix `[num_bins, N]`.
    ///   Treated as a constant — the autodiff path goes through the moving
    ///   image's interpolation only.
    /// * `n` — Number of points `N` in `w_fixed_transposed`. Must match the
    ///   second dimension of `w_fixed_transposed`.
    ///
    /// # Returns
    /// Joint histogram `[num_bins, num_bins]` on autodiff graph.
    ///
    /// # Performance
    /// For a 256³ volume with Mattes MI (50 bins), the full-grid
    /// `W_fixed^T` matrix is `[50, 16M]` = ~3.2 GB. Recomputing it every
    /// iteration costs ~400 ms on a 16-core CPU; reusing it across iterations
    /// of one level saves ~99 % of that on iteration 2+. See
    /// `docs/audit_optimization_sprint_350.md` §2.3 for the breakdown.
    pub fn compute_image_joint_histogram_with_w_fixed<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        w_fixed_transposed: &Tensor<B, 2>,
        n: usize,
    ) -> Tensor<B, 2> {
        let device = fixed.data().device();

        // Reuse cached fixed_points if available, else build them on demand.
        // We do NOT populate the internal HistogramCache here — the caller
        // owns the W_fixed^T.
        let fixed_points = if let Some(pts) = extract_cached_points(fixed, &self.cache) {
            pts
        } else {
            let indices = ritk_core::image::grid::generate_grid(fixed.shape(), &device);
            fixed.index_to_world_tensor(indices)
        };

        if n <= crate::wgpu_compat::WGPU_CHUNK_SIZE {
            // ── Non-chunked path ──────────────────────────────────────────────
            let (moving_values, oob_mask): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                let moving_points = transform.transform_points(fixed_points);
                let moving_indices = moving.world_to_index_tensor(moving_points);
                let oob = if D == 3 {
                    Some(super::compute_oob_mask_3d(
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
            // ── Chunked path ──────────────────────────────────────────────────
            let num_chunks = n.div_ceil(crate::wgpu_compat::WGPU_CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            for i in 0..num_chunks {
                let start = i * crate::wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + crate::wgpu_compat::WGPU_CHUNK_SIZE, n);
                let chunk_range = start..end;
                #[allow(clippy::single_range_in_vec_init)]
                let chunk_fixed_points = fixed_points.clone().slice([chunk_range.clone()]);

                let (chunk_moving_values, chunk_oob): (Tensor<B, 1>, Option<Tensor<B, 1>>) = {
                    let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                    let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                    let oob = if D == 3 {
                        Some(super::compute_oob_mask_3d(
                            &chunk_moving_indices,
                            moving.shape().as_ref(),
                        ))
                    } else {
                        None
                    };
                    let values = interpolator.interpolate(moving.data(), chunk_moving_indices);
                    (values, oob)
                };

                // Slice the caller's W_fixed^T per chunk. The dense tensor matmul
                // path is autodiff-safe, so it works for both RSGD and CMA-ES.
                let chunk_w_fixed_t = w_fixed_transposed
                    .clone()
                    .slice([0..self.num_bins, chunk_range]);
                let chunk_hist = self.compute_joint_histogram_from_cache_dispatch(
                    &chunk_w_fixed_t,
                    &chunk_moving_values,
                    chunk_oob.as_ref(),
                );

                joint_hist_acc = joint_hist_acc + chunk_hist;
            }

            joint_hist_acc
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
        fixed: &Image<B, D>,
        n: usize,
    ) -> Option<WFixedCache<B>> {
        let cache_guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let cache = cache_guard.as_ref()?;
        if !cache_matches_image(cache, fixed) {
            return None;
        }
        let w_fixed_t = cache.w_fixed_transposed.clone()?;
        Some(WFixedCache::from_image(fixed, n, w_fixed_t))
    }
}

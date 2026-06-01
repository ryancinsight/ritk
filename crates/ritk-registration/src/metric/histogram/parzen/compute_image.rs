//! Image-level joint histogram computation with spatial transform and caching.
//!
//! Extracted from `compute.rs` to keep the 500-line structural limit.

use super::super::cache::HistogramCache;
#[cfg(feature = "direct-parzen")]
use super::super::cache::SparseWFixedCache;
use super::ParzenJointHistogram;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

/// Check whether a cached histogram entry matches the given image's spatial metadata.
fn cache_matches_image<B: Backend, const D: usize>(
    cache: &HistogramCache<B>,
    fixed: &Image<B, D>,
) -> bool {
    let fs = fixed.shape();
    cache.shape.as_slice() == fs
        && cache.origin.iter().eq(fixed.origin().0.iter())
        && cache.spacing.iter().eq(fixed.spacing().0.iter())
        && cache.direction.iter().eq(fixed.direction().0.iter())
}

/// Helper: read the dense W_fixed^T from the cache if it matches the fixed image.
fn get_cached_w_fixed_t<B: Backend, const D: usize>(
    cache_guard: &Option<HistogramCache<B>>,
    fixed: &Image<B, D>,
) -> Option<Tensor<B, 2>> {
    cache_guard.as_ref().and_then(|c| {
        cache_matches_image(c, fixed)
            .then(|| c.w_fixed_transposed.clone())
            .flatten()
    })
}

/// Helper: read or lazily build the sparse W_fixed^T from the cache.
///
/// If the sparse cache already exists, returns a clone. Otherwise, if the
/// cache contains `fixed_norm` (the normalized fixed-image values), builds
/// the sparse cache from it, stores it in the cache for future use, and
/// returns it. This lazy construction reduces peak memory: on the first
/// cache-miss only the dense `w_fixed_transposed` tensor and the small
/// `fixed_norm` Vec are allocated; the ~2 MB sparse cache is deferred until
/// the sparse dispatch path is first requested.
#[cfg(feature = "direct-parzen")]
fn get_cached_sparse_w_fixed<B: Backend, const D: usize>(
    cache_guard: &mut Option<HistogramCache<B>>,
    fixed: &Image<B, D>,
    num_bins: usize,
    sigma_sq_fix: f32,
) -> Option<super::direct::SparseWFixedT> {
    let cache = cache_guard.as_mut()?;
    if !cache_matches_image(cache, fixed) {
        return None;
    }
    cache.get_or_build_sparse_w_fixed(num_bins, sigma_sq_fix)
}

/// Normalize fixed-image values for lazy sparse cache construction.
///
/// Returns the normalized `Vec<f32>` in `[0, num_bins - 1]` so it can be
/// stored in the cache and later used by `get_cached_sparse_w_fixed` to
/// build the sparse W_fixed^T on first access. This avoids eagerly
/// constructing the sparse cache (~2 MB) on every cache-miss; only the
/// ~128 KB `fixed_norm` Vec is stored up front.
#[cfg(feature = "direct-parzen")]
fn normalize_fixed_values<B: Backend>(
    fixed_values: &Tensor<B, 1>,
    min_intensity: f32,
    max_intensity: f32,
    num_bins: usize,
) -> Vec<f32> {
    super::dispatch::normalize_and_extract(fixed_values, min_intensity, max_intensity, num_bins)
}

/// Construct a HistogramCache with dense representation and normalized fixed values.
///
/// Two `make_cache` overloads exist because `HistogramCache.sparse_w_fixed` and
/// `HistogramCache.fixed_norm` are gated by `#[cfg(feature = "direct-parzen")]` on
/// the struct fields themselves. When that feature is off the fields simply do not
/// exist, so a single function cannot construct both variants — the struct literal
/// would fail to compile in one cfg or the other. The `not(direct-parzen)` overload
/// accepts `_fixed_norm: Option<()>` so callers can write the same call-site under
/// both cfgs without duplicating surrounding logic.
///
/// The sparse cache (`sparse_w_fixed`) is **not** built here — it is constructed
/// lazily by `get_cached_sparse_w_fixed` on first access from `fixed_norm`. This
/// reduces peak memory on the initial cache-miss from ~6.5 MB (dense + sparse)
/// to ~4.1 MB (dense + ~128 KB `fixed_norm` Vec).
#[cfg(feature = "direct-parzen")]
fn make_cache<B: Backend, const D: usize>(
    points: Tensor<B, 2>,
    w_fixed_transposed: Tensor<B, 2>,
    fixed: &Image<B, D>,
    fixed_norm: Option<Vec<f32>>,
) -> HistogramCache<B> {
    HistogramCache {
        points,
        w_fixed_transposed: Some(w_fixed_transposed),
        sparse_w_fixed: None,
        fixed_norm,
        shape: fixed.shape().to_vec(),
        origin: fixed.origin().0.iter().cloned().collect(),
        spacing: fixed.spacing().0.iter().cloned().collect(),
        direction: fixed.direction().0.iter().cloned().collect(),
    }
}

/// Construct a HistogramCache with only the dense representation.
///
/// See the `direct-parzen` overload doc-comment for why both versions must exist.
#[cfg(not(feature = "direct-parzen"))]
fn make_cache<B: Backend, const D: usize>(
    points: Tensor<B, 2>,
    w_fixed_transposed: Tensor<B, 2>,
    fixed: &Image<B, D>,
    _fixed_norm: Option<()>,
) -> HistogramCache<B> {
    HistogramCache {
        points,
        w_fixed_transposed: Some(w_fixed_transposed),
        shape: fixed.shape().to_vec(),
        origin: fixed.origin().0.iter().cloned().collect(),
        spacing: fixed.spacing().0.iter().cloned().collect(),
        direction: fixed.direction().0.iter().cloned().collect(),
    }
}

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

        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
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
                fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone())
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
                    interpolator.interpolate(fixed.data(), fixed_indices.clone().unwrap())
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
                    *cache = Some(make_cache(
                        fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone()),
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
            let num_chunks = n.div_ceil(CHUNK_SIZE);
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
                let pts = fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone());
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
                        *cache = Some(make_cache(
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
                    *cache = Some(make_cache(pts.clone(), w_fixed_t, fixed, fixed_norm));
                }
                pts
            } else {
                Tensor::zeros([1, 1], &device)
            };

            let have_all_points = !use_sampling;

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                let chunk_range = start..end;

                let chunk_fixed_points = if have_all_points {
                    all_fixed_points.clone().slice([chunk_range.clone()])
                } else {
                    let chunk_indices = fixed_indices
                        .as_ref()
                        .unwrap()
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
                            .unwrap()
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
                            .unwrap()
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
}

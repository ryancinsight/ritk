//! Chunked iteration strategy for image-level joint histogram computation.
//!
//! When the number of voxels `N` exceeds [`WGPU_CHUNK_SIZE`](crate::wgpu_compat::WGPU_CHUNK_SIZE),
//! the spatial domain is split into chunks so each batch fits within GPU dispatch
//! and memory limits. Each chunk independently computes its histogram contribution,
//! and results are accumulated into the final joint histogram.
//!
//! Extracted from `compute_image/mod.rs` (SRP-360-09) to bring that file below the
//! 500-line structural limit.

use super::super::image_cache_helpers::get_cached_w_fixed_t;
#[cfg(feature = "direct-parzen")]
use super::super::image_cache_helpers::{get_cached_sparse_w_fixed, normalize_fixed_values};
use super::super::ParzenJointHistogram;
use super::SamplingMode;
use crate::metric::histogram::cache;
use crate::metric::histogram::parzen::compute_oob_mask_3d;
#[cfg(feature = "direct-parzen")]
use crate::metric::histogram::parzen::direct;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

impl<B: Backend> ParzenJointHistogram<B> {
    /// Chunked histogram computation for [`compute_image_joint_histogram`](Self::compute_image_joint_histogram).
    ///
    /// Splits the spatial domain into chunks of size `WGPU_CHUNK_SIZE`, computes
    /// each chunk's histogram contribution via the best available dispatch path
    /// (sparse cache > dense cache > cache miss), and accumulates the results.
    ///
    /// # Arguments
    /// * `fixed`             — Fixed reference image (spatial metadata, interpolation).
    /// * `moving`            — Moving image.
    /// * `transform`         — Current candidate spatial transform.
    /// * `interpolator`      — Interpolator for sampling `moving`.
    /// * `n`                 — Total number of points.
    /// * `use_sampling`      — Whether stochastic subsampling is active.
    /// * `fixed_indices`     — Pre-generated random indices (`Some` when `use_sampling == SamplingMode::Sampled`).
    /// * `cached_points`     — Pre-extracted world-space points from cache (`Some` on cache hit).
    ///
    /// # Returns
    /// Joint histogram `[num_bins, num_bins]`.
    pub(super) fn compute_image_joint_histogram_chunked<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        n: usize,
        use_sampling: SamplingMode,
        fixed_indices: Option<Tensor<B, 2>>,
        cached_points: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let device = fixed.data().device();
        let num_chunks = n.div_ceil(crate::wgpu_compat::WGPU_CHUNK_SIZE);
        let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

        let cached_w_fixed_t = (use_sampling == SamplingMode::Dense)
            .then(|| {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                get_cached_w_fixed_t(&cache, fixed)
            })
            .flatten();

        #[cfg(feature = "direct-parzen")]
        let cached_sparse = (use_sampling == SamplingMode::Dense)
            .then(|| {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq();
                get_cached_sparse_w_fixed(&mut cache, fixed, self.num_bins, sigma_sq_fix)
            })
            .flatten();

        let all_fixed_points = if let Some(pts) = cached_points {
            pts
        } else if use_sampling == SamplingMode::Dense {
            let pts = fixed.index_to_world_tensor(
                fixed_indices
                    .as_ref()
                    .expect("fixed_indices must be Some when no cached points exist")
                    .clone(),
            );
            if let Some(w_fixed_t) = &cached_w_fixed_t {
                // Cache hit for W_fixed^T — only update cache if points are missing or
                // the cached tensor dimensions are stale. The sparse cache is not
                // passed here; it is built lazily from `fixed_norm` on first access.
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                if cache.as_ref().is_none_or(|c| c.points.dims() != [n, D]) {
                    #[cfg(feature = "direct-parzen")]
                    let fixed_norm_for_cache: Option<Vec<f32>> = None;
                    #[cfg(not(feature = "direct-parzen"))]
                    let fixed_norm_for_cache: Option<()> = None;
                    *cache = Some(cache::make_cache(
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
                *cache = Some(cache::make_cache(pts.clone(), w_fixed_t, fixed, fixed_norm));
            }
            pts
        } else {
            Tensor::zeros([1, 1], &device)
        };

        let have_all_points = use_sampling == SamplingMode::Dense;

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
                    Some(compute_oob_mask_3d(
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
            // 1. Sparse cache hit — iterate only ~7 non-zero fixed bins per sample.
            //    Slicing the sparse cache is trivial (just `sparse[start..end].to_vec()`)
            //    and avoids the old `w_fixed_t.clone().slice()` pattern that cloned
            //    the entire [num_bins × N] dense tensor (~4 MB for N=32K) just to
            //    extract one chunk. The sparse slice copies only ~56 bytes per sample.
            // 2. Dense cache hit — slice [num_bins, start..end] and use the tensor
            //    matmul path (autodiff-safe, needed for RSGD).
            // 3. Cache miss (sampling) — compute both W_fixed and W_moving from scratch.

            #[cfg(feature = "direct-parzen")]
            let chunk_hist = if let Some(ref sparse) = cached_sparse {
                if !chunk_moving_values.is_require_grad() {
                    let chunk_sparse: direct::SparseWFixedT = sparse[start..end].to_vec();
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
                    let chunk_fixed_values = if use_sampling == SamplingMode::Sampled {
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
                }
            } else if let Some(ref w_fixed_t) = cached_w_fixed_t {
                let chunk_w_fixed_t = w_fixed_t.clone().slice([0..self.num_bins, chunk_range]);
                self.compute_joint_histogram_from_cache_dispatch(
                    &chunk_w_fixed_t,
                    &chunk_moving_values,
                    chunk_oob.as_ref(),
                )
            } else {
                let chunk_fixed_values = if use_sampling == SamplingMode::Sampled {
                    let chunk_indices = fixed_indices
                        .as_ref()
                        .expect("fixed_indices must be Some in sampling chunk path (direct-parzen)")
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
                let chunk_fixed_values = if use_sampling == SamplingMode::Sampled {
                    let chunk_indices = fixed_indices
                        .as_ref()
                        .expect(
                            "fixed_indices must be Some in sampling chunk path (no direct-parzen)",
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

            joint_hist_acc = joint_hist_acc + chunk_hist;
        }

        joint_hist_acc
    }

    /// Chunked histogram computation for
    /// [`compute_image_joint_histogram_with_w_fixed`](Self::compute_image_joint_histogram_with_w_fixed).
    ///
    /// Splits the spatial domain into chunks of size `WGPU_CHUNK_SIZE` and
    /// accumulates each chunk's histogram using the dense `w_fixed_transposed`
    /// matmul path (autodiff-safe, works for both RSGD and CMA-ES).
    ///
    /// # Arguments
    /// * `fixed`                — Fixed reference image (spatial metadata).
    /// * `moving`               — Moving image.
    /// * `transform`            — Current candidate spatial transform.
    /// * `interpolator`         — Interpolator for sampling `moving`.
    /// * `w_fixed_transposed`   — Caller-supplied Parzen weight matrix `[num_bins, N]`.
    /// * `fixed_points`         — World-space points `[N, D]` (cached or freshly built).
    /// * `n`                    — Total number of points.
    ///
    /// # Returns
    /// Joint histogram `[num_bins, num_bins]`.
    pub(super) fn compute_image_joint_histogram_with_w_fixed_chunked<const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        interpolator: &LinearInterpolator,
        w_fixed_transposed: &Tensor<B, 2>,
        fixed_points: &Tensor<B, 2>,
        n: usize,
    ) -> Tensor<B, 2> {
        let device = fixed.data().device();
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
                    Some(compute_oob_mask_3d(
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

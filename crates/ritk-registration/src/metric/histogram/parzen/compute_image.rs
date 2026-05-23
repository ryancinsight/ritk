//! Image-level joint histogram computation with spatial transform and caching.
//!
//! Extracted from `compute.rs` to keep the 500-line structural limit.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

use super::super::cache::HistogramCache;
use super::{compute_oob_mask_3d, ParzenJointHistogram};

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
            // Check cache (slice comparison avoids heap allocation per iteration)
            let cached_points = {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(c) = cache.as_ref() {
                    let fs = fixed.shape();
                    if c.shape.as_slice() == &fs
                        && c.origin.iter().eq(fixed.origin().0.iter())
                        && c.spacing.iter().eq(fixed.spacing().0.iter())
                        && c.direction.iter().eq(fixed.direction().0.iter())
                    {
                        Some(c.points.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some(pts) = cached_points {
                (None, total_voxels, false, Some(pts))
            } else {
                let indices = grid::generate_grid(fixed_shape, &device);
                (Some(indices), total_voxels, false, None)
            }
        };

        // Use a chunk size that respects wgpu dispatch limits.
        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
            // ── Non-chunked path ──
            // For the non-sampling case, check if we have a cached W_fixed^T.
            // This avoids recomputing the O(N × num_bins) Parzen matrix for the
            // constant fixed image on every registration iteration.
            let cached_w_fixed_t: Option<Tensor<B, 2>> = if !use_sampling {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.as_ref().and_then(|c| {
                    let fs = fixed.shape();
                    let matches = c.shape.as_slice() == &fs
                        && c.origin.iter().eq(fixed.origin().0.iter())
                        && c.spacing.iter().eq(fixed.spacing().0.iter())
                        && c.direction.iter().eq(fixed.direction().0.iter());
                    if matches {
                        c.w_fixed_transposed.clone()
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            let fixed_points = if let Some(pts) = cached_points {
                pts
            } else {
                fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone())
            };

            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);

            // Compute OOB mask before consuming moving_indices.
            let oob_mask: Option<Tensor<B, 1>> = if D == 3 {
                let shape_arr = moving.shape();
                Some(compute_oob_mask_3d(&moving_indices, shape_arr.as_ref()))
            } else {
                None
            };

            let moving_values = interpolator.interpolate(moving.data(), moving_indices);

            if let Some(w_fixed_t) = cached_w_fixed_t {
                self.compute_joint_histogram_from_cache(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            } else {
                // When sampling, fixed_values come from interpolation at sample points.
                // When not sampling, fixed_values are the full flattened image data.
                let fixed_values = if use_sampling {
                    interpolator.interpolate(fixed.data(), fixed_indices.clone().unwrap())
                } else {
                    fixed.data().clone().reshape([n])
                };
                let w_fixed_t = self.compute_w_fixed_transposed(&fixed_values, n);
                if !use_sampling {
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    *cache = Some(HistogramCache {
                        points: fixed
                            .index_to_world_tensor(fixed_indices.as_ref().unwrap().clone()),
                        w_fixed_transposed: Some(w_fixed_t.clone()),
                        shape: fixed.shape().to_vec(),
                        origin: fixed.origin().0.iter().cloned().collect(),
                        spacing: fixed.spacing().0.iter().cloned().collect(),
                        direction: fixed.direction().0.iter().cloned().collect(),
                    });
                }
                self.compute_joint_histogram_from_cache(
                    &w_fixed_t,
                    &moving_values,
                    oob_mask.as_ref(),
                )
            }
        } else {
            // ── Chunked path ──
            let num_chunks = n.div_ceil(CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            // ── Fixed-image cache for the chunked path ─────────────────────────
            //
            // When not using stochastic sampling, the fixed image is constant across
            // all registration iterations. Computing W_fixed^T once and caching it
            // eliminates O(N × num_bins) Parzen weight recomputation per CMA-ES
            // objective evaluation — a significant speedup for large volumes.
            //
            // On the first call (cache miss), we compute the full-N W_fixed^T and
            // store it. On subsequent calls (cache hit), we slice it per-chunk and
            // use `compute_joint_histogram_from_cache` instead of recomputing from
            // fixed values.
            let cached_w_fixed_t: Option<Tensor<B, 2>> = if !use_sampling {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.as_ref().and_then(|c| {
                    let fs = fixed.shape();
                    let matches = c.shape.as_slice() == &fs
                        && c.origin.iter().eq(fixed.origin().0.iter())
                        && c.spacing.iter().eq(fixed.spacing().0.iter())
                        && c.direction.iter().eq(fixed.direction().0.iter());
                    if matches {
                        c.w_fixed_transposed.clone()
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            let all_fixed_points = if let Some(pts) = cached_points {
                pts
            } else if !use_sampling {
                let pts = fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone());

                // If W_fixed^T is not yet cached, compute it now and store it.
                if cached_w_fixed_t.is_none() {
                    let fixed_data_flat = fixed.data().clone().reshape([n]);
                    let w_fixed_t = self.compute_w_fixed_transposed(&fixed_data_flat, n);
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    *cache = Some(HistogramCache {
                        points: pts.clone(),
                        w_fixed_transposed: Some(w_fixed_t),
                        shape: fixed.shape().to_vec(),
                        origin: fixed.origin().0.iter().cloned().collect(),
                        spacing: fixed.spacing().0.iter().cloned().collect(),
                        direction: fixed.direction().0.iter().cloned().collect(),
                    });
                } else {
                    // W_fixed^T is already cached; just cache the points.
                    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                    if cache.as_ref().is_none_or(|c| c.points.dims() != [n, D]) {
                        *cache = Some(HistogramCache {
                            points: pts.clone(),
                            w_fixed_transposed: cached_w_fixed_t.clone(),
                            shape: fixed.shape().to_vec(),
                            origin: fixed.origin().0.iter().cloned().collect(),
                            spacing: fixed.spacing().0.iter().cloned().collect(),
                            direction: fixed.direction().0.iter().cloned().collect(),
                        });
                    }
                }
                pts
            } else {
                // Sampling, no caching, use per-chunk computation
                Tensor::zeros([1, 1], &device)
            };

            let have_all_points = !use_sampling;

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);

                // Pipeline for this chunk
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

                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);

                // Compute per-chunk OOB mask before consuming chunk_moving_indices.
                let chunk_oob: Option<Tensor<B, 1>> = if D == 3 {
                    let shape_arr = moving.shape();
                    Some(compute_oob_mask_3d(
                        &chunk_moving_indices,
                        shape_arr.as_ref(),
                    ))
                } else {
                    None
                };

                let chunk_moving_values =
                    interpolator.interpolate(moving.data(), chunk_moving_indices);

                // ── Chunked histogram computation ──────────────────────────────────
                // When W_fixed^T is cached, slice it per-chunk and use the faster
                // `compute_joint_histogram_from_cache` path, which only recomputes
                // the moving-image Parzen weights (O(chunk × bins) instead of
                // O(2 × chunk × bins)). This halves the per-chunk Parzen computation
                // and eliminates the fixed-image autodiff graph overhead.
                let chunk_hist = if let Some(ref w_fixed_t) = cached_w_fixed_t {
                    // Cache hit: slice [num_bins, start..end] from full-N W_fixed^T
                    let chunk_w_fixed_t = w_fixed_t.clone().slice([0..self.num_bins, chunk_range]);
                    self.compute_joint_histogram_from_cache(
                        &chunk_w_fixed_t,
                        &chunk_moving_values,
                        chunk_oob.as_ref(),
                    )
                } else {
                    // Cache miss (sampling path): compute both W_fixed and W_moving
                    let chunk_fixed_values = if use_sampling {
                        let chunk_indices = fixed_indices
                            .as_ref()
                            .unwrap()
                            .clone()
                            .slice([chunk_range.clone()]);
                        interpolator.interpolate(fixed.data(), chunk_indices)
                    } else {
                        // Fallback: fixed data not yet cached (should not happen after
                        // the cache population above, but kept for robustness).
                        fixed.data().clone().reshape([n]).slice([chunk_range])
                    };
                    self.compute_joint_histogram(
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

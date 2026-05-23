use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;

use super::super::cache::HistogramCache;
use super::{compute_oob_mask_3d, ParzenJointHistogram};

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute the transposed Parzen weight matrix `W_fixed^T [num_bins, N]` for fixed-image values.
    ///
    /// This is the constant (non-autodiff) matrix that only depends on the fixed
    /// image and can be computed once and cached across all registration iterations.
    /// Used by both the non-chunked and chunked paths of `compute_image_joint_histogram`.
    fn compute_w_fixed_transposed(&self, fixed_values: &Tensor<B, 1>, n: usize) -> Tensor<B, 2> {
        let device = fixed_values.device();
        // Convert parzen_sigma from intensity units to bin-index units.
        let bin_width_intensity =
            (self.max_intensity - self.min_intensity) / (self.num_bins as f32 - 1.0).max(1.0);
        let sigma_in_bins = self.parzen_sigma / bin_width_intensity.max(f32::EPSILON);
        let sigma_sq = sigma_in_bins * sigma_in_bins;

        let fixed_norm = {
            let t = fixed_values.clone() - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
            let t = t * (self.num_bins as f32 - 1.0);
            t.clamp(0.0, self.num_bins as f32 - 1.0)
        };

        let bins_exp = Tensor::<B, 1, Int>::arange(0..self.num_bins as i64, &device)
            .float()
            .reshape([1, self.num_bins]);

        let vals_exp = fixed_norm.reshape([n, 1]);
        let diff = vals_exp - bins_exp;
        (diff.powf_scalar(2.0) * (-0.5 / sigma_sq))
            .exp()
            .transpose() // [num_bins, N]
    }

    /// Compute joint histogram from a precomputed W_fixed^T [num_bins, N] and live moving values [N].
    ///
    /// `w_fixed_transposed` is the transposed Parzen weight matrix for the fixed image,
    /// precomputed once and cached. `moving_values` carries the autodiff gradient path.
    /// Only the moving-image side needs recomputation each iteration.
    ///
    /// `oob_mask` is an optional `[N]` float tensor (`1.0` = in-bounds, `0.0` = out-of-bounds).
    /// When provided, OOB samples are zeroed out of W_moving before the histogram matmul.
    pub(super) fn compute_joint_histogram_from_cache(
        &self,
        w_fixed_transposed: &Tensor<B, 2>, // [num_bins, N]
        moving_values: &Tensor<B, 1>,      // [N]
        oob_mask: Option<&Tensor<B, 1>>,   // [N] in-bounds mask (1.0=in, 0.0=out)
    ) -> Tensor<B, 2> {
        let device = moving_values.device();
        let [n] = moving_values.dims();
        let num_bins = self.num_bins;

        // Moving-image normalization parameters (fall back to fixed-image range when not set).
        let mov_min = self.moving_min_intensity.unwrap_or(self.min_intensity);
        let mov_max = self.moving_max_intensity.unwrap_or(self.max_intensity);
        let mov_sigma = self.moving_parzen_sigma.unwrap_or(self.parzen_sigma);

        // Express `mov_sigma` in bin-index units so the Parzen kernel is correctly
        // scaled relative to the moving-image's own intensity range.
        let bin_width_mov = (mov_max - mov_min) / (num_bins as f32 - 1.0).max(1.0);
        let sigma_in_bins = mov_sigma / bin_width_mov.max(f32::EPSILON);
        let sigma_sq = sigma_in_bins * sigma_in_bins;

        // Normalize moving values to [0, num_bins-1] using the moving-image range.
        let moving_norm = {
            let t = moving_values.clone() - mov_min;
            let t = t / (mov_max - mov_min);
            let t = t * (num_bins as f32 - 1.0);
            t.clamp(0.0, num_bins as f32 - 1.0)
        };

        // Bin centers [1, num_bins]
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &device)
            .float()
            .reshape([1, num_bins]);

        // W_moving [N, num_bins] = exp(-0.5 * ((val - bin) / sigma)^2)
        let vals_exp = moving_norm.reshape([n, 1]);
        let diff = vals_exp - bins_exp;
        let w_moving = (diff.powf_scalar(2.0) * (-0.5 / sigma_sq)).exp();

        // Apply OOB mask: zero out rows for out-of-bounds samples.
        let w_moving = if let Some(mask) = oob_mask {
            w_moving * mask.clone().reshape([n, 1])
        } else {
            w_moving
        };

        // Joint histogram [num_bins, num_bins] = W_fixed^T @ W_moving
        w_fixed_transposed.clone().matmul(w_moving)
    }

    /// Compute soft joint histogram between two images (vectorized).
    /// Uses Gaussian kernel for differentiability.
    pub fn compute_joint_histogram(
        &self,
        fixed: &Tensor<B, 1>,
        moving: &Tensor<B, 1>,
        oob_mask: Option<&Tensor<B, 1>>,
    ) -> Tensor<B, 2> {
        let device = fixed.device();
        let [n] = fixed.dims();
        let num_bins = self.num_bins;
        let num_bins_f = num_bins as f32 - 1.0;

        // Fixed-image normalization parameters.
        let fix_min = self.min_intensity;
        let fix_max = self.max_intensity;
        let fix_sigma = self.parzen_sigma;

        // Moving-image normalization parameters — independent when a separate range is set,
        // otherwise fall back to the fixed-image range (backward-compatible).
        let mov_min = self.moving_min_intensity.unwrap_or(fix_min);
        let mov_max = self.moving_max_intensity.unwrap_or(fix_max);
        let mov_sigma = self.moving_parzen_sigma.unwrap_or(fix_sigma);

        // Normalize intensities to [0, num_bins-1] using the supplied range.
        let normalize = |t: Tensor<B, 1>, min: f32, max: f32| -> Tensor<B, 1> {
            let t = t - min;
            let t = t / (max - min);
            let t = t * num_bins_f;
            t.clamp(0.0, num_bins_f)
        };

        // Create bin centers [1, Bins]
        let bins = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &device).float();
        let bins_exp = bins.reshape([1, num_bins]);

        // Convert sigma from intensity units → bin-index units for each image.
        let bin_width_fix = (fix_max - fix_min) / num_bins_f.max(1.0);
        let sigma_sq_fix = (fix_sigma / bin_width_fix.max(f32::EPSILON)).powi(2);
        let bin_width_mov = (mov_max - mov_min) / num_bins_f.max(1.0);
        let sigma_sq_mov = (mov_sigma / bin_width_mov.max(f32::EPSILON)).powi(2);

        // Vectorized Weight Computation
        // weights: [N, Bins]
        // W[i, b] = exp(-0.5 * ((val[i] - b) / sigma)^2)
        let compute_weights = |vals: Tensor<B, 1>, size: usize, sigma_sq: f32| -> Tensor<B, 2> {
            let vals_exp = vals.reshape([size, 1]); // [N, 1]
            let diff = vals_exp - bins_exp.clone(); // [N, Bins]
            let exponent = diff.powf_scalar(2.0) * (-0.5 / sigma_sq);
            exponent.exp()
        };

        // WGPU dispatch limit workaround
        // The matmul (Bins, N) * (N, Bins) reduces along N.
        // If N is too large, it exceeds dispatch limits.
        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
            let fixed_norm = normalize(fixed.clone(), fix_min, fix_max);
            let moving_norm = normalize(moving.clone(), mov_min, mov_max);

            let w_fixed = compute_weights(fixed_norm, n, sigma_sq_fix);
            let w_moving = compute_weights(moving_norm, n, sigma_sq_mov);

            // Apply OOB mask: zero out rows for out-of-bounds samples.
            let w_moving = if let Some(mask) = oob_mask {
                w_moving * mask.clone().reshape([n, 1])
            } else {
                w_moving
            };

            w_fixed.transpose().matmul(w_moving)
        } else {
            let mut joint_hist = Tensor::<B, 2>::zeros([num_bins, num_bins], &device);
            let num_chunks = n.div_ceil(CHUNK_SIZE);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                let current_chunk_size = end - start;
                let chunk_range = start..end;

                let fixed_chunk = fixed.clone().slice([chunk_range.clone()]);
                let moving_chunk = moving.clone().slice([chunk_range]);

                let fixed_norm = normalize(fixed_chunk, fix_min, fix_max);
                let moving_norm = normalize(moving_chunk, mov_min, mov_max);

                let w_fixed = compute_weights(fixed_norm, current_chunk_size, sigma_sq_fix);
                let w_moving = compute_weights(moving_norm, current_chunk_size, sigma_sq_mov);

                // Apply per-chunk OOB mask if provided.
                let w_moving = match oob_mask {
                    Some(m) => {
                        w_moving
                            * m.clone()
                                .slice([start..end])
                                .reshape([current_chunk_size, 1])
                    }
                    None => w_moving,
                };

                joint_hist = joint_hist + w_fixed.transpose().matmul(w_moving);
            }
            joint_hist
        }
    }

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
            // Check cache
            let cached_points = {
                let cache = self.cache.lock().unwrap();
                if let Some(c) = cache.as_ref() {
                    let current_shape = fixed.shape().to_vec();
                    let current_origin: Vec<f64> = fixed.origin().0.iter().cloned().collect();
                    let current_spacing: Vec<f64> = fixed.spacing().0.iter().cloned().collect();
                    let current_direction: Vec<f64> = fixed.direction().0.iter().cloned().collect();
                    if c.shape == current_shape
                        && c.origin == current_origin
                        && c.spacing == current_spacing
                        && c.direction == current_direction
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
                let cache = self.cache.lock().unwrap();
                cache.as_ref().and_then(|c| {
                    let matches = c.shape == fixed.shape().to_vec()
                        && c.origin == fixed.origin().0.iter().cloned().collect::<Vec<f64>>()
                        && c.spacing == fixed.spacing().0.iter().cloned().collect::<Vec<f64>>()
                        && c.direction == fixed.direction().0.iter().cloned().collect::<Vec<f64>>();
                    if matches {
                        c.w_fixed_transposed.clone()
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            // Determine fixed world-space points (from cache or freshly computed).
            let fixed_points = if let Some(pts) = cached_points {
                pts
            } else {
                fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone())
            };

            let moving_points = transform.transform_points(fixed_points.clone());
            let moving_indices = moving.world_to_index_tensor(moving_points);

            // Compute OOB mask before consuming moving_indices (uses immutable borrow).
            let oob_mask: Option<Tensor<B, 1>> = if D == 3 {
                let shape_arr = moving.shape();
                Some(compute_oob_mask_3d(&moving_indices, shape_arr.as_ref()))
            } else {
                None
            };

            let moving_values = interpolator.interpolate(moving.data(), moving_indices);

            if !use_sampling {
                if let Some(ref w_ft) = cached_w_fixed_t {
                    // Cache hit: W_fixed^T already computed; only W_moving needs autodiff.
                    return self.compute_joint_histogram_from_cache(
                        w_ft,
                        &moving_values,
                        oob_mask.as_ref(),
                    );
                }
            }

            // Cache miss (or sampling): compute fixed_values and W_fixed, then populate cache.
            let fixed_values = if use_sampling {
                interpolator.interpolate(fixed.data(), fixed_indices.unwrap())
            } else {
                fixed.data().clone().reshape([n])
            };

            // For the non-sampling path, precompute W_fixed^T and store alongside points.
            if !use_sampling {
                let w_fixed_t = self.compute_w_fixed_transposed(&fixed_values, n);

                let mut cache = self.cache.lock().unwrap();
                *cache = Some(HistogramCache {
                    points: fixed_points,
                    w_fixed_transposed: Some(w_fixed_t),
                    shape: fixed.shape().to_vec(),
                    origin: fixed.origin().0.iter().cloned().collect(),
                    spacing: fixed.spacing().0.iter().cloned().collect(),
                    direction: fixed.direction().0.iter().cloned().collect(),
                });
            }

            self.compute_joint_histogram(&fixed_values, &moving_values, oob_mask.as_ref())
        } else {
            // Process in chunks
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
                let cache = self.cache.lock().unwrap();
                cache.as_ref().and_then(|c| {
                    let matches = c.shape == fixed.shape().to_vec()
                        && c.origin == fixed.origin().0.iter().cloned().collect::<Vec<f64>>()
                        && c.spacing == fixed.spacing().0.iter().cloned().collect::<Vec<f64>>()
                        && c.direction == fixed.direction().0.iter().cloned().collect::<Vec<f64>>();
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
                    let mut cache = self.cache.lock().unwrap();
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
                    let mut cache = self.cache.lock().unwrap();
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

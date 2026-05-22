//! Histogram computation utilities for Mutual Information metrics.
//!
//! This module provides shared implementations for differentiable soft histogramming
//! using Parzen windowing.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ritk_core::image::grid;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct HistogramCache<B: Backend> {
    /// World-space coordinates of all fixed-image voxels [N, D].
    points: Tensor<B, 2>,
    /// Precomputed Parzen weight matrix for the fixed image, transposed: [num_bins, N].
    /// Constant across all registration iterations because the fixed image never changes.
    /// Reusing this avoids O(N × num_bins) kernel computation and removes the fixed-image
    /// Parzen path from the autodiff graph on every iteration after the first.
    w_fixed_transposed: Option<Tensor<B, 2>>,
    shape: Vec<usize>,
    origin: Vec<f64>,
    spacing: Vec<f64>,
    direction: Vec<f64>,
}

/// Joint Histogram Calculator using Parzen windowing.
#[derive(Clone, Debug)]
pub struct ParzenJointHistogram<B: Backend> {
    /// Number of histogram bins
    pub num_bins: usize,
    /// Minimum intensity value
    pub min_intensity: f32,
    /// Maximum intensity value
    pub max_intensity: f32,
    /// Parzen window sigma for histogram smoothing
    pub parzen_sigma: f32,
    /// Cache for fixed image points to avoid recomputation
    cache: Arc<Mutex<Option<HistogramCache<B>>>>,
    /// Phantom data
    _phantom: PhantomData<B>,
}

impl<B: Backend> ParzenJointHistogram<B> {
    /// Create a new Parzen Joint Histogram calculator.
    pub fn new(num_bins: usize, min_intensity: f32, max_intensity: f32, parzen_sigma: f32) -> Self {
        Self {
            num_bins,
            min_intensity,
            max_intensity,
            parzen_sigma,
            cache: Arc::new(Mutex::new(None)),
            _phantom: PhantomData,
        }
    }

    /// Compute joint histogram from a precomputed W_fixed^T [num_bins, N] and live moving values [N].
    ///
    /// `w_fixed_transposed` is the transposed Parzen weight matrix for the fixed image,
    /// precomputed once and cached. `moving_values` carries the autodiff gradient path.
    /// Only the moving-image side needs recomputation each iteration.
    fn compute_joint_histogram_from_cache(
        &self,
        w_fixed_transposed: &Tensor<B, 2>, // [num_bins, N]
        moving_values: &Tensor<B, 1>,      // [N]
    ) -> Tensor<B, 2> {
        let device = moving_values.device();
        let [n] = moving_values.dims();
        let num_bins = self.num_bins;
        // `normalize()` maps intensity → bin-index units [0, num_bins-1].
        // One bin-index unit corresponds to `bin_width_intensity` in intensity space.
        // We must express `parzen_sigma` in the same bin-index units before squaring,
        // otherwise the Parzen kernel is mis-scaled (too wide for large intensity
        // ranges such as CT HU, too narrow for [0,1] inputs).
        let bin_width_intensity =
            (self.max_intensity - self.min_intensity) / (num_bins as f32 - 1.0).max(1.0);
        let sigma_in_bins = self.parzen_sigma / bin_width_intensity.max(f32::EPSILON);
        let sigma_sq = sigma_in_bins * sigma_in_bins;

        // Normalize moving values to [0, num_bins-1]
        let moving_norm = {
            let t = moving_values.clone() - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
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

        // Joint histogram [num_bins, num_bins] = W_fixed^T @ W_moving
        w_fixed_transposed.clone().matmul(w_moving)
    }

    /// Compute soft joint histogram between two images (vectorized).
    /// Uses Gaussian kernel for differentiability.
    pub fn compute_joint_histogram(
        &self,
        fixed: &Tensor<B, 1>,
        moving: &Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let device = fixed.device();
        let [n] = fixed.dims();

        // Normalize intensities to [0, num_bins-1]
        let normalize = |t: Tensor<B, 1>| -> Tensor<B, 1> {
            let t = t - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
            let t = t * (self.num_bins as f32 - 1.0);
            t.clamp(0.0, self.num_bins as f32 - 1.0)
        };

        // Create bin centers [Bins]
        let bins = Tensor::<B, 1, Int>::arange(0..self.num_bins as i64, &device).float();
        let bins_exp = bins.clone().reshape([1, self.num_bins]); // [1, Bins]
                                                                 // Convert sigma from raw intensity units → bin-index units so that the
                                                                 // Parzen kernel width is consistent regardless of the image intensity range.
        let bin_width_intensity =
            (self.max_intensity - self.min_intensity) / (self.num_bins as f32 - 1.0).max(1.0);
        let sigma_in_bins = self.parzen_sigma / bin_width_intensity.max(f32::EPSILON);
        let sigma_sq = sigma_in_bins * sigma_in_bins;

        // Vectorized Weight Computation
        // weights: [N, Bins]
        // Use Gaussian kernel for Parzen windowing:
        // W[i, b] = exp(-0.5 * ((val[i] - b) / sigma)^2)

        let compute_weights = |vals: Tensor<B, 1>, size: usize| -> Tensor<B, 2> {
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
            let fixed_norm = normalize(fixed.clone());
            let moving_norm = normalize(moving.clone());

            let w_fixed = compute_weights(fixed_norm, n);
            let w_moving = compute_weights(moving_norm, n);

            w_fixed.transpose().matmul(w_moving)
        } else {
            let mut joint_hist = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);
            let num_chunks = n.div_ceil(CHUNK_SIZE);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                let current_chunk_size = end - start;

                let chunk_range = start..end;
                let fixed_chunk = fixed.clone().slice([chunk_range.clone()]);
                let moving_chunk = moving.clone().slice([chunk_range]);

                let fixed_norm = normalize(fixed_chunk);
                let moving_norm = normalize(moving_chunk);

                let w_fixed = compute_weights(fixed_norm, current_chunk_size);
                let w_moving = compute_weights(moving_norm, current_chunk_size);

                joint_hist = joint_hist + w_fixed.transpose().matmul(w_moving);
            }
            joint_hist
        }
    }

    /// Compute Entropy of a distribution P.
    pub fn compute_entropy(&self, p: Tensor<B, 1>) -> Tensor<B, 1> {
        let eps = 1e-10;
        let log_p = (p.clone() + eps).log();

        p.mul(log_p).sum().neg()
    }

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
            let moving_values = interpolator.interpolate(moving.data(), moving_voxel_indices);

            self.compute_joint_histogram(&fixed_values, &moving_values)
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
                let chunk_moving_vals = interpolator.interpolate(moving.data(), chunk_moving_idx);

                joint_hist_acc = joint_hist_acc
                    + self.compute_joint_histogram(&chunk_fixed_vals, &chunk_moving_vals);
            }
            joint_hist_acc
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
            let moving_values = interpolator.interpolate(moving.data(), moving_indices);

            if !use_sampling {
                if let Some(ref w_ft) = cached_w_fixed_t {
                    // Cache hit: W_fixed^T already computed; only W_moving needs autodiff.
                    return self.compute_joint_histogram_from_cache(w_ft, &moving_values);
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
                // Same sigma normalisation as in compute_joint_histogram / from_cache:
                // convert parzen_sigma from intensity units to bin-index units.
                let bin_width_intensity = (self.max_intensity - self.min_intensity)
                    / (self.num_bins as f32 - 1.0).max(1.0);
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
                let w_fixed_t = (diff.powf_scalar(2.0) * (-0.5 / sigma_sq))
                    .exp()
                    .transpose(); // [num_bins, N]

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

            self.compute_joint_histogram(&fixed_values, &moving_values)
        } else {
            // Process in chunks
            let num_chunks = n.div_ceil(CHUNK_SIZE);
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);

            // Populate cache if needed (and we are not using cached points yet)
            let all_fixed_points = if let Some(pts) = cached_points {
                pts
            } else if !use_sampling {
                // Compute all points and cache (chunk path: W_fixed^T not cached per-chunk
                // because the chunk boundary sizes vary; full-N W_fixed^T is not assembled here).
                let pts = fixed.index_to_world_tensor(fixed_indices.as_ref().unwrap().clone());
                let mut cache = self.cache.lock().unwrap();
                *cache = Some(HistogramCache {
                    points: pts.clone(),
                    w_fixed_transposed: None, // chunk path does not cache W_fixed^T
                    shape: fixed.shape().to_vec(),
                    origin: fixed.origin().0.iter().cloned().collect(),
                    spacing: fixed.spacing().0.iter().cloned().collect(),
                    direction: fixed.direction().0.iter().cloned().collect(),
                });
                pts
            } else {
                // Sampling, no caching, use per-chunk computation
                Tensor::zeros([1, 1], &device)
            };

            // Flatten fixed data once if not sampling
            let fixed_data_flat = if !use_sampling {
                Some(fixed.data().clone().reshape([n]))
            } else {
                None
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
                let chunk_moving_values =
                    interpolator.interpolate(moving.data(), chunk_moving_indices);

                // Get fixed values
                let chunk_fixed_values = if use_sampling {
                    let chunk_indices = fixed_indices
                        .as_ref()
                        .unwrap()
                        .clone()
                        .slice([chunk_range.clone()]);
                    interpolator.interpolate(fixed.data(), chunk_indices)
                } else {
                    fixed_data_flat
                        .as_ref()
                        .unwrap()
                        .clone()
                        .slice([chunk_range])
                };

                // Compute partial histogram
                let chunk_hist =
                    self.compute_joint_histogram(&chunk_fixed_values, &chunk_moving_values);
                joint_hist_acc = joint_hist_acc + chunk_hist;
            }
            joint_hist_acc
        }
    }
}

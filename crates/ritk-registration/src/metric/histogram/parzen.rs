use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::transform::Transform;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use super::cache::HistogramCache;

/// Joint Histogram Calculator using Parzen windowing.
#[derive(Clone, Debug)]
pub struct ParzenJointHistogram<B: Backend> {
    /// Number of histogram bins
    pub num_bins: usize,
    /// Minimum intensity value (fixed-image axis)
    pub min_intensity: f32,
    /// Maximum intensity value (fixed-image axis)
    pub max_intensity: f32,
    /// Parzen window sigma for histogram smoothing (fixed-image axis)
    pub parzen_sigma: f32,
    /// Optional separate minimum intensity for the moving image.
    /// When `None`, falls back to `min_intensity` (shared-range behaviour).
    pub moving_min_intensity: Option<f32>,
    /// Optional separate maximum intensity for the moving image.
    /// When `None`, falls back to `max_intensity` (shared-range behaviour).
    pub moving_max_intensity: Option<f32>,
    /// Optional separate Parzen sigma for the moving image.
    /// When `None`, falls back to `parzen_sigma`.
    pub moving_parzen_sigma: Option<f32>,
    /// Cache for fixed image points to avoid recomputation
    pub(super) cache: Arc<Mutex<Option<HistogramCache<B>>>>,
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
            moving_min_intensity: None,
            moving_max_intensity: None,
            moving_parzen_sigma: None,
            cache: Arc::new(Mutex::new(None)),
            _phantom: PhantomData,
        }
    }

    /// Configure a separate intensity range for the moving image (elastix-style independent binning).
    ///
    /// When set, each axis of the joint histogram uses its own normalization:
    /// the fixed axis spans `[min_intensity, max_intensity]` and the moving axis
    /// spans `[moving_min, moving_max]`, giving each image the full bin resolution.
    ///
    /// `moving_parzen_sigma` is set to `(moving_max - moving_min).max(1e-6) / num_bins`
    /// (Mattes parameterization: sigma = bin_width).
    pub fn with_separate_moving_range(mut self, moving_min: f32, moving_max: f32) -> Self {
        let sigma = (moving_max - moving_min).max(1e-6) / self.num_bins as f32;
        self.moving_min_intensity = Some(moving_min);
        self.moving_max_intensity = Some(moving_max);
        self.moving_parzen_sigma = Some(sigma);
        self
    }

    /// Compute joint histogram from a precomputed W_fixed^T [num_bins, N] and live moving values [N].
    ///
    /// `w_fixed_transposed` is the transposed Parzen weight matrix for the fixed image,
    /// precomputed once and cached. `moving_values` carries the autodiff gradient path.
    /// Only the moving-image side needs recomputation each iteration.
    ///
    /// `oob_mask` is an optional `[N]` float tensor (`1.0` = in-bounds, `0.0` = out-of-bounds).
    /// When provided, OOB samples are zeroed out of W_moving before the histogram matmul.
    fn compute_joint_histogram_from_cache(
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
                    Some(m) => w_moving * m.clone().slice([start..end]).reshape([current_chunk_size, 1]),
                    None => w_moving,
                };

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
                    return self.compute_joint_histogram_from_cache(w_ft, &moving_values, oob_mask.as_ref());
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

            self.compute_joint_histogram(&fixed_values, &moving_values, oob_mask.as_ref())
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
                // Compute per-chunk OOB mask before consuming chunk_moving_indices.
                let chunk_oob: Option<Tensor<B, 1>> = if D == 3 {
                    let shape_arr = moving.shape();
                    Some(compute_oob_mask_3d(&chunk_moving_indices, shape_arr.as_ref()))
                } else {
                    None
                };
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
                    self.compute_joint_histogram(&chunk_fixed_values, &chunk_moving_values, chunk_oob.as_ref());
                joint_hist_acc = joint_hist_acc + chunk_hist;
            }

            joint_hist_acc
        }
    }
}

/// Compute a `{0.0, 1.0}` in-bounds mask for 3-D moving-image voxel indices.
///
/// Returns an `[N]` float tensor: `1.0` = in-bounds, `0.0` = out-of-bounds.
/// Mirrors the zero-pad criterion in `LinearInterpolator`: a sample is
/// in-bounds when `floor(coord_d) ∈ [0, dim_d − 1]` for every axis.
///
/// Column convention (matches `interpolation::linear::dim3`):
/// - column 0 → x (→ `shape[2]`, the X / last dimension)
/// - column 1 → y (→ `shape[1]`, the Y / middle dimension)
/// - column 2 → z (→ `shape[0]`, the Z / first dimension)
pub(super) fn compute_oob_mask_3d<B: Backend>(
    indices: &Tensor<B, 2>, // [N, 3]
    shape: &[usize],        // at least 3 elements: [d0=Z, d1=Y, d2=X]
) -> Tensor<B, 1> {
    let d0 = shape[0]; // Z
    let d1 = shape[1]; // Y
    let d2 = shape[2]; // X

    let x = indices.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
    let z = indices.clone().narrow(1, 2, 1).squeeze_dims(&[1]);

    let x0 = x.clone().floor();
    let y0 = y.clone().floor();
    let z0 = z.clone().floor();

    let x_in = x0.clone().equal(x0.clamp(0.0, (d2 - 1) as f64)).float();
    let y_in = y0.clone().equal(y0.clamp(0.0, (d1 - 1) as f64)).float();
    let z_in = z0.clone().equal(z0.clamp(0.0, (d0 - 1) as f64)).float();

    x_in * y_in * z_in
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn device() -> <B as burn::tensor::backend::Backend>::Device {
        Default::default()
    }

    // ─── compute_oob_mask_3d ─────────────────────────────────────────────────

    #[test]
    fn oob_mask_3d_in_bounds_all_ones() {
        // 4×4×4 volume; every coordinate strictly inside returns 1.0
        let dev = device();
        // [x=1.5, y=1.5, z=1.5] — floor = [1,1,1], dims = [4,4,4] → in-bounds on all axes
        let indices = Tensor::<B, 2>::from_floats(
            [[1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            &dev,
        );
        let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
        let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(vals, vec![1.0, 1.0, 1.0], "all in-bounds coords must give 1.0");
    }

    #[test]
    fn oob_mask_3d_oob_all_zeros() {
        let dev = device();
        // x=-1 (OOB), y=5 > d1-1=3 (OOB), z=-0.1 → floor=-1 (OOB)
        let indices = Tensor::<B, 2>::from_floats(
            [[-1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, -0.1]],
            &dev,
        );
        let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
        let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(vals, vec![0.0, 0.0, 0.0], "all OOB coords must give 0.0");
    }

    #[test]
    fn oob_mask_3d_mixed_in_and_out() {
        let dev = device();
        // shape [Z=2, Y=4, X=4]: valid x in [0,3], y in [0,3], z in [0,1]
        let indices = Tensor::<B, 2>::from_floats(
            [
                [1.5, 1.5, 0.5],  // in-bounds
                [-0.5, 1.5, 0.5], // x OOB (floor=-1)
                [1.5, 4.0, 0.5],  // y OOB (floor=4 > 3)
                [1.5, 1.5, 2.0],  // z OOB (floor=2 > 1)
                [3.0, 3.0, 1.0],  // boundary, in-bounds
            ],
            &dev,
        );
        let mask = compute_oob_mask_3d(&indices, &[2, 4, 4]);
        let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(
            vals,
            vec![1.0, 0.0, 0.0, 0.0, 1.0],
            "mixed: in=1 OOB=0, boundary is in-bounds"
        );
    }

    // ─── compute_joint_histogram with OOB mask ───────────────────────────────

    #[test]
    fn oob_mask_zeros_out_oob_contribution() {
        // Verify that applying an all-zero OOB mask produces a zero histogram.
        let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
        let dev = device();

        let fixed = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
        let moving = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
        let all_oob = Tensor::<B, 1>::zeros([3], &dev); // all samples are OOB

        let h = hist.compute_joint_histogram(&fixed, &moving, Some(&all_oob));
        let sum: f32 = h.into_data().as_slice::<f32>().unwrap().iter().sum();
        assert!(
            sum < 1e-6,
            "histogram with all-OOB mask must be zero, got sum={sum}"
        );
    }

    #[test]
    fn oob_mask_partial_filters_correctly() {
        // With a partial OOB mask (only first sample in-bounds), the histogram
        // should be dominated by the first sample's contribution.
        let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
        let dev = device();

        // Three samples: first is identity (128, 128), rest are extreme (0, 255)
        let fixed = Tensor::<B, 1>::from_floats([128.0, 0.0, 255.0], &dev);
        let moving = Tensor::<B, 1>::from_floats([128.0, 255.0, 0.0], &dev);
        // Only the first sample is in-bounds
        let partial_mask = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0], &dev);

        let h_masked = hist.compute_joint_histogram(&fixed, &moving, Some(&partial_mask));
        let h_unmasked = hist.compute_joint_histogram(&fixed, &moving, None);

        // The masked histogram should have strictly less total weight than unmasked.
        let sum_masked: f32 = h_masked.into_data().as_slice::<f32>().unwrap().iter().sum();
        let sum_unmasked: f32 = h_unmasked.into_data().as_slice::<f32>().unwrap().iter().sum();
        assert!(
            sum_masked < sum_unmasked,
            "masked sum ({sum_masked}) must be less than unmasked sum ({sum_unmasked})"
        );
    }

    #[test]
    fn oob_mask_all_in_bounds_equivalent_to_no_mask() {
        // A mask of all 1.0 must produce the same result as passing None.
        let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0);
        let dev = device();

        let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
        let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);
        let all_in = Tensor::<B, 1>::ones([5], &dev);

        let h_with_mask = hist
            .compute_joint_histogram(&fixed, &moving, Some(&all_in))
            .into_data();
        let h_no_mask = hist
            .compute_joint_histogram(&fixed, &moving, None)
            .into_data();

        let s1 = h_with_mask.as_slice::<f32>().unwrap();
        let s2 = h_no_mask.as_slice::<f32>().unwrap();
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "all-ones mask must match no-mask: {a} vs {b}"
            );
        }
    }
}

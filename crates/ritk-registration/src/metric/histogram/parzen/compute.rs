use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::ParzenJointHistogram;

/// Construct the bin-center row `[1, num_bins]` used for Parzen weight broadcasting.
///
/// Returns `arange(0..num_bins).float().reshape([1, num_bins])` — a [1, B] row
/// of floating-point bin indices. Constructed once and cached on the struct
/// (`ParzenJointHistogram::bins_exp`) to eliminate the `arange` + `int→float`
/// kernel dispatches on every weight computation call.
fn arange_bins<B: Backend>(num_bins: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1, Int>::arange(0..num_bins as i64, device)
        .float()
        .reshape([1, num_bins])
}

impl<B: Backend> ParzenJointHistogram<B> {
    /// Compute the transposed Parzen weight matrix `W_fixed^T [num_bins, N]` for fixed-image values.
    ///
    /// This is the constant (non-autodiff) matrix that only depends on the fixed
    /// image and can be computed once and cached across all registration iterations.
    /// Used by both the non-chunked and chunked paths of `compute_image_joint_histogram`.
    pub(super) fn compute_w_fixed_transposed(
        &self,
        fixed_values: &Tensor<B, 1>,
        n: usize,
    ) -> Tensor<B, 2> {
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

        // Pre-compute bin centers [1, num_bins] for broadcasting.
        // Uses the struct-cached bins_exp when available, falling back to
        // on-the-fly construction for the rare case where `new()` wasn't used.
        let bins_exp = self
            .bins_exp
            .as_ref()
            .cloned()
            .unwrap_or_else(|| arange_bins(self.num_bins, &device));
        let vals_exp = fixed_norm.reshape([n, 1]);
        let diff = vals_exp - bins_exp;
        // Element-wise square: `diff * diff` compiles to a single fmul per element,
        // whereas `powf_scalar(2.0)` dispatches to a general-purpose pow() kernel
        // that is 5–10× slower. Mathematically identical for finite values.
        let sq = diff.clone() * diff;
        (sq * (-0.5 / sigma_sq)).exp().transpose() // [num_bins, N]
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

        // Pre-computed bin centers [1, num_bins] — avoids 2 GPU kernel dispatches
        // (arange + int-to-float cast) per call on the hot path.
        let bins_exp = self
            .bins_exp
            .as_ref()
            .cloned()
            .unwrap_or_else(|| arange_bins(num_bins, &device));

        // W_moving [N, num_bins] = exp(-0.5 * ((val - bin) / sigma)^2)
        let vals_exp = moving_norm.reshape([n, 1]);
        let diff = vals_exp - bins_exp;
        // Element-wise square instead of powf_scalar(2.0) — single fmul vs pow() kernel.
        let sq = diff.clone() * diff;
        let w_moving = (sq * (-0.5 / sigma_sq)).exp();

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

        // Pre-computed bin centers [1, Bins] — avoids 2 GPU kernel dispatches per call.
        let bins_exp = self
            .bins_exp
            .as_ref()
            .cloned()
            .unwrap_or_else(|| arange_bins(num_bins, &device));

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
                                                    // Element-wise square: `diff * diff` compiles to fmul, not pow().
            let sq = diff.clone() * diff;
            let exponent = sq * (-0.5 / sigma_sq);
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
}

// Image-level joint histogram computation (compute_image_joint_histogram) is in
// compute_image.rs to keep this file under the 500-line structural limit.

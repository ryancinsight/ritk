use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::ParzenJointHistogram;

/// Construct the bin-center row `[1, num_bins]` used for Parzen weight broadcasting.
///
/// Returns `arange(0..num_bins).float().reshape([1, num_bins])` — a [1, B] row
/// of floating-point bin indices. Called once in `ParzenJointHistogram::new()` to
/// eagerly initialize `bins_exp`, eliminating the `arange` + `int→float` kernel
/// dispatches on every weight computation call.
pub(super) fn arange_bins<B: Backend>(num_bins: usize, device: &B::Device) -> Tensor<B, 2> {
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
    pub(in crate::metric::histogram) fn compute_w_fixed_transposed(
        &self,
        fixed_values: &Tensor<B, 1>,
        n: usize,
    ) -> Tensor<B, 2> {
        // DRY-320-01: delegate to ParzenConfig via fixed_sigma_cfg()
        let sigma_sq = self.fixed_sigma_cfg().sigma_sq;

        let fixed_num_bins_f = self.num_bins as f32 - 1.0;
        let fixed_scale = fixed_num_bins_f / (self.max_intensity - self.min_intensity);
        let fixed_offset = -self.min_intensity * fixed_scale;
        let fixed_norm =
            (fixed_values.clone() * fixed_scale + fixed_offset).clamp(0.0, fixed_num_bins_f);

        // Pre-computed bin centers [1, num_bins] — eagerly initialized in `new()`.
        let bins_exp = self.bins_exp.as_ref().cloned().unwrap();

        Self::compute_parzen_weights(fixed_norm, n, sigma_sq, &bins_exp).transpose()
    }

    /// Compute Parzen weight matrix `[N, num_bins]`.
    ///
    /// Fused computation that minimizes intermediate tensor allocations:
    /// 1. `vals.reshape([N, 1])` — no allocation (view)
    /// 2. `vals_exp - bins_exp` — one allocation [N, num_bins]
    /// 3. `diff * diff` — one allocation [N, num_bins] (reuse diff via clone)
    /// 4. `sq * coeff + bias` — fused multiply-add, one allocation [N, num_bins]
    /// 5. `.exp()` — one allocation [N, num_bins]
    ///
    /// Total: 4 allocations of [N, num_bins] (down from 5 with separate mul + add).
    fn compute_parzen_weights(
        vals_norm: Tensor<B, 1>,
        n: usize,
        sigma_sq: f32,
        bins_exp: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let vals_exp = vals_norm.reshape([n, 1]); // [N, 1] — view, no allocation
        let diff = vals_exp - bins_exp.clone(); // [N, num_bins]
        let sq = diff.clone() * diff; // [N, num_bins] — element-wise square

        // Fused: exp(sq * (-0.5 / σ²)) — single scalar multiply + exp
        // This combines what was previously two ops (scalar mul, then exp)
        // into a chain where the scalar multiply is cheap and exp is the
        // dominant cost. Using `diff.clone() * diff` instead of `powf_scalar(2.0)`
        // compiles to a single fmul per element, 5-10× faster than pow().
        (sq * (-0.5 / sigma_sq)).exp() // [N, num_bins]
    }

    /// Compute joint histogram from a precomputed W_fixed^T [num_bins, N] and live moving values [N].
    ///
    /// `w_fixed_transposed` is the transposed Parzen weight matrix for the fixed image,
    /// precomputed once and cached. `moving_values` carries the autodiff gradient path.
    /// Only the moving-image side needs recomputation each iteration.
    ///
    /// `oob_mask` is an optional `[N]` float tensor (`1.0` = in-bounds, `0.0` = out-of-bounds).
    /// When provided, OOB samples are zeroed out of W_moving before the histogram matmul.
    #[allow(dead_code)] // Used by #[cfg(not(feature = "direct-parzen"))] fallback path
    pub(super) fn compute_joint_histogram_from_cache(
        &self,
        w_fixed_transposed: &Tensor<B, 2>, // [num_bins, N]
        moving_values: &Tensor<B, 1>,      // [N]
        oob_mask: Option<&Tensor<B, 1>>,   // [N] in-bounds mask (1.0=in, 0.0=out)
    ) -> Tensor<B, 2> {
        let [n] = moving_values.dims();
        let num_bins = self.num_bins;

        // DRY-320-01: delegate to ParzenConfig via moving_sigma_cfg()
        let sigma_sq = self.moving_sigma_cfg().sigma_sq;

        // Normalize moving values to [0, num_bins-1] using the moving-image range.
        let mov_num_bins_f = num_bins as f32 - 1.0;
        let mov_min = self.moving_min_intensity.unwrap_or(self.min_intensity);
        let mov_max = self.moving_max_intensity.unwrap_or(self.max_intensity);
        let mov_scale = mov_num_bins_f / (mov_max - mov_min);
        let mov_offset = -mov_min * mov_scale;
        let moving_norm =
            (moving_values.clone() * mov_scale + mov_offset).clamp(0.0, mov_num_bins_f);

        // Pre-computed bin centers [1, num_bins] — eagerly initialized in `new()`.
        let bins_exp = self.bins_exp.as_ref().cloned().unwrap();

        let w_moving = Self::compute_parzen_weights(moving_norm, n, sigma_sq, &bins_exp);

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
    #[allow(clippy::single_range_in_vec_init)] // Burn's .slice() takes an array of ranges, one per dimension; [start..end] is correct for 1-D
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

        let fix_min = self.min_intensity;
        let fix_max = self.max_intensity;
        // Moving-image range — fall back to fixed-image range (backward-compatible).
        let mov_min = self.moving_min_intensity.unwrap_or(fix_min);
        let mov_max = self.moving_max_intensity.unwrap_or(fix_max);

        // Normalize intensities to [0, num_bins-1] using the supplied range.
        // Fused: (t - min) / (max - min) * num_bins_f → t * scale + offset
        // reduces 3 intermediate tensor allocations to 1.
        let normalize = |t: Tensor<B, 1>, min: f32, max: f32| -> Tensor<B, 1> {
            let scale = num_bins_f / (max - min);
            let offset = -min * scale;
            (t * scale + offset).clamp(0.0, num_bins_f)
        };

        // Pre-computed bin centers [1, Bins] — eagerly initialized in `new()`.
        let bins_exp = self.bins_exp.as_ref().cloned().unwrap();

        // DRY-320-01: delegate to ParzenConfig via helper methods
        let sigma_sq_fix = self.fixed_sigma_cfg().sigma_sq;
        let sigma_sq_mov = self.moving_sigma_cfg().sigma_sq;

        // WGPU dispatch limit workaround
        // The matmul (Bins, N) * (N, Bins) reduces along N.
        // If N is too large, it exceeds dispatch limits.
        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
            let fixed_norm = normalize(fixed.clone(), fix_min, fix_max);
            let moving_norm = normalize(moving.clone(), mov_min, mov_max);

            let w_fixed = Self::compute_parzen_weights(fixed_norm, n, sigma_sq_fix, &bins_exp);
            let w_moving = Self::compute_parzen_weights(moving_norm, n, sigma_sq_mov, &bins_exp);

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

                let w_fixed = Self::compute_parzen_weights(
                    fixed_norm,
                    current_chunk_size,
                    sigma_sq_fix,
                    &bins_exp,
                );
                let w_moving = Self::compute_parzen_weights(
                    moving_norm,
                    current_chunk_size,
                    sigma_sq_mov,
                    &bins_exp,
                );

                // Apply per-chunk OOB mask if provided.
                let w_moving = match oob_mask {
                    Some(m) => {
                        w_moving
                            * m.clone()
                                .slice([start..end])
                                .reshape([current_chunk_size, 1])
                    }
                    _ => w_moving,
                };

                joint_hist = joint_hist + w_fixed.transpose().matmul(w_moving);
            }
            joint_hist
        }
    }
}

// Image-level joint histogram computation (compute_image_joint_histogram) is in
// compute_image.rs to keep this file under the 500-line structural limit.

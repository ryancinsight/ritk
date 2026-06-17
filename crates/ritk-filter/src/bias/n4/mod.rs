//! N4 bias field correction filter.
//!
//! # Reference
//! Tustison, N.J., et al. (2010). N4ITK: Improved N3 Bias Correction.
//! *IEEE Trans. Med. Imaging*, 29(6):1310–1320. doi:10.1109/TMI.2010.2046908
//!
//! # Algorithm
//! Models multiplicative bias: I(x) = S(x)·B(x). In log-space this becomes
//! v(x) = ln(S(x)) + ln(B(x)). The accumulated log-bias field b is estimated
//! via a multi-resolution B-spline fitting loop.
//!
//! Per-level iteration:
//! 1. w = v − b (current debiased log-intensity)
//! 2. w̃ = histogram_sharpen(w) (Wiener deconvolution sharpens tissue peaks)
//! 3. r = w − w̃ (residual ≈ remaining low-frequency bias)
//! 4. Δb = bspline_smooth(r) (smooth B-spline fit to residual)
//! 5. b ← b + Δb (additive accumulation)
//! 6. Converge when ‖Δb‖_RMS < threshold.
//!
//! Corrected image: exp(v − b).
//!
//! # Histogram Sharpening
//! Faithful to ITK `N4BiasFieldCorrectionImageFilter::SharpenImage`. Models
//! H_observed = H_true ∗ G_noise, recovers the deconvolved density U via Wiener
//! deconvolution Û\[k\] = Ĥ\[k\]·Ĝ*\[k\] / (|Ĝ\[k\]|² + wiener_noise), then maps
//! each intensity through the conditional expectation
//! E\[i\] = (U·c ⋆ G)\[i\] / (U ⋆ G)\[i\] (c = bin centre). This pulls intensities
//! toward the sharpened tissue peaks — the actual N4 sharpening, in place of the
//! earlier rank-preserving CDF/quantile transfer that left N4 behaving like N3.

mod dft;
mod histogram_sharpen;

use super::bspline_bias::{bspline_evaluate, bspline_fit};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use histogram_sharpen::{histogram_sharpen, HistogramSharpenScratch};
use ritk_core::image::Image;
use ritk_spatial::VolumeDims;
use ritk_tensor_ops::extract_vec;

// Re-export the DFT helpers so the `tests_n4` module can reach them
// via `use super::*;` (the tests were failing with E0425 "cannot find
// function" because the dft submodule is private and the functions
// were not re-exported).
#[cfg(test)]
pub(crate) use dft::{dft_real_into, idft_real_into, next_pow2};

// ── Public types ───────────────────────────────────────────────────────────────

/// Configuration for the N4 bias field correction filter.
#[derive(Debug, Clone)]
pub struct N4Config {
    /// Number of multi-resolution B-spline fitting levels.
    pub num_fitting_levels: usize,
    /// Maximum iterations per fitting level.
    pub num_iterations: usize,
    /// Convergence threshold: ‖Δb‖_RMS < threshold triggers early exit.
    pub convergence_threshold: f64,
    /// Number of histogram bins for Wiener-based sharpening.
    pub num_histogram_bins: usize,
    /// Histogram-sharpening full-width-at-half-maximum in log-intensity units
    /// (ITK/ANTs `m_BiasFieldFullWidthAtHalfMaximum`, default 0.15). Converted to
    /// bins via `fwhm / bin_width` (ITK's `scaledFWHM`) to set the Gaussian width
    /// in the Wiener deconvolution.
    pub bias_field_fwhm: f64,
    /// Initial control-point count per dimension at level 0.
    /// Doubles each level: cg\[d\] = initial\[d\] * 2^level, clamped to \[4, n/2+2\].
    pub initial_control_points: VolumeDims,
    /// Wiener filter noise term in the histogram-sharpening deconvolution
    /// (ITK/ANTs default 0.01): Û = Ĥ·conj(Ĝ) / (|Ĝ|² + wiener_noise).
    pub noise_estimate: f64,
    /// Maximum voxels used in the B-spline fitting step (uniform subsampling).
    pub max_fitting_points: usize,
}

impl Default for N4Config {
    fn default() -> Self {
        Self {
            num_fitting_levels: 4,
            num_iterations: 50,
            convergence_threshold: 0.001,
            num_histogram_bins: 200,
            bias_field_fwhm: 0.15,
            initial_control_points: VolumeDims::new([4, 4, 4]),
            noise_estimate: 0.01,
            max_fitting_points: 10_000,
        }
    }
}

/// N4 bias field correction filter.
///
/// Corrects spatially varying multiplicative intensity non-uniformity
/// (bias field) in 3-D volumetric images using the N4ITK algorithm.
pub struct N4BiasFieldCorrectionFilter {
    pub config: N4Config,
}

impl N4BiasFieldCorrectionFilter {
    /// Construct with the provided configuration.
    pub fn new(config: N4Config) -> Self {
        Self { config }
    }

    /// Apply N4 bias field correction to a 3-D image with f32 element type.
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be read as `f32`, or if the
    /// B-spline normal equations are degenerate (protected by Tikhonov λ=1e-6).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        // ── 1. Extract CPU data ────────────────────────────────────────────
        let (vals, shape) = extract_vec(image)?;
        let [nz, ny, nx] = shape;
        let n = nz * ny * nx;
        let dims = [nz, ny, nx];

        // ── 2. Log-intensity: v = ln(max(I, ε)) ───────────────────────────
        const EPS: f32 = 1e-4;
        let v: Vec<f32> = vals.iter().map(|&x| x.max(EPS).ln()).collect();

        // ── 3. Multi-resolution bias estimation ────────────────────────────
        // b accumulates the log-bias field across all levels and iterations.
        let mut b = vec![0.0f32; n];
        // Pre-allocated scratch buffers: reused across all iterations to eliminate
        // O(iterations × 2) full-volume heap allocations per level.
        let mut w = vec![0.0f32; n];
        let mut r = vec![0.0f32; n];
        // Pre-allocated histogram-sharpen scratch: eliminates ~8 allocations
        // per iteration × num_iterations × num_fitting_levels.
        let mut hs_scratch = HistogramSharpenScratch::new(self.config.num_histogram_bins, n);

        for level in 0..self.config.num_fitting_levels {
            // Control grid doubles each level; clamped to [4, n_d/2 + 2].
            let shift = 1usize << level;
            let cg: [usize; 3] = std::array::from_fn(|d| {
                (self.config.initial_control_points.0[d] * shift)
                    .min(dims[d] / 2 + 2)
                    .max(4)
            });

            for _ in 0..self.config.num_iterations {
                // w = v − b (in-place into pre-allocated buffer)
                for i in 0..n {
                    w[i] = v[i] - b[i];
                }

                // Wiener-sharpened estimate of the "true" log-intensity
                // distribution (removes Rician noise blurring of tissue peaks).
                histogram_sharpen(
                    &w,
                    self.config.num_histogram_bins,
                    self.config.bias_field_fwhm,
                    self.config.noise_estimate,
                    &mut hs_scratch,
                )?;

                // r = w − w_sharp (in-place into pre-allocated buffer)
                for i in 0..n {
                    r[i] = w[i] - hs_scratch.w_sharp[i];
                }

                // Fit a smooth B-spline to the residual → bias correction Δb.
                let ctrl = bspline_fit(&r, dims, cg, self.config.max_fitting_points)?;
                let delta = bspline_evaluate(&ctrl, cg, dims);

                // Convergence criterion: ‖Δb‖_RMS < threshold.
                let change: f64 = {
                    let ss: f64 = delta.iter().map(|&x| (x as f64).powi(2)).sum();
                    (ss / n as f64).sqrt()
                };

                // Additive accumulation of the log-bias field.
                for i in 0..n {
                    b[i] += delta[i];
                }

                if change < self.config.convergence_threshold {
                    break;
                }
            }
        }

        // ── 4. Corrected image: exp(v − b) ─────────────────────────────────
        // Reuse w buffer for final output instead of allocating result_vals.
        for i in 0..n {
            w[i] = (v[i] - b[i]).exp();
        }

        // ── 5. Reconstruct tensor ───────────────────────────────────────────
        let td2 = TensorData::new(w, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td2, &image.data().device());

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

#[cfg(test)]
#[path = "tests_n4.rs"]
mod tests_n4;

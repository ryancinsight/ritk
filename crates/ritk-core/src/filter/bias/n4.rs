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
//!   1. w = v − b                      (current debiased log-intensity)
//!   2. w̃ = histogram_sharpen(w)       (Wiener deconvolution sharpens tissue peaks)
//!   3. r = w − w̃                      (residual ≈ remaining low-frequency bias)
//!   4. Δb = bspline_smooth(r)         (smooth B-spline fit to residual)
//!   5. b ← b + Δb                     (additive accumulation)
//!   6. Converge when ‖Δb‖_RMS < threshold.
//!
//! Corrected image: exp(v − b).
//!
//! # Histogram Sharpening
//! Models H_observed = H_true ∗ G_noise then recovers H_true via
//! Wiener deconvolution:
//!   Ĥ_sharp[k] = Ĥ[k] · Ĝ*[k] / (|Ĝ[k]|² + σ_noise²)
//! followed by CDF-based quantile transfer from H_observed to H_sharp.

use super::bspline_bias::{bspline_evaluate, bspline_fit};
use crate::filter::ops::extract_vec;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

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
    /// Initial control-point count per dimension at level 0.
    /// Doubles each level: cg[d] = initial[d] * 2^level, clamped to [4, n/2+2].
    pub initial_control_points: [usize; 3],
    /// Noise fraction: σ_bins = max(0.5, noise_estimate · n_bins).
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
            initial_control_points: [4, 4, 4],
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

        for level in 0..self.config.num_fitting_levels {
            // Control grid doubles each level; clamped to [4, n_d/2 + 2].
            let shift = 1usize << level;
            let cg: [usize; 3] = std::array::from_fn(|d| {
                (self.config.initial_control_points[d] * shift)
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
                // w_sharp allocates internally (cannot be eliminated).
                let w_sharp = histogram_sharpen(
                    &w,
                    self.config.num_histogram_bins,
                    self.config.noise_estimate,
                )?;

                // r = w − w_sharp (in-place into pre-allocated buffer)
                for i in 0..n {
                    r[i] = w[i] - w_sharp[i];
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

// ── Private helpers ────────────────────────────────────────────────────────────

/// Histogram sharpening via Wiener deconvolution, followed by CDF-based
/// quantile transfer from H_observed to H_sharp.
///
/// # Algorithm
/// 1. Build histogram H of `w` with `n_bins` bins.
/// 2. σ_bins = max(0.5, noise_fraction · n_bins).
/// 3. Gaussian kernel G(σ_bins), zero-padded DFT of length N = next_pow2(n_bins).
/// 4. Wiener deconvolution:
///    Ĥ_sharp[k] = Ĥ[k]·Ĝ*[k] / (|Ĝ[k]|² + σ_noise)
///    where σ_noise = 0.01 · max_k |Ĥ[k]|².
/// 5. IDFT → H_sharp; clamp negatives to 0.
/// 6. CDF transfer: each voxel intensity maps to the quantile bin in H_sharp
///    whose CDF matches the voxel's CDF rank in H.
fn histogram_sharpen(w: &[f32], n_bins: usize, noise_fraction: f64) -> anyhow::Result<Vec<f32>> {
    if w.is_empty() || n_bins < 2 {
        return Ok(w.to_vec());
    }

    let w_min = w.iter().cloned().fold(f32::INFINITY, f32::min);
    let w_max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (w_max - w_min) as f64;

    // Degenerate case: constant or near-constant input — no sharpening needed.
    if range < 1e-6 {
        return Ok(w.to_vec());
    }

    let bin_width = range / n_bins as f64;

    // ── Build histogram (normalised to probability density) ───────────────
    // Normalisation is required so that Ĥ[0] = 1 and the Wiener noise_power
    // is calibrated on a [0, 1] scale.  Raw-count histograms have
    // Ĥ[0] = N_voxels, making noise_power = 0.01·N² which dominates every
    // non-DC frequency and collapses H_sharp to a constant.
    let mut h_raw = vec![0.0f64; n_bins];
    for &wi in w {
        let b = (((wi - w_min) as f64 / bin_width).floor() as isize).clamp(0, n_bins as isize - 1)
            as usize;
        h_raw[b] += 1.0;
    }
    let total_raw: f64 = h_raw.iter().sum::<f64>().max(1.0);
    let h: Vec<f64> = h_raw.iter().map(|&v| v / total_raw).collect();

    // ── Gaussian kernel (σ in histogram-bin units) ─────────────────────────
    let sigma_bins = (noise_fraction * n_bins as f64).max(0.5);
    let g = gaussian_kernel_1d(sigma_bins);

    // ── Wiener deconvolution in DFT domain ─────────────────────────────────
    let n_dft = next_pow2(n_bins.max(2));
    let h_hat = dft_real(&h, n_dft);
    let g_hat = dft_real(&g, n_dft);

    // noise_power = 0.01 · max_k |Ĥ[k]|²
    // With the normalised histogram Ĥ[0] = 1, so noise_power ≤ 0.01 and
    // the Wiener filter can now properly deconvolve non-DC frequency components
    // where |Ĝ[k]|² is comparable to or greater than noise_power.
    let noise_power = 0.01
        * h_hat
            .iter()
            .map(|(re, im)| re * re + im * im)
            .fold(0.0f64, f64::max);

    // Ĥ_sharp[k] = Ĥ[k]·Ĝ*[k] / (|Ĝ[k]|² + σ_noise)
    // where Ĥ·Ĝ* = (hr + i·hi)·(gr − i·gi) = (hr·gr + hi·gi) + i·(hi·gr − hr·gi)
    let h_sharp_hat: Vec<(f64, f64)> = h_hat
        .iter()
        .zip(g_hat.iter())
        .map(|(&(hr, hi), &(gr, gi))| {
            let num_re = hr * gr + hi * gi;
            let num_im = hi * gr - hr * gi;
            let denom = gr * gr + gi * gi + noise_power;
            if denom < f64::EPSILON {
                (0.0, 0.0)
            } else {
                (num_re / denom, num_im / denom)
            }
        })
        .collect();

    // IDFT; clamp negatives — probability density must be ≥ 0.
    let h_sharp_raw = idft_real(&h_sharp_hat, n_bins);
    let h_sharp: Vec<f64> = h_sharp_raw.iter().map(|&v| v.max(0.0)).collect();

    // ── CDF construction ───────────────────────────────────────────────────
    // h is already normalised (sums to ~1); use raw counts for the guard.
    let total_h: f64 = h_raw.iter().sum();
    let total_s: f64 = h_sharp.iter().sum();

    // Guard 1: if sharpening collapsed the density to zero, pass through.
    if total_h < 1.0 || total_s < 1e-12 {
        return Ok(w.to_vec());
    }

    // Guard 2: concentration check.
    // For a sharpened histogram, peak bins must be more prominent than in the
    // original.  If the maximum normalised bin value in H_sharp is lower than
    // in H, the Wiener deconvolution broadened the distribution (this happens
    // for discrete-spike inputs where H_true ≈ H_observed and there is no
    // G_noise component to invert).  Applying the CDF transfer in this case
    // widens voxel intensities, increasing CoV.  Detect and bail out.
    let max_h: f64 = h.iter().cloned().fold(0.0_f64, f64::max);
    let max_s: f64 = {
        let s_sum = total_s.max(1e-12);
        h_sharp.iter().map(|&v| v / s_sum).fold(0.0_f64, f64::max)
    };
    if max_s <= max_h * 1.01 {
        // Sharpening did not increase peak concentration; return input unchanged.
        return Ok(w.to_vec());
    }

    let cdf_h: Vec<f64> = {
        let mut cdf = vec![0.0f64; n_bins];
        let mut acc = 0.0f64;
        for (i, &hi) in h.iter().enumerate() {
            acc += hi;
            cdf[i] = acc / total_h;
        }
        cdf
    };

    let cdf_s: Vec<f64> = {
        let mut cdf = vec![0.0f64; n_bins];
        let mut acc = 0.0f64;
        for (i, &si) in h_sharp.iter().enumerate() {
            acc += si;
            cdf[i] = acc / total_s;
        }
        cdf
    };

    // ── Quantile transfer per voxel ────────────────────────────────────────
    // For each voxel: find its CDF rank in H, then find the matching bin in H_sharp.
    let w_min_f64 = w_min as f64;
    let w_sharp: Vec<f32> = w
        .iter()
        .map(|&wi| {
            let bin_i = (((wi - w_min) as f64 / bin_width).floor() as isize)
                .clamp(0, n_bins as isize - 1) as usize;
            let q = cdf_h[bin_i];
            // First index in the monotone cdf_s where cdf_s[t] ≥ q.
            let target = cdf_s.partition_point(|&v| v < q).min(n_bins - 1);
            (w_min_f64 + (target as f64 + 0.5) * bin_width) as f32
        })
        .collect();

    Ok(w_sharp)
}

/// Real DFT of `data` zero-padded to length `n` (O(n²), acceptable for n ≤ 512).
///
/// Returns `n` complex coefficients X[k] = Σ_{j=0}^{n-1} x[j]·e^{−2πi·jk/n}.
fn dft_real(data: &[f64], n: usize) -> Vec<(f64, f64)> {
    let len = data.len().min(n);
    let mut out = vec![(0.0f64, 0.0f64); n];
    let two_pi_n = -2.0 * std::f64::consts::PI / n as f64;
    for (k, out_k) in out.iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &dj) in data[..len].iter().enumerate() {
            let angle = two_pi_n * k as f64 * j as f64;
            re += dj * angle.cos();
            im += dj * angle.sin();
        }
        *out_k = (re, im);
    }
    out
}

/// Inverse real DFT of `freq` (length N), returning the first `n` real-valued
/// samples: x[j] = (1/N) Σ_{k=0}^{N-1} X[k]·e^{2πi·jk/N}  (real part only).
///
/// O(N²) — acceptable for N ≤ 512.
fn idft_real(freq: &[(f64, f64)], n: usize) -> Vec<f64> {
    let big_n = freq.len();
    if big_n == 0 {
        return vec![0.0; n];
    }
    let two_pi_n = 2.0 * std::f64::consts::PI / big_n as f64;
    (0..n)
        .map(|j| {
            let val: f64 = freq
                .iter()
                .enumerate()
                .map(|(k, &(re, im))| {
                    let angle = two_pi_n * k as f64 * j as f64;
                    re * angle.cos() - im * angle.sin()
                })
                .sum();
            val / big_n as f64
        })
        .collect()
}

/// 1-D Gaussian kernel of σ `sigma` (in bins), radius = ⌈3σ⌉, L1-normalised.
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let len = 2 * radius + 1;
    let two_s2 = 2.0 * sigma * sigma;
    let mut kernel: Vec<f64> = (0..len)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / two_s2).exp()
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|v| *v /= sum);
    kernel
}

/// Smallest power of two ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
#[path = "tests_n4.rs"]
mod tests_n4;

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

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

use super::bspline_bias::{bspline_evaluate, bspline_fit};

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
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("f32 required: {:?}", e))?
            .to_vec();
        let shape = image.shape();
        let [nz, ny, nx] = shape;
        let n = nz * ny * nx;
        let dims = [nz, ny, nx];

        // ── 2. Log-intensity: v = ln(max(I, ε)) ───────────────────────────
        const EPS: f32 = 1e-4;
        let v: Vec<f32> = vals.iter().map(|&x| x.max(EPS).ln()).collect();

        // ── 3. Multi-resolution bias estimation ────────────────────────────
        // b accumulates the log-bias field across all levels and iterations.
        let mut b = vec![0.0f32; n];

        for level in 0..self.config.num_fitting_levels {
            // Control grid doubles each level; clamped to [4, n_d/2 + 2].
            let shift = 1usize << level;
            let cg: [usize; 3] = std::array::from_fn(|d| {
                (self.config.initial_control_points[d] * shift)
                    .min(dims[d] / 2 + 2)
                    .max(4)
            });

            for _ in 0..self.config.num_iterations {
                // Current debiased log-intensity estimate.
                let w: Vec<f32> = (0..n).map(|i| v[i] - b[i]).collect();

                // Wiener-sharpened estimate of the "true" log-intensity
                // distribution (removes Rician noise blurring of tissue peaks).
                let w_sharp = histogram_sharpen(
                    &w,
                    self.config.num_histogram_bins,
                    self.config.noise_estimate,
                )?;

                // Residual captures the remaining low-frequency bias component
                // not yet accounted for in b.
                let r: Vec<f32> = (0..n).map(|i| w[i] - w_sharp[i]).collect();

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
        let result_vals: Vec<f32> = (0..n).map(|i| (v[i] - b[i]).exp()).collect();

        // ── 5. Reconstruct tensor ───────────────────────────────────────────
        let td2 = TensorData::new(result_vals, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td2, &image.data().device());

        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
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
///      Ĥ_sharp[k] = Ĥ[k]·Ĝ*[k] / (|Ĝ[k]|² + σ_noise)
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
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &dj) in data[..len].iter().enumerate() {
            let angle = two_pi_n * k as f64 * j as f64;
            re += dj * angle.cos();
            im += dj * angle.sin();
        }
        out[k] = (re, im);
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

/// Root mean square difference between two equal-length f32 slices.
#[cfg(test)]
fn rms_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let ss: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ((ai - bi) as f64).powi(2))
        .sum();
    (ss / a.len() as f64).sqrt()
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

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let [nz, ny, nx] = dims;
        let td = TensorData::new(vals, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::<3>::new([0.0, 0.0, 0.0]),
            Spacing::<3>::new([1.0, 1.0, 1.0]),
            Direction::<3>::identity(),
        )
    }

    fn extract_vals(img: Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Coefficient of variation (σ/μ) for a subset of voxels identified by indices.
    fn within_class_cov(vals: &[f32], indices: &[usize]) -> f64 {
        assert!(!indices.is_empty(), "within_class_cov: empty class");
        let n = indices.len() as f64;
        let mean: f64 = indices.iter().map(|&i| vals[i] as f64).sum::<f64>() / n;
        let var: f64 = indices
            .iter()
            .map(|&i| ((vals[i] as f64) - mean).powi(2))
            .sum::<f64>()
            / n;
        var.sqrt() / mean.abs().max(1e-10)
    }

    /// N4 stability on a two-class image: the algorithm must not diverge or
    /// increase within-class CoV by more than a small tolerance.
    ///
    /// # Limitation note
    /// N4 (and the underlying histogram-sharpening step) is designed for images
    /// with **continuous** intensity histograms, such as real MRI data where
    /// thermal noise spreads each tissue peak into a smooth Gaussian.  Synthetic
    /// images with only a handful of discrete intensity levels produce delta-like
    /// histogram spikes for which the Wiener deconvolution step cannot
    /// distinguish the bias-induced spread from image noise, so the residual fed
    /// to the B-spline fitter is effectively zero.  This test therefore only
    /// verifies **stability** (CoV does not increase) for such a degenerate
    /// input; the `histogram_sharpen_continuous_bimodal_reduces_spread` test
    /// below verifies the sharpening step on a continuous bimodal distribution.
    #[test]
    fn two_class_n4_stability_discrete_histogram() {
        let nz = 16usize;
        let ny = 16usize;
        let nx = 16usize;
        let n = nz * ny * nx;

        let mut class_a: Vec<usize> = Vec::new();
        let mut class_b: Vec<usize> = Vec::new();

        let vals: Vec<f32> = (0..n)
            .map(|vi| {
                let ix = vi % nx;
                let iy = (vi / nx) % ny;
                let bias = 1.0_f32 + 0.25 * (ix as f32 / (nx - 1) as f32 - 0.5);
                let true_intensity = if iy < ny / 2 { 100.0_f32 } else { 40.0_f32 };
                if iy < ny / 2 {
                    class_a.push(vi);
                } else {
                    class_b.push(vi);
                }
                true_intensity * bias
            })
            .collect();

        let cov_a_before = within_class_cov(&vals, &class_a);
        let cov_b_before = within_class_cov(&vals, &class_b);

        let image = make_image(vals, [nz, ny, nx]);
        // 1 level × 5 iterations keeps ctrl_grid at [4,4,4] = 64 CP,
        // so LU decomposition is O(64³) ≈ 260k ops — fast even in debug mode.
        let config = N4Config {
            num_fitting_levels: 1,
            num_iterations: 5,
            convergence_threshold: 1e-4,
            num_histogram_bins: 200,
            initial_control_points: [4, 4, 4],
            noise_estimate: 0.07,
            max_fitting_points: 256,
        };

        let out = extract_vals(
            N4BiasFieldCorrectionFilter::new(config)
                .apply(&image)
                .expect("N4 two-class stability apply failed"),
        );

        let cov_a_after = within_class_cov(&out, &class_a);
        let cov_b_after = within_class_cov(&out, &class_b);

        // Stability: CoV must not increase by more than 1 % relative.
        // (For this degenerate discrete input the algorithm does minimal work.)
        assert!(
            cov_a_after <= cov_a_before * 1.01,
            "Class A CoV increased: before={cov_a_before:.4} after={cov_a_after:.4}"
        );
        assert!(
            cov_b_after <= cov_b_before * 1.01,
            "Class B CoV increased: before={cov_b_before:.4} after={cov_b_after:.4}"
        );
        // Output must remain finite and positive.
        for &v in &out {
            assert!(v > 0.0 && v.is_finite(), "non-positive/nan output: {v}");
        }
    }

    /// histogram_sharpen reduces within-mode spread for a continuous bimodal
    /// log-intensity distribution.
    ///
    /// # Setup
    /// Two uniformly-distributed modes, each spanning 200 linearly-spaced
    /// values over a range of width W = 0.10 log-intensity units:
    ///   Mode A: 200 values in [4.45, 4.55]   (centred on ln(85.7) ≈ 4.45…4.55)
    ///   Mode B: 200 values in [3.38, 3.48]   (centred on ln(32)  ≈ 3.43)
    ///
    /// Total histogram range ≈ 4.55 − 3.38 = 1.17.
    /// Mode width in bins = W / bin_width = 0.10 / (1.17/200) ≈ 17.1 bins.
    ///
    /// noise_estimate chosen so that sigma_bins ≈ mode_width_bins:
    ///   sigma_bins = noise_estimate × n_bins = 0.087 × 200 ≈ 17.4 bins  ✓
    ///
    /// # Invariant
    /// After sharpening, the within-mode variance must decrease (the mode must
    /// be narrower) relative to the original uniformly-spread mode.
    #[test]
    fn histogram_sharpen_continuous_bimodal_reduces_spread() {
        // Mode A: 200 uniformly-spaced values in [4.45, 4.55].
        // Mode B: 200 uniformly-spaced values in [3.38, 3.48].
        let n_per_mode = 200usize;
        let mode_a_lo = 4.45_f32;
        let mode_a_hi = 4.55_f32;
        let mode_b_lo = 3.38_f32;
        let mode_b_hi = 3.48_f32;

        let mut w = Vec::with_capacity(2 * n_per_mode);
        for i in 0..n_per_mode {
            let t = i as f32 / (n_per_mode - 1) as f32;
            w.push(mode_a_lo + t * (mode_a_hi - mode_a_lo));
        }
        for i in 0..n_per_mode {
            let t = i as f32 / (n_per_mode - 1) as f32;
            w.push(mode_b_lo + t * (mode_b_hi - mode_b_lo));
        }

        // noise_estimate ≈ mode_width / total_range = 0.10 / 1.17 ≈ 0.085
        let w_sharp =
            histogram_sharpen(&w, 200, 0.087).expect("histogram_sharpen failed on bimodal input");

        assert_eq!(w_sharp.len(), w.len(), "output length must match input");

        // Within-mode-A variance: compare spread around the mode centre 4.50.
        let centre_a = 0.5 * (mode_a_lo + mode_a_hi);
        let var_a_before: f64 = w
            .iter()
            .take(n_per_mode)
            .map(|&v| ((v - centre_a) as f64).powi(2))
            .sum::<f64>()
            / n_per_mode as f64;

        let mean_a_after: f64 = w_sharp
            .iter()
            .take(n_per_mode)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n_per_mode as f64;
        let var_a_after: f64 = w_sharp
            .iter()
            .take(n_per_mode)
            .map(|&v| ((v as f64) - mean_a_after).powi(2))
            .sum::<f64>()
            / n_per_mode as f64;

        assert!(
            var_a_after < var_a_before,
            "histogram_sharpen did not reduce Mode-A variance: \
             before={var_a_before:.6} after={var_a_after:.6}"
        );

        // Within-mode-B variance.
        let centre_b = 0.5 * (mode_b_lo + mode_b_hi);
        let var_b_before: f64 = w
            .iter()
            .skip(n_per_mode)
            .map(|&v| ((v - centre_b) as f64).powi(2))
            .sum::<f64>()
            / n_per_mode as f64;

        let mean_b_after: f64 = w_sharp
            .iter()
            .skip(n_per_mode)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n_per_mode as f64;
        let var_b_after: f64 = w_sharp
            .iter()
            .skip(n_per_mode)
            .map(|&v| ((v as f64) - mean_b_after).powi(2))
            .sum::<f64>()
            / n_per_mode as f64;

        assert!(
            var_b_after < var_b_before,
            "histogram_sharpen did not reduce Mode-B variance: \
             before={var_b_before:.6} after={var_b_after:.6}"
        );
    }

    /// Constant image: no crash; all output values within 100.0 ± 5.0.
    ///
    /// A constant log-intensity produces a degenerate histogram (range < 1e-6),
    /// which triggers the early-exit path in histogram_sharpen.  The residual
    /// r = w − w_sharp = 0, so Δb ≈ 0 and the corrected image ≈ input.
    #[test]
    fn constant_image_stable() {
        let dims = [8usize, 8, 8];
        let n = 8 * 8 * 8;
        let image = make_image(vec![100.0f32; n], dims);

        let config = N4Config {
            num_fitting_levels: 1,
            num_iterations: 5,
            convergence_threshold: 0.001,
            num_histogram_bins: 50,
            initial_control_points: [4, 4, 4],
            noise_estimate: 0.01,
            max_fitting_points: 512,
        };

        let out = extract_vals(
            N4BiasFieldCorrectionFilter::new(config)
                .apply(&image)
                .expect("N4 constant failed"),
        );

        for &v in &out {
            assert!(
                (v - 100.0).abs() < 5.0,
                "constant image: expected ~100.0, got {v:.4}"
            );
        }
    }

    /// All corrected output values are strictly positive.
    ///
    /// Since result[i] = exp(v[i] − b[i]) and exp is always positive (> 0),
    /// this invariant must hold regardless of the bias estimate.
    #[test]
    fn output_all_positive() {
        let nz = 8;
        let ny = 8;
        let nx = 8;
        let n = nz * ny * nx;

        // Mild quadratic bias along z.
        let vals: Vec<f32> = (0..n)
            .map(|vi| {
                let iz = vi / (ny * nx);
                let t = iz as f32 / (nz - 1) as f32;
                50.0_f32 * (1.0 + 0.3 * t * t)
            })
            .collect();

        let image = make_image(vals, [nz, ny, nx]);
        let config = N4Config {
            num_fitting_levels: 1,
            num_iterations: 5,
            convergence_threshold: 0.001,
            num_histogram_bins: 50,
            initial_control_points: [4, 4, 4],
            noise_estimate: 0.01,
            max_fitting_points: 512,
        };

        let out = extract_vals(
            N4BiasFieldCorrectionFilter::new(config)
                .apply(&image)
                .expect("N4 positive failed"),
        );

        for &v in &out {
            assert!(v > 0.0, "non-positive output value: {v}");
        }
    }

    // ── Unit tests for private helpers ──────────────────────────────────────

    /// next_pow2 correctness at boundary values.
    #[test]
    fn next_pow2_boundaries() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(128), 128);
        assert_eq!(next_pow2(129), 256);
        assert_eq!(next_pow2(200), 256);
    }

    /// Gaussian kernel is L1-normalised and symmetric.
    #[test]
    fn gaussian_kernel_normalised_and_symmetric() {
        for &sigma in &[0.5_f64, 1.0, 2.5, 5.0] {
            let k = gaussian_kernel_1d(sigma);
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "sigma={sigma}: kernel sum = {sum:.15}"
            );
            // Symmetric: k[i] == k[len-1-i]
            let len = k.len();
            for i in 0..len / 2 {
                assert!(
                    (k[i] - k[len - 1 - i]).abs() < 1e-15,
                    "sigma={sigma}: asymmetry at i={i}"
                );
            }
        }
    }

    /// rms_diff is zero for identical slices and positive for different ones.
    #[test]
    fn rms_diff_identity_and_positive() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(rms_diff(&a, &a), 0.0);

        let b = vec![2.0f32, 3.0, 4.0, 5.0]; // each element +1
        let d = rms_diff(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "expected rms=1.0, got {d}");
    }

    /// DFT round-trip: IDFT(DFT(x)) ≈ x for a short real sequence.
    #[test]
    fn dft_round_trip() {
        let data = vec![1.0f64, 2.0, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0];
        let n = data.len();
        let freq = dft_real(&data, n);
        let recovered = idft_real(&freq, n);
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-9,
                "index {i}: orig={orig:.10} rec={rec:.10}"
            );
        }
    }

    /// histogram_sharpen returns the input unchanged for a constant signal.
    #[test]
    fn histogram_sharpen_passthrough_for_constant_input() {
        let w = vec![3.14f32; 64];
        let out = histogram_sharpen(&w, 100, 0.01).unwrap();
        for (&o, &i) in out.iter().zip(w.iter()) {
            assert_eq!(o, i);
        }
    }
}

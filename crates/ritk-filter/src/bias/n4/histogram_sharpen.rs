//! Histogram sharpening via Wiener deconvolution for N4 bias correction.

use super::dft::{dft_real_into, idft_real_into, next_pow2};

/// Near-zero weight guard to prevent degenerate histogram sharpening.
const NEAR_ZERO_WEIGHT: f64 = 1e-12;

/// Pre-allocated scratch buffers for [`histogram_sharpen`], eliminating
/// per-iteration heap allocations inside the N4 inner loop.
///
/// All buffers are sized at construction time and reused across calls.
/// `w_sharp` carries the output of the last `histogram_sharpen` call.
pub(crate) struct HistogramSharpenScratch {
    /// Raw histogram counts (length `n_bins`).
    pub(crate) h_raw: Vec<f64>,
    /// Normalised histogram density (length `n_bins`).
    pub(crate) h: Vec<f64>,
    /// Gaussian kernel (length varies with σ; resized as needed).
    pub(crate) g: Vec<f64>,
    /// DFT of normalised histogram (length `n_dft`).
    pub(crate) h_hat: Vec<(f64, f64)>,
    /// DFT of Gaussian kernel (length `n_dft`).
    pub(crate) g_hat: Vec<(f64, f64)>,
    /// Wiener-deconvolved DFT coefficients (length `n_dft`).
    pub(crate) h_sharp_hat: Vec<(f64, f64)>,
    /// IDFT output before clamping (length `n_dft`, only first `n_bins` used).
    pub(crate) h_sharp_raw: Vec<f64>,
    /// Clamped sharpened histogram density (length `n_bins`).
    pub(crate) h_sharp: Vec<f64>,
    /// CDF of observed histogram H (length `n_bins`).
    pub(crate) cdf_h: Vec<f64>,
    /// CDF of sharpened histogram H_sharp (length `n_bins`).
    pub(crate) cdf_s: Vec<f64>,
    /// Output: Wiener-sharpened voxel intensities (length `n_voxels`).
    pub(crate) w_sharp: Vec<f32>,
}

impl HistogramSharpenScratch {
    /// Pre-allocate all buffers for the given histogram and volume sizes.
    ///
    /// `n_dft` is computed as `next_pow2(n_bins)` internally so that DFT
    /// buffers are correctly sized from the start.
    pub(crate) fn new(n_bins: usize, n_voxels: usize) -> Self {
        let n_dft = next_pow2(n_bins.max(2));
        Self {
            h_raw: vec![0.0; n_bins],
            h: vec![0.0; n_bins],
            // g is variable-length; start with a reasonable capacity.
            g: Vec::with_capacity(n_bins),
            h_hat: vec![(0.0, 0.0); n_dft],
            g_hat: vec![(0.0, 0.0); n_dft],
            h_sharp_hat: vec![(0.0, 0.0); n_dft],
            h_sharp_raw: vec![0.0; n_dft],
            h_sharp: vec![0.0; n_bins],
            cdf_h: vec![0.0; n_bins],
            cdf_s: vec![0.0; n_bins],
            w_sharp: vec![0.0; n_voxels],
        }
    }

    /// Ensure DFT-length buffers can hold `n_dft` elements, resizing if needed.
    ///
    /// This handles the rare case where `n_bins` changes between calls
    /// (not expected in current N4 usage, but defensive).
    pub(crate) fn ensure_dft_capacity(&mut self, n_dft: usize) {
        if self.h_hat.len() < n_dft {
            self.h_hat.resize(n_dft, (0.0, 0.0));
            self.g_hat.resize(n_dft, (0.0, 0.0));
            self.h_sharp_hat.resize(n_dft, (0.0, 0.0));
            self.h_sharp_raw.resize(n_dft, 0.0);
        }
    }

    /// Ensure histogram-length buffers can hold `n_bins` elements.
    pub(crate) fn ensure_bins_capacity(&mut self, n_bins: usize) {
        if self.h_raw.len() < n_bins {
            self.h_raw.resize(n_bins, 0.0);
            self.h.resize(n_bins, 0.0);
            self.h_sharp.resize(n_bins, 0.0);
            self.cdf_h.resize(n_bins, 0.0);
            self.cdf_s.resize(n_bins, 0.0);
        }
    }

    /// Ensure the voxel output buffer can hold `n_voxels` elements.
    pub(crate) fn ensure_voxels_capacity(&mut self, n_voxels: usize) {
        if self.w_sharp.len() < n_voxels {
            self.w_sharp.resize(n_voxels, 0.0);
        }
    }
}

/// Histogram sharpening via Wiener deconvolution, followed by CDF-based
/// quantile transfer from H_observed to H_sharp.
///
/// Writes the result into `scratch.w_sharp` instead of returning a new `Vec`,
/// eliminating per-iteration heap allocations in the N4 inner loop.
///
/// # Algorithm
/// 1. Build histogram H of `w` with `n_bins` bins.
/// 2. σ_bins = max(0.5, noise_fraction · n_bins).
/// 3. Gaussian kernel G(σ_bins), zero-padded DFT of length N = next_pow2(n_bins).
/// 4. Wiener deconvolution:
///    Ĥ_sharp\[k\] = Ĥ\[k\]·Ĝ*\[k\] / (|Ĝ\[k\]|² + σ_noise)
///    where σ_noise = 0.01 · max_k |Ĥ\[k\]|².
/// 5. IDFT → H_sharp; clamp negatives to 0.
/// 6. CDF transfer: each voxel intensity maps to the quantile bin in H_sharp
///    whose CDF matches the voxel's CDF rank in H.
pub(crate) fn histogram_sharpen(
    w: &[f32],
    n_bins: usize,
    noise_fraction: f64,
    scratch: &mut HistogramSharpenScratch,
) -> anyhow::Result<()> {
    let n_voxels = w.len();

    if w.is_empty() || n_bins < 2 {
        scratch.ensure_voxels_capacity(n_voxels);
        scratch.w_sharp[..n_voxels].copy_from_slice(w);
        return Ok(());
    }

    let w_min = w.iter().cloned().fold(f32::INFINITY, f32::min);
    let w_max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (w_max - w_min) as f64;

    // Degenerate case: constant or near-constant input — no sharpening needed.
    if range < 1e-6 {
        scratch.ensure_voxels_capacity(n_voxels);
        scratch.w_sharp[..n_voxels].copy_from_slice(w);
        return Ok(());
    }

    let bin_width = range / n_bins as f64;

    // Ensure scratch buffers are large enough for current parameters.
    // Must happen before any &mut slices are taken from scratch fields.
    let n_dft = next_pow2(n_bins.max(2));
    scratch.ensure_bins_capacity(n_bins);
    scratch.ensure_dft_capacity(n_dft);
    scratch.ensure_voxels_capacity(n_voxels);

    // ── Build histogram (normalised to probability density) ───────────────
    // Normalisation is required so that Ĥ[0] = 1 and the Wiener noise_power
    // is calibrated on a [0, 1] scale. Raw-count histograms have
    // Ĥ[0] = N_voxels, making noise_power = 0.01·N² which dominates every
    // non-DC frequency and collapses H_sharp to a constant.
    let h_raw = &mut scratch.h_raw[..n_bins];
    h_raw.fill(0.0);
    for &wi in w {
        let b = (((wi - w_min) as f64 / bin_width).floor() as isize).clamp(0, n_bins as isize - 1)
            as usize;
        h_raw[b] += 1.0;
    }
    let total_raw: f64 = h_raw.iter().sum::<f64>().max(1.0);
    let h = &mut scratch.h[..n_bins];
    for (i, &raw) in h_raw.iter().enumerate() {
        h[i] = raw / total_raw;
    }

    // ── Gaussian kernel (σ in histogram-bin units) ─────────────────────────
    let sigma_bins = (noise_fraction * n_bins as f64).max(0.5);
    scratch.g = crate::gaussian_kernel(sigma_bins, None);

    // ── Wiener deconvolution in DFT domain ─────────────────────────────────
    let h_hat = &mut scratch.h_hat[..n_dft];
    dft_real_into(h, n_dft, h_hat);

    let g_hat = &mut scratch.g_hat[..n_dft];
    dft_real_into(&scratch.g, n_dft, g_hat);

    // noise_power = 0.01 · max_k |Ĥ[k]|²
    // With the normalised histogram Ĥ[0] = 1, so noise_power ≤ 0.01 and
    // the Wiener filter can now properly deconvolve non-DC frequency components
    // where |Ĝ[k]|² is comparable to or greater than noise_power.
    let noise_power = 0.01
        * h_hat
            .iter()
            .map(|(re, im)| re * re + im * im)
            .fold(0.0f64, f64::max);

    // Ĥ_sharp\[k\] = Ĥ\[k\]·Ĝ*\[k\] / (|Ĝ\[k\]|² + σ_noise)
    // where Ĥ·Ĝ* = (hr + i·hi)·(gr − i·gi) = (hr·gr + hi·gi) + i·(hi·gr − hr·gi)
    let h_sharp_hat = &mut scratch.h_sharp_hat[..n_dft];
    for ((out, &(hr, hi)), &(gr, gi)) in h_sharp_hat.iter_mut().zip(h_hat.iter()).zip(g_hat.iter())
    {
        let num_re = hr * gr + hi * gi;
        let num_im = hi * gr - hr * gi;
        let denom = gr * gr + gi * gi + noise_power;
        if denom < f64::EPSILON {
            *out = (0.0, 0.0);
        } else {
            *out = (num_re / denom, num_im / denom);
        }
    }

    // IDFT; clamp negatives — probability density must be ≥ 0.
    let h_sharp_raw = &mut scratch.h_sharp_raw[..n_dft];
    idft_real_into(h_sharp_hat, n_bins, h_sharp_raw);
    let h_sharp = &mut scratch.h_sharp[..n_bins];
    for (i, &v) in h_sharp_raw[..n_bins].iter().enumerate() {
        h_sharp[i] = v.max(0.0);
    }

    // ── CDF construction ───────────────────────────────────────────────────
    // h is already normalised (sums to ~1); use raw counts for the guard.
    let total_h: f64 = total_raw;
    let total_s: f64 = h_sharp.iter().sum();

    // Guard 1: if sharpening collapsed the density to zero, pass through.
    if total_h < 1.0 || total_s < NEAR_ZERO_WEIGHT {
        scratch.w_sharp[..n_voxels].copy_from_slice(w);
        return Ok(());
    }

    // Guard 2: concentration check.
    // For a sharpened histogram, peak bins must be more prominent than in the
    // original. If the maximum normalised bin value in H_sharp is lower than
    // in H, the Wiener deconvolution broadened the distribution (this happens
    // for discrete-spike inputs where H_true ≈ H_observed and there is no
    // G_noise component to invert). Applying the CDF transfer in this case
    // widens voxel intensities, increasing CoV. Detect and bail out.
    let max_h: f64 = h.iter().cloned().fold(0.0_f64, f64::max);
    let max_s: f64 = {
        let s_sum = total_s.max(NEAR_ZERO_WEIGHT);
        h_sharp.iter().map(|&v| v / s_sum).fold(0.0_f64, f64::max)
    };
    if max_s <= max_h * 1.01 {
        // Sharpening did not increase peak concentration; return input unchanged.
        scratch.w_sharp[..n_voxels].copy_from_slice(w);
        return Ok(());
    }

    let cdf_h = &mut scratch.cdf_h[..n_bins];
    {
        let mut acc = 0.0f64;
        for (i, &hi) in h.iter().enumerate() {
            acc += hi;
            cdf_h[i] = acc / total_h;
        }
    }

    let cdf_s = &mut scratch.cdf_s[..n_bins];
    {
        let mut acc = 0.0f64;
        for (i, &si) in h_sharp.iter().enumerate() {
            acc += si;
            cdf_s[i] = acc / total_s;
        }
    }

    // ── Quantile transfer per voxel ────────────────────────────────────────
    // For each voxel: find its CDF rank in H, then find the matching bin in H_sharp.
    // Widen to f64 for accumulation precision; bin boundaries require sub-f32 precision.
    let w_min_wide = w_min as f64;
    let w_sharp = &mut scratch.w_sharp[..n_voxels];
    for (i, &wi) in w.iter().enumerate() {
        let bin_i = (((wi - w_min) as f64 / bin_width).floor() as isize)
            .clamp(0, n_bins as isize - 1) as usize;
        let q = cdf_h[bin_i];
        // First index in the monotone cdf_s where cdf_s[t] ≥ q.
        let target = cdf_s.partition_point(|&v| v < q).min(n_bins - 1);
        w_sharp[i] = (w_min_wide + (target as f64 + 0.5) * bin_width) as f32;
    }

    Ok(())
}

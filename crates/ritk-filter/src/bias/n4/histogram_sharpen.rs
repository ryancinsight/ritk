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

/// N4 histogram sharpening, faithful to ITK's `N4BiasFieldCorrectionImageFilter`
/// `SharpenImage` (Tustison 2010).
///
/// Writes the sharpened per-voxel log-intensities into `scratch.w_sharp`.
///
/// # Algorithm
/// 1. Histogram `H` of `w` over `[w_min, w_max]`, `n_bins` bins (raw counts).
/// 2. Circular Gaussian `G` whose width comes from the bias-field FWHM
///    (`fwhm`, in log-intensity units): `σ = FWHM / (2√(2 ln 2))`,
///    `σ_bins = σ / bin_width`, with `exp_factor = 4 ln 2 / fwhm_bins²` and the
///    ITK normalisation `√(exp_factor/π)`.
/// 3. Wiener deconvolution `Û = Ĥ·conj(Ĝ) / (|Ĝ|² + wiener_noise)` → `U`
///    (the deconvolved/sharpened density), clamped to ≥ 0.
/// 4. Expectation mapping `E[v|u]` via two Gaussian convolutions:
///    `E[i] = (U·c ⋆ G)[i] / (U ⋆ G)[i]`, where `c[i]` is the bin centre. This
///    pulls each observed intensity toward the deconvolved tissue peaks — the
///    actual N4 sharpening, replacing the earlier CDF/quantile transfer (which
///    was rank-preserving, hence insensitive to the smoothing width, leaving the
///    filter behaving like N3).
/// 5. Each voxel maps by linear interpolation of `E` at its continuous bin index.
pub(crate) fn histogram_sharpen(
    w: &[f32],
    n_bins: usize,
    fwhm: f64,
    wiener_noise: f64,
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
    let w_min_wide = w_min as f64;

    let n_dft = next_pow2(n_bins.max(2));
    scratch.ensure_bins_capacity(n_bins);
    scratch.ensure_dft_capacity(n_dft);
    scratch.ensure_voxels_capacity(n_voxels);
    if scratch.g.len() < n_dft {
        scratch.g.resize(n_dft, 0.0);
    }

    // ── 1. Histogram (raw counts) ─────────────────────────────────────────
    let h = &mut scratch.h[..n_bins];
    h.fill(0.0);
    for &wi in w {
        let b = (((wi - w_min) as f64 / bin_width).floor() as isize).clamp(0, n_bins as isize - 1)
            as usize;
        h[b] += 1.0;
    }

    // ── 2. Circular Gaussian G (ITK normalisation) ────────────────────────
    // FWHM is expressed in the histogram's (log-intensity) units → bins.
    let fwhm_bins = (fwhm / bin_width).max(1e-3);
    let exp_factor = 4.0 * std::f64::consts::LN_2 / (fwhm_bins * fwhm_bins); // 1/(2σ²)
    let scale = (exp_factor / std::f64::consts::PI).sqrt();
    {
        let g = &mut scratch.g[..n_dft];
        for (k, gk) in g.iter_mut().enumerate() {
            // Centre the Gaussian at index 0 with circular wraparound.
            let arg = if k <= n_dft / 2 {
                k as f64
            } else {
                k as f64 - n_dft as f64
            };
            *gk = scale * (-exp_factor * arg * arg).exp();
        }
    }

    // ── 3. Wiener deconvolution: Û = Ĥ·conj(Ĝ) / (|Ĝ|² + noise) ────────────
    dft_real_into(&scratch.h[..n_bins], n_dft, &mut scratch.h_hat[..n_dft]);
    dft_real_into(&scratch.g[..n_dft], n_dft, &mut scratch.g_hat[..n_dft]);
    {
        let (vf, vg, vu) = (&scratch.h_hat, &scratch.g_hat, &mut scratch.h_sharp_hat);
        for ((out, &(fr, fi)), &(gr, gi)) in
            vu[..n_dft].iter_mut().zip(vf[..n_dft].iter()).zip(vg[..n_dft].iter())
        {
            let num_re = fr * gr + fi * gi; // Vf·conj(Vg)
            let num_im = fi * gr - fr * gi;
            let denom = gr * gr + gi * gi + wiener_noise;
            *out = (num_re / denom, num_im / denom);
        }
    }
    idft_real_into(
        &scratch.h_sharp_hat[..n_dft],
        n_bins,
        &mut scratch.h_sharp_raw[..n_dft],
    );
    // U = clamp(real(IFFT(Û)), ≥ 0): the deconvolved density.
    for i in 0..n_bins {
        scratch.h_sharp[i] = scratch.h_sharp_raw[i].max(0.0);
    }

    // ── 4. Expectation map E[i] = (U·c ⋆ G)[i] / (U ⋆ G)[i] ────────────────
    // numerator = conv(U·centre, G); denominator = conv(U, G).
    // Reuse `h` for (U·centre), then FFT and multiply by Vg (= conv with G).
    for i in 0..n_bins {
        let centre = w_min_wide + (i as f64 + 0.5) * bin_width;
        scratch.h[i] = scratch.h_sharp[i] * centre;
    }
    dft_real_into(&scratch.h[..n_bins], n_dft, &mut scratch.h_hat[..n_dft]);
    dft_real_into(
        &scratch.h_sharp[..n_bins],
        n_dft,
        &mut scratch.h_sharp_hat[..n_dft],
    );
    for k in 0..n_dft {
        let (gr, gi) = scratch.g_hat[k];
        let (nr, ni) = scratch.h_hat[k];
        scratch.h_hat[k] = (nr * gr - ni * gi, nr * gi + ni * gr); // numerator ⋆ G
        let (dr, di) = scratch.h_sharp_hat[k];
        scratch.h_sharp_hat[k] = (dr * gr - di * gi, dr * gi + di * gr); // denominator ⋆ G
    }
    idft_real_into(&scratch.h_hat[..n_dft], n_bins, &mut scratch.cdf_h[..n_bins]);
    idft_real_into(
        &scratch.h_sharp_hat[..n_dft],
        n_bins,
        &mut scratch.cdf_s[..n_bins],
    );
    // E[i] = numerator/denominator (fall back to the bin centre when empty),
    // stored back into cdf_h.
    for i in 0..n_bins {
        let num = scratch.cdf_h[i];
        let den = scratch.cdf_s[i];
        scratch.cdf_h[i] = if den.abs() > NEAR_ZERO_WEIGHT {
            num / den
        } else {
            w_min_wide + (i as f64 + 0.5) * bin_width
        };
    }

    // ── 5. Per-voxel linear interpolation of E at the continuous bin index ──
    let HistogramSharpenScratch { w_sharp, cdf_h, .. } = scratch;
    for (i, &wi) in w.iter().enumerate() {
        // Continuous index aligned to bin centres (centre of bin j is at j).
        let cidx = (wi - w_min) as f64 / bin_width - 0.5;
        let sharp = if cidx <= 0.0 {
            cdf_h[0]
        } else if cidx >= (n_bins - 1) as f64 {
            cdf_h[n_bins - 1]
        } else {
            let lo = cidx.floor() as usize;
            let frac = cidx - lo as f64;
            cdf_h[lo] * (1.0 - frac) + cdf_h[lo + 1] * frac
        };
        w_sharp[i] = sharp as f32;
    }

    Ok(())
}

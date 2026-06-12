//! Histogram matching (histogram specification).
//!
//! Transforms the intensity histogram of a source image so that it approximates
//! the intensity histogram of a reference image.
//!
//! # Mathematical Specification
//!
//! Let F_src and F_ref denote the empirical CDFs of the source and reference images.
//! The histogram matching transform T is defined as:
//!
//!   T(v) = F_ref⁻¹( F_src(v) )
//!
//! where F_ref⁻¹ is the generalised inverse CDF (quantile function) of the reference.
//!
//! # Algorithm
//! 1. Sort both source and reference value arrays in ascending order.
//! 2. Build a piecewise-linear LUT over `num_bins` equally-spaced intensity bins
//!    spanning [src_min, src_max]:
//!    - For each bin centre v, compute CDF_src(v) = |{x ∈ src : x ≤ v}| / n_src
//!      via binary search (partition_point) on sorted_src.
//!    - Map CDF_src(v) to a reference intensity via the empirical quantile:
//!   - ref_idx = ⌊CDF_src(v) · (n_ref − 1)⌋ → lut\[bin\] = sorted_ref\[ref_idx\].
//! 3. For each source pixel value, look up its mapped intensity via the LUT
//!    with linear interpolation between adjacent bin entries.
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the source input.
//! - The LUT spans exclusively \[src_min, src_max\] of the source distribution.
//!   Values ≤ src_min clamp to lut\[0\]; values ≥ src_max clamp to lut\[last\].
//! - A constant source image (src_min == src_max) is returned unchanged because
//!   no CDF slope can be estimated from a degenerate distribution.

use crate::filter::ops::{extract_vec_infallible, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Histogram matcher via empirical CDF-based lookup.
///
/// Applies F_ref⁻¹ ∘ F_src to every pixel of the source image, where F_src and
/// F_ref are the empirical CDFs of the source and reference images respectively.
pub struct HistogramMatcher {
    /// Number of equally-spaced intensity bins used to build the lookup table.
    pub num_bins: usize,
}

impl HistogramMatcher {
    /// Create a `HistogramMatcher` with `num_bins` lookup-table bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn new(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Match the intensity histogram of `source` to that of `reference`.
    ///
    /// # Arguments
    /// * `source`    – Image whose histogram is to be transformed.
    /// * `reference` – Image whose histogram serves as the target distribution.
    ///
    /// # Returns
    /// A new `Image` with the same shape and spatial metadata as `source` but
    /// with intensities mapped so that the histogram approximates `reference`.
    ///
    /// If `source` is constant (src_min == src_max within float epsilon), the
    /// source image is returned unchanged.
    pub fn match_histograms<B: Backend, const D: usize>(
        &self,
        source: &Image<B, D>,
        reference: &Image<B, D>,
    ) -> Image<B, D> {
        let shape: [usize; D] = source.shape();

        // ── 1. Extract pixel arrays ───────────────────────────────────────────
        let (src_vec, _) = extract_vec_infallible(source);
        let src_slice = &src_vec;
        let (ref_vec, _) = extract_vec_infallible(reference);
        let ref_slice = &ref_vec;

        // ── 2. Sort both arrays (ascending) ──────────────────────────────────
        let mut sorted_src: Vec<f32> = src_slice.to_vec();
        sorted_src.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_ref: Vec<f32> = ref_slice.to_vec();
        sorted_ref.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_src = sorted_src.len();
        let n_ref = sorted_ref.len();

        let src_min = sorted_src[0];
        let src_max = sorted_src[n_src - 1];

        // Constant source: CDF slope is undefined — return source unchanged.
        if (src_max - src_min).abs() < f32::EPSILON {
            return Image::new(
                source.data().clone(),
                *source.origin(),
                *source.spacing(),
                *source.direction(),
            );
        }

        // ── 3. Build LUT over num_bins equally-spaced bin centres ─────────────
        //
        // lut[i] = T(bin_centre[i])  where T = F_ref⁻¹ ∘ F_src
        //
        // The i-th bin centre is:  src_min + i/(num_bins−1) · (src_max − src_min)
        let num_bins = self.num_bins;
        let mut lut: Vec<f32> = Vec::with_capacity(num_bins);

        for bin in 0..num_bins {
            let t = bin as f64 / (num_bins - 1) as f64; // ∈ [0, 1]
            let bin_val = src_min as f64 + t * (src_max - src_min) as f64;

            // CDF_src(bin_val): fraction of source pixels ≤ bin_val.
            // partition_point returns the first index where sorted_src[i] > bin_val,
            // which equals the count of elements ≤ bin_val.
            let rank = sorted_src.partition_point(|&x| (x as f64) <= bin_val);

            // Normalise to [0, 1].
            let cdf_val = rank as f64 / n_src as f64;

            // Empirical quantile of reference at cdf_val.
            // ref_idx = ⌊cdf_val · (n_ref − 1)⌋ clamped to [0, n_ref − 1].
            let ref_idx = ((cdf_val * (n_ref - 1) as f64).floor() as usize).min(n_ref - 1);
            lut.push(sorted_ref[ref_idx]);
        }

        // ── 4. Apply LUT to every source pixel with linear interpolation ──────
        let bin_width = (src_max - src_min) / (num_bins - 1) as f32;
        let n_total: usize = shape.iter().product();
        let mut output: Vec<f32> = Vec::with_capacity(n_total);

        for &v in src_slice.iter() {
            output.push(Self::apply_lut(v, src_min, src_max, bin_width, &lut));
        }

        // ── 5. Reconstruct image with matched intensities ─────────────────────
        rebuild(output, shape, source)
    }

    /// Map a single pixel value through the LUT with linear interpolation.
    ///
    /// # Clamping contract
    /// - `v ≤ src_min` → `lut[0]`
    /// - `v ≥ src_max` → `lut[last]`
    /// - `src_min < v < src_max` → linear interpolation between adjacent entries
    #[inline]
    fn apply_lut(v: f32, src_min: f32, src_max: f32, bin_width: f32, lut: &[f32]) -> f32 {
        if v <= src_min {
            return lut[0];
        }
        if v >= src_max {
            return *lut.last().expect("LUT must not be empty");
        }

        // Continuous bin position in [0, num_bins − 1).
        let pos = (v - src_min) / bin_width;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(lut.len() - 1);

        // Guard: lo == hi only when bin_width rounds to zero (handled above).
        if lo == hi {
            return lut[lo];
        }

        let frac = pos - lo as f32; // ∈ [0, 1)
        lut[lo].mul_add(1.0 - frac, lut[hi] * frac)
    }
}

impl Default for HistogramMatcher {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
#[path = "tests_histogram_matching.rs"]
mod tests_histogram_matching;

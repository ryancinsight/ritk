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
//!      ref_idx = ⌊CDF_src(v) · (n_ref − 1)⌋  →  lut[bin] = sorted_ref[ref_idx].
//! 3. For each source pixel value, look up its mapped intensity via the LUT
//!    with linear interpolation between adjacent bin entries.
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the source input.
//! - The LUT spans exclusively [src_min, src_max] of the source distribution.
//!   Values ≤ src_min clamp to lut[0]; values ≥ src_max clamp to lut[last].
//! - A constant source image (src_min == src_max) is returned unchanged because
//!   no CDF slope can be estimated from a degenerate distribution.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

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
        let device = source.data().device();
        let shape: [usize; D] = source.shape();

        // ── 1. Extract pixel arrays ───────────────────────────────────────────
        let src_data = source.data().clone().into_data();
        let src_slice = src_data.as_slice::<f32>().expect("f32 source tensor data");

        let ref_data = reference.data().clone().into_data();
        let ref_slice = ref_data
            .as_slice::<f32>()
            .expect("f32 reference tensor data");

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
                source.origin().clone(),
                source.spacing().clone(),
                source.direction().clone(),
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
        let out_tensor =
            Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            out_tensor,
            source.origin().clone(),
            source.spacing().clone(),
            source.direction().clone(),
        )
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
            return *lut.last().unwrap();
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Positive tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_self_match_is_approximately_identity() {
        // Matching a source against itself: T(v) = F_src⁻¹(F_src(v)) ≈ v.
        // Due to discrete LUT quantisation, the tolerance is one LUT step.
        let data: Vec<f32> = (0u16..256).map(|x| x as f32 * 32.0 / 255.0).collect();
        let image = make_image_1d(data.clone());
        let matcher = HistogramMatcher::new(256);
        let result = matcher.match_histograms(&image, &image);
        let values = get_values(&result);

        let step = 32.0_f32 / 255.0; // one LUT bin width = reference quantization step for n=256
        for (i, (&orig, &out)) in data.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - out).abs() <= step + 1e-3,
                "self-match diverged at index {}: orig={} out={} tol={}",
                i,
                orig,
                out,
                step + 1e-3
            );
        }
    }

    #[test]
    fn test_match_shifts_mean_toward_reference() {
        // Source in [0, 10]; reference in [90, 100].
        // After matching, output mean must be close to reference mean ≈ 95.
        let source: Vec<f32> = (0u8..=10).map(|x| x as f32).collect();
        let reference: Vec<f32> = (90u8..=100).map(|x| x as f32).collect();

        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(64);
        let result = matcher.match_histograms(&src_image, &ref_image);
        let values = get_values(&result);

        let out_mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let ref_mean = 95.0_f32;
        assert!(
            (out_mean - ref_mean).abs() < 5.0,
            "output mean {} not close to reference mean {}",
            out_mean,
            ref_mean
        );
    }

    #[test]
    fn test_output_values_bounded_by_reference_range() {
        // All output values must lie within [ref_min, ref_max] plus one LUT step.
        let source: Vec<f32> = (0u8..=20).map(|x| x as f32).collect();
        let reference: Vec<f32> = (50u8..=70).map(|x| x as f32).collect();

        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(64);
        let result = matcher.match_histograms(&src_image, &ref_image);
        let values = get_values(&result);

        let (ref_min, ref_max) = (50.0_f32, 70.0_f32);
        let tol = 1.0_f32; // one reference bin step

        for &v in &values {
            assert!(
                v >= ref_min - tol && v <= ref_max + tol,
                "output value {} outside reference range [{}, {}] ± {}",
                v,
                ref_min,
                ref_max,
                tol
            );
        }
    }

    #[test]
    fn test_output_shape_matches_source() {
        // Output shape must equal source shape regardless of reference shape.
        let source: Vec<f32> = (0u8..16).map(|x| x as f32).collect();
        let reference: Vec<f32> = (0u8..64).map(|x| x as f32).collect();

        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::default();
        let result = matcher.match_histograms(&src_image, &ref_image);

        assert_eq!(
            result.shape(),
            src_image.shape(),
            "output shape must match source shape"
        );
    }

    #[test]
    fn test_preserves_spatial_metadata() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let make_3d = |vals: Vec<f32>| {
            let tensor = Tensor::<TestBackend, 3>::from_data(
                TensorData::new(vals, Shape::new([3, 3, 3])),
                &device,
            );
            let origin = Point::new([1.0, 2.0, 3.0]);
            let spacing = Spacing::new([0.5, 0.5, 0.5]);
            let direction = Direction::<3>::identity();
            Image::new(tensor, origin, spacing, direction)
        };

        let src_vals: Vec<f32> = (0u16..27).map(|x| x as f32).collect();
        let ref_vals: Vec<f32> = (10u16..37).map(|x| x as f32).collect();
        let source = make_3d(src_vals);
        let reference = make_3d(ref_vals);

        let matcher = HistogramMatcher::default();
        let result = matcher.match_histograms(&source, &reference);

        assert_eq!(result.origin(), source.origin(), "origin must be preserved");
        assert_eq!(
            result.spacing(),
            source.spacing(),
            "spacing must be preserved"
        );
        assert_eq!(
            result.direction(),
            source.direction(),
            "direction must be preserved"
        );
        assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
    }

    #[test]
    fn test_monotone_output_for_monotone_input() {
        // A monotonically increasing source matched against a monotonically
        // increasing reference must produce a monotonically non-decreasing output.
        let source: Vec<f32> = (0u8..=16).map(|x| x as f32).collect();
        let reference: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();

        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(128);
        let result = matcher.match_histograms(&src_image, &ref_image);
        let values = get_values(&result);

        for i in 0..values.len().saturating_sub(1) {
            assert!(
                values[i] <= values[i + 1] + 1e-4,
                "monotonicity violated: values[{}]={} > values[{}]={}",
                i,
                values[i],
                i + 1,
                values[i + 1]
            );
        }
    }

    #[test]
    fn test_lut_endpoints_clamp_correctly() {
        // src_min must map to ≈ ref_min; src_max must map to ≈ ref_max.
        // With a 1:1 uniform mapping, the first/last source values map to
        // the first/last reference values.
        let n = 32usize;
        let source: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let reference: Vec<f32> = (100..100 + n).map(|x| x as f32).collect();

        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(256);
        let result = matcher.match_histograms(&src_image, &ref_image);
        let values = get_values(&result);

        // First source pixel (src_min=0) → should map close to ref_min=100.
        assert!(
            (values[0] - 100.0).abs() < 1.0,
            "src_min should map near ref_min=100, got {}",
            values[0]
        );
        // Last source pixel (src_max=31) → should map close to ref_max=131.
        let last = *values.last().unwrap();
        assert!(
            (last - 131.0).abs() < 1.0,
            "src_max should map near ref_max=131, got {}",
            last
        );
    }

    // ── Boundary / edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_constant_source_returns_unchanged() {
        // Constant source: CDF slope is undefined → source returned unchanged.
        let source = make_image_1d(vec![5.0; 8]);
        let reference: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(64);
        let result = matcher.match_histograms(&source, &ref_image);
        let values = get_values(&result);

        for &v in &values {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "constant source must be returned unchanged, got {}",
                v
            );
        }
    }

    #[test]
    fn test_single_source_voxel_does_not_panic() {
        // Single source voxel must produce a single output value without panic.
        let source = make_image_1d(vec![0.5]);
        let reference: Vec<f32> = (0u8..16).map(|x| x as f32).collect();
        let ref_image = make_image_1d(reference);

        let matcher = HistogramMatcher::new(64);
        let result = matcher.match_histograms(&source, &ref_image);

        assert_eq!(result.shape(), [1]);
    }

    #[test]
    fn test_single_reference_voxel() {
        // Single reference value: all output values must equal that reference value.
        let source: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
        let src_image = make_image_1d(source);
        let ref_image = make_image_1d(vec![42.0]);

        let matcher = HistogramMatcher::new(64);
        let result = matcher.match_histograms(&src_image, &ref_image);
        let values = get_values(&result);

        for &v in &values {
            assert!(
                (v - 42.0).abs() < 1e-4,
                "single reference → all outputs must equal 42.0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_default_uses_256_bins() {
        let m = HistogramMatcher::default();
        assert_eq!(m.num_bins, 256);
    }
}

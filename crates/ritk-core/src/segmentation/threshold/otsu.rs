//! Otsu's single-threshold segmentation method.
//!
//! # Mathematical Specification
//!
//! Otsu's method selects the intensity threshold t* that maximises the
//! between-class variance of two intensity classes:
//!
//!   σ²_B(t) = P₁(t) · P₂(t) · (μ₁(t) − μ₂(t))²
//!
//! where:
//! - P₁(t) = Σ_{i=0}^{t−1} h\[i\]              (weight of class 1, bins 0..t−1)
//! - P₂(t) = 1 − P₁(t)                         (weight of class 2, bins t..N−1)
//! - μ₁(t) = Σ_{i=0}^{t−1} i·h\[i\] / P₁(t)   (mean bin index of class 1)
//! - μ₂(t) = Σ_{i=t}^{N−1} i·h\[i\] / P₂(t)   (mean bin index of class 2)
//! - h\[i\] = count\[i\] / n_total              (normalised histogram)
//!
//! The optimal threshold in original intensity units is:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins using prefix sums.
//! Total:                  O(n + N).

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Single-threshold Otsu segmentation.
///
/// Selects a threshold t* that maximises the between-class variance of the
/// intensity histogram, then applies it to produce a binary mask.
pub struct OtsuThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl OtsuThreshold {
    /// Create an `OtsuThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create an `OtsuThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Otsu threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises between-class variance.
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        compute_otsu_threshold_impl(image, self.num_bins)
    }

    /// Apply the Otsu threshold to produce a binary mask.
    ///
    /// - Pixels with intensity ≥ t* → 1.0 (foreground).
    /// - Pixels with intensity <  t* → 0.0 (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let threshold = self.compute(image);
        let device = image.data().device();
        let shape: [usize; D] = image.shape();

        let img_data = image.data().clone().into_data();
        let slice = img_data.as_slice::<f32>().expect("f32 image tensor data");

        let output: Vec<f32> = slice
            .iter()
            .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
            .collect();

        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        )
    }
}

impl Default for OtsuThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute the Otsu threshold with 256 bins.
pub fn otsu_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    compute_otsu_threshold_impl(image, 256)
}

/// Compute the Otsu threshold directly from a flat `&[f32]` slice.
///
/// Equivalent to [`otsu_threshold`] but accepts pre-extracted slice data,
/// enabling zero-copy extraction when the caller has already obtained a slice
/// from the backend primitive (e.g., NdArray `ArcArray::as_slice_memory_order`).
///
/// # Arguments
/// * `slice`    - Flat pixel intensities in any order.
/// * `num_bins` - Number of equally-spaced histogram bins; must be >= 2.
///
/// # Returns
/// The threshold intensity value t* that maximises between-class variance.
/// For an empty or constant input, returns 0.0 or the uniform intensity respectively.
pub fn compute_otsu_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    let n = slice.len();
    if n == 0 {
        return 0.0;
    }

    // -- Intensity range -------------------------------------------------------
    let (x_min, x_max) = slice.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(mn, mx), &v| (mn.min(v), mx.max(v)),
    );

    // Degenerate case: constant image has no separable classes.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f32;

    // -- Build normalised histogram --------------------------------------------
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // -- Total weighted mean over bin indices ----------------------------------
    // Used with the prefix-sum trick: mu2 = (total_mu - mu1_partial) / w2.
    let total_mu: f64 = (0..num_bins).map(|i| i as f64 * h[i]).sum();

    // -- Prefix-sum scan: O(N) threshold search --------------------------------
    // At threshold index t:
    //   Class 1 = bins [0, t-1],   Class 2 = bins [t, N-1].
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;

    let mut w1 = 0.0_f64;        // sum h[0..t-1]
    let mut mu1_partial = 0.0_f64; // sum i*h[i] for i in [0, t-1]

    for t in 1..num_bins {
        // Extend class 1 to include bin t-1.
        w1 += h[t - 1];
        mu1_partial += (t - 1) as f64 * h[t - 1];

        let w2 = 1.0 - w1;

        // Skip degenerate splits where one class is empty.
        if w1 < 1e-12 || w2 < 1e-12 {
            continue;
        }

        let mu1 = mu1_partial / w1;
        let mu2 = (total_mu - mu1_partial) / w2;

        let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);

        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }

    // -- Convert best bin index to intensity units ----------------------------
    // t*_intensity = x_min + best_t / (N - 1) * range
    x_min + best_t as f32 / num_bins_f * range
}


/// Core Otsu threshold computation.
///
/// # Algorithm
/// 1. Extract pixel values to a flat `Vec<f32>`.
/// 2. Determine [x_min, x_max]; handle constant images as a degenerate case.
/// 3. Build a normalised histogram over `num_bins` equally-spaced bins using the
///    mapping: bin(v) = ⌊(v − x_min) / range · (N − 1)⌋ clamped to [0, N−1].
/// 4. Run an O(N) prefix-sum scan over candidate thresholds t ∈ [1, N−1]:
///    σ²_B(t) = P₁(t) · P₂(t) · (μ₁(t) − μ₂(t))²
/// 5. t* = argmax σ²_B.
/// 6. t*_intensity = x_min + t* / (N − 1) · range.
fn compute_otsu_threshold_impl<B: Backend, const D: usize>(
    image: &Image<B, D>,
    num_bins: usize,
) -> f32 {
    let tensor_data = image.data().clone().into_data();
    let slice = tensor_data
        .as_slice::<f32>()
        .expect("f32 image tensor data");

    let n = slice.len();
    if n == 0 {
        return 0.0;
    }

    // ── Intensity range ────────────────────────────────────────────────────────
    let (x_min, x_max) = slice.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(mn, mx), &v| (mn.min(v), mx.max(v)),
    );

    // Degenerate case: constant image has no separable classes.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f32;

    // ── Build normalised histogram ─────────────────────────────────────────────
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // ── Total weighted mean over bin indices ───────────────────────────────────
    // Used with the prefix-sum trick: μ₂ = (total_mu − μ₁_partial) / P₂.
    let total_mu: f64 = (0..num_bins).map(|i| i as f64 * h[i]).sum();

    // ── Prefix-sum scan: O(N) threshold search ─────────────────────────────────
    // At threshold index t:
    //   Class 1 = bins [0, t−1],   Class 2 = bins [t, N−1].
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;

    let mut w1 = 0.0_f64; // Σ h[0..t−1]
    let mut mu1_partial = 0.0_f64; // Σ i·h[i] for i ∈ [0, t−1]

    for t in 1..num_bins {
        // Extend class 1 to include bin t−1.
        w1 += h[t - 1];
        mu1_partial += (t - 1) as f64 * h[t - 1];

        let w2 = 1.0 - w1;

        // Skip degenerate splits where one class is empty.
        if w1 < 1e-12 || w2 < 1e-12 {
            continue;
        }

        let mu1 = mu1_partial / w1;
        let mu2 = (total_mu - mu1_partial) / w2;

        let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);

        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }

    // ── Convert best bin index to intensity units ──────────────────────────────
    // The threshold is at the lower boundary of bin best_t:
    //   t*_intensity = x_min + best_t / (N − 1) · range
    x_min + best_t as f32 / num_bins_f * range
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

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_slice_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
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
    fn test_bimodal_threshold_separates_modes() {
        // Bimodal distribution: 50 values at 20.0 and 50 values at 200.0.
        // Optimal threshold must lie strictly between the two modes.
        let mut data = vec![20.0f32; 50];
        data.extend(vec![200.0f32; 50]);
        let image = make_image_1d(data);
        let t = otsu_threshold(&image);

        assert!(
            t > 20.0,
            "threshold must exceed lower mode (20.0), got {}",
            t
        );
        assert!(
            t < 200.0,
            "threshold must be below upper mode (200.0), got {}",
            t
        );
    }

    #[test]
    fn test_bimodal_apply_correct_labeling() {
        // Lower-mode pixels → background (0.0); upper-mode pixels → foreground (1.0).
        let mut data = vec![10.0f32; 50];
        data.extend(vec![240.0f32; 50]);
        let image = make_image_1d(data);

        let otsu = OtsuThreshold::new();
        let mask = otsu.apply(&image);
        let result = get_slice_1d(&mask);

        for (i, &v) in result[..50].iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "pixel {} (value 10.0) must be background, got {}",
                i, v
            );
        }
        for (i, &v) in result[50..].iter().enumerate() {
            assert_eq!(
                v, 1.0,
                "pixel {} (value 240.0) must be foreground, got {}",
                i, v
            );
        }
    }

    #[test]
    fn test_three_mode_distribution_threshold_between_outer_modes() {
        // Three clusters: 30 × 10.0, 30 × 128.0, 30 × 250.0.
        // Single Otsu threshold must fall strictly between 10.0 and 250.0.
        let mut data = vec![10.0f32; 30];
        data.extend(vec![128.0f32; 30]);
        data.extend(vec![250.0f32; 30]);
        let image = make_image_1d(data);
        let t = otsu_threshold(&image);

        assert!(t > 10.0, "threshold must exceed 10.0, got {}", t);
        assert!(t < 250.0, "threshold must be below 250.0, got {}", t);
    }

    #[test]
    fn test_threshold_is_deterministic() {
        // The same image must always produce the same threshold.
        let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let t1 = otsu_threshold(&image);
        let t2 = otsu_threshold(&image);
        assert_eq!(t1, t2, "threshold must be deterministic");
    }

    #[test]
    fn test_convenience_fn_matches_struct_compute() {
        let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let t_fn = otsu_threshold(&image);
        let t_struct = OtsuThreshold::new().compute(&image);
        assert_eq!(
            t_fn, t_struct,
            "convenience fn and struct::compute must agree"
        );
    }

    #[test]
    fn test_apply_output_is_strictly_binary() {
        // Every output pixel must be exactly 0.0 or 1.0.
        let data: Vec<f32> = (0u8..=200).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let mask = OtsuThreshold::new().apply(&image);
        let values = get_slice_1d(&mask);

        for &v in &values {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {}", v);
        }
    }

    #[test]
    fn test_apply_preserves_spatial_metadata() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0u8..27).map(|x| x as f32).collect::<Vec<f32>>(),
                Shape::new([3, 3, 3]),
            ),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let mask = OtsuThreshold::new().apply(&image);

        assert_eq!(mask.origin(), &origin, "origin must be preserved");
        assert_eq!(mask.spacing(), &spacing, "spacing must be preserved");
        assert_eq!(mask.direction(), &direction, "direction must be preserved");
        assert_eq!(mask.shape(), [3, 3, 3], "shape must be preserved");
    }

    #[test]
    fn test_threshold_lies_within_intensity_range() {
        // t* must satisfy x_min ≤ t* ≤ x_max.
        let data: Vec<f32> = (0u8..=100).map(|x| x as f32 * 0.5).collect();
        let image = make_image_1d(data);
        let t = otsu_threshold(&image);

        assert!(t >= 0.0, "threshold must be ≥ x_min (0.0), got {}", t);
        assert!(t <= 50.0, "threshold must be ≤ x_max (50.0), got {}", t);
    }

    #[test]
    fn test_unequal_class_sizes_threshold_between_modes() {
        // Unbalanced: 90 pixels at 5.0, 10 pixels at 100.0.
        // Threshold must separate the two modes.
        let mut data = vec![5.0f32; 90];
        data.extend(vec![100.0f32; 10]);
        let image = make_image_1d(data);
        let t = otsu_threshold(&image);

        assert!(t > 5.0, "threshold above background mode (5.0), got {}", t);
        assert!(
            t < 100.0,
            "threshold below foreground mode (100.0), got {}",
            t
        );
    }

    #[test]
    fn test_3d_bimodal_threshold_correct() {
        // 3D image: first 13 voxels at 10.0, remaining 14 at 240.0.
        let mut data = vec![10.0f32; 13];
        data.extend(vec![240.0f32; 14]);
        let image = make_image_3d(data, [3, 3, 3]);
        let t = OtsuThreshold::new().compute(&image);

        assert!(
            t > 10.0 && t < 240.0,
            "3D threshold must lie between modes, got {}",
            t
        );
    }

    #[test]
    fn test_3d_apply_output_binary() {
        // All output values in a 3D mask must be 0.0 or 1.0.
        let data: Vec<f32> = (0u8..27)
            .map(|x| if x < 13 { 10.0 } else { 240.0 })
            .collect();
        let image = make_image_3d(data, [3, 3, 3]);
        let mask = OtsuThreshold::new().apply(&image);

        let result_data = mask.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();
        for &v in slice {
            assert!(v == 0.0 || v == 1.0, "3D output must be binary, got {}", v);
        }
    }

    #[test]
    fn test_custom_bins_still_separates_modes() {
        // With 64 bins, threshold must still fall between the two modes.
        let mut data = vec![30.0f32; 50];
        data.extend(vec![220.0f32; 50]);
        let image = make_image_1d(data);

        let t_default = OtsuThreshold::new().compute(&image);
        let t_custom = OtsuThreshold::with_bins(64).compute(&image);

        assert!(
            t_default > 30.0 && t_default < 220.0,
            "256-bin threshold must separate modes, got {}",
            t_default
        );
        assert!(
            t_custom > 30.0 && t_custom < 220.0,
            "64-bin threshold must separate modes, got {}",
            t_custom
        );
    }

    // ── Edge / boundary cases ──────────────────────────────────────────────────

    #[test]
    fn test_constant_image_returns_constant_value() {
        // Constant image → degenerate case → returns the single intensity value.
        let image = make_image_1d(vec![42.0f32; 64]);
        let t = otsu_threshold(&image);
        assert_eq!(t, 42.0, "constant image → threshold = 42.0, got {}", t);
    }

    #[test]
    fn test_single_voxel_returns_its_value() {
        // n = 1: constant image degenerate case.
        let image = make_image_1d(vec![7.0f32]);
        let t = otsu_threshold(&image);
        assert_eq!(t, 7.0, "single voxel → threshold = 7.0, got {}", t);
    }

    #[test]
    fn test_two_voxels_bimodal_threshold_between_values() {
        // [0.0, 255.0]: threshold must lie strictly between them.
        let image = make_image_1d(vec![0.0f32, 255.0f32]);
        let t = otsu_threshold(&image);
        assert!(
            t > 0.0 && t < 255.0,
            "threshold must lie in (0, 255), got {}",
            t
        );
    }

    #[test]
    fn test_default_is_256_bins() {
        let o = OtsuThreshold::default();
        assert_eq!(o.num_bins, 256);
    }

    // ── Negative tests ─────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_one_panics() {
        let _ = OtsuThreshold::with_bins(1);
    }

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_zero_panics() {
        let _ = OtsuThreshold::with_bins(0);
    }

    /// `compute_otsu_threshold_from_slice` must produce bit-identical output to
    /// `OtsuThreshold::compute` for any input slice.
    ///
    /// Mathematical justification: both call the same prefix-sum scan over the same
    /// normalised histogram; the only difference is the data source.
    #[test]
    fn test_compute_otsu_from_slice_matches_filter() {
        use crate::spatial::{Direction, Point, Spacing};
        use burn::tensor::{Shape, Tensor, TensorData};
        use burn_ndarray::NdArray;

        type B = NdArray<f32>;

        // Bimodal distribution: 128 voxels at 10.0 and 128 at 90.0.
        let mut vals: Vec<f32> = vec![10.0_f32; 128];
        vals.extend(vec![90.0_f32; 128]);

        let device = Default::default();
        let tensor = Tensor::<B, 1>::from_data(
            TensorData::new(vals.clone(), Shape::new([256])),
            &device,
        );
        let img = crate::image::Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        );

        let filter = OtsuThreshold::new();
        let threshold_filter = filter.compute(&img);
        let threshold_slice = compute_otsu_threshold_from_slice(&vals, 256);

        assert_eq!(
            threshold_filter.to_bits(),
            threshold_slice.to_bits(),
            "OtsuThreshold::compute vs compute_otsu_threshold_from_slice differ: \
             filter={threshold_filter}, from_slice={threshold_slice}"
        );

        // Threshold must lie between the two modes.
        assert!(
            threshold_filter > 10.0 && threshold_filter < 90.0,
            "threshold {threshold_filter} must lie between 10.0 and 90.0"
        );
    }
}

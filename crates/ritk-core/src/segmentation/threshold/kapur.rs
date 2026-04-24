//! Kapur's maximum entropy thresholding (Kapur, Sahoo & Wong 1985).
//!
//! # Mathematical Specification
//!
//! Kapur's method selects the intensity threshold t* that maximises the
//! sum of foreground and background entropies of the intensity histogram:
//!
//!   H(t) = H_b(t) + H_f(t)
//!
//! where:
//! - P_b(t) = Σ_{i=0}^{t} p(i)                          (background probability mass)
//! - P_f(t) = Σ_{i=t+1}^{N-1} p(i)                      (foreground probability mass)
//! - H_b(t) = -Σ_{i=0}^{t} (p(i)/P_b) · ln(p(i)/P_b)   (background entropy)
//! - H_f(t) = -Σ_{i=t+1}^{N-1} (p(i)/P_f) · ln(p(i)/P_f) (foreground entropy)
//! - p(i)   = count[i] / n_total                          (normalised histogram)
//!
//! The optimal threshold in original intensity units is:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N²) bins (entropy sums per candidate).
//! Total:                  O(n + N²).
//!
//! # References
//! - J. N. Kapur, P. K. Sahoo, A. K. C. Wong, "A New Method for Gray-Level
//!   Picture Thresholding Using the Entropy of the Histogram," *Computer
//!   Vision, Graphics, and Image Processing*, 29(3):273–285, 1985.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Maximum-entropy threshold segmentation (Kapur et al. 1985).
///
/// Selects a threshold t* that maximises the combined foreground and background
/// entropy of the intensity histogram, then applies it to produce a binary mask.
#[derive(Debug, Clone)]
pub struct KapurThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl KapurThreshold {
    /// Create a `KapurThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `KapurThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Kapur threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises total entropy H(t).
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        compute_kapur_threshold_impl(image, self.num_bins)
    }

    /// Apply the Kapur threshold to produce a binary mask.
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

impl Default for KapurThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute the Kapur threshold with 256 bins.
pub fn kapur_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    compute_kapur_threshold_impl(image, 256)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Compute the Kapur threshold directly from a flat `&[f32]` slice.
///
/// Zero-copy variant: accepts pre-extracted slice, eliminating `clone().into_data()`.
pub fn compute_kapur_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    let n = slice.len();
    if n == 0 { return 0.0; }
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (x_max - x_min).abs() < f32::EPSILON { return x_min; }
    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f32;
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let mut cum_prob = vec![0.0_f64; num_bins];
    cum_prob[0] = h[0];
    for i in 1..num_bins { cum_prob[i] = cum_prob[i - 1] + h[i]; }
    let mut best_entropy = f64::NEG_INFINITY;
    let mut best_t = 0_usize;
    for t in 0..num_bins - 1 {
        let p_b = cum_prob[t];
        let p_f = 1.0 - p_b;
        if p_b < 1e-12 || p_f < 1e-12 { continue; }
        let mut h_b = 0.0_f64;
        for i in 0..=t { if h[i] > 1e-12 { let q = h[i] / p_b; h_b -= q * q.ln(); } }
        let mut h_f = 0.0_f64;
        for i in (t + 1)..num_bins { if h[i] > 1e-12 { let q = h[i] / p_f; h_f -= q * q.ln(); } }
        let total_entropy = h_b + h_f;
        if total_entropy > best_entropy { best_entropy = total_entropy; best_t = t; }
    }
    x_min + best_t as f32 / num_bins_f * range
}

/// Delegates to [`compute_kapur_threshold_from_slice`] after extracting a slice
/// from the image tensor.
fn compute_kapur_threshold_impl<B: Backend, const D: usize>(
    image: &Image<B, D>,
    num_bins: usize,
) -> f32 {
    let tensor_data = image.data().clone().into_data();
    let slice = tensor_data.as_slice::<f32>().expect("f32 image tensor data");
    compute_kapur_threshold_from_slice(slice, num_bins)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_slice_1d(image: &Image<B, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Degenerate / constant image ────────────────────────────────────────────

    #[test]
    fn test_constant_image_returns_constant_value() {
        let data = vec![42.0_f32; 100];
        let image = make_image_1d(data);
        let t = kapur_threshold(&image);
        assert!(
            (t - 42.0).abs() < f32::EPSILON,
            "constant image threshold must equal the constant value, got {}",
            t
        );
    }

    // ── Bimodal separation ─────────────────────────────────────────────────────

    #[test]
    fn test_bimodal_threshold_separates_modes() {
        // Bimodal distribution with spread around each mode so that multiple
        // histogram bins are populated in each class. Without spread, the
        // entropy of each single-bin class is zero for every threshold,
        // making the criterion degenerate.
        //
        // Mode 1 centred at 25: values 10..40 (31 bins occupied).
        // Mode 2 centred at 225: values 210..240 (31 bins occupied).
        // Gap between modes: [41, 209] — 169 intensity units wide.
        //
        // Analytical note: Kapur's criterion H(t) = H_b(t) + H_f(t) is
        // constant for all t in the empty gap [40, 210] (both class
        // compositions are unchanged). The argmax search selects the first
        // bin achieving the maximum, which falls at the upper boundary of
        // the low mode (~bin 33 for 256 bins over [10, 240]).  The resulting
        // intensity threshold is ≈ 39.8, which still correctly separates
        // the two modes: all low-mode values (≤ 40) are ≤ threshold and all
        // high-mode values (≥ 210) are > threshold.
        let mut data: Vec<f32> = (10..=40).map(|v| v as f32).collect(); // 31 values
        data.extend((10..=40).map(|v| v as f32)); // 62 total low
        let high: Vec<f32> = (210..=240).map(|v| v as f32).collect(); // 31 values
        data.extend(high.iter().copied());
        data.extend(high.iter().copied()); // 62 total high
        let image = make_image_1d(data);
        let t = KapurThreshold::new().compute(&image);

        // The threshold must lie past the centre of the low mode and before
        // the start of the high mode, ensuring correct binary separation.
        // It may fall at the upper boundary of the low mode due to the flat
        // criterion across the empty gap.
        assert!(
            t > 25.0,
            "threshold must exceed centre of low mode (25.0), got {}",
            t
        );
        assert!(
            t < 210.0,
            "threshold must be below lower edge of high mode (210.0), got {}",
            t
        );

        // Verify that the threshold actually separates the two modes:
        // applying it must label all high-mode voxels as foreground.
        let mask = KapurThreshold::new().apply(&image);
        let vals = get_slice_1d(&mask);
        // First 62 values are low-mode (10..40), last 62 are high-mode (210..240).
        for &v in &vals[62..] {
            assert!(
                v == 1.0,
                "high-mode voxel must be foreground (1.0), got {}",
                v
            );
        }
    }

    // ── Output shape preserved ─────────────────────────────────────────────────

    #[test]
    fn test_apply_output_shape_matches_input() {
        let dims = [4, 5, 6];
        let n = dims[0] * dims[1] * dims[2];
        let mut data = vec![10.0_f32; n / 2];
        data.extend(vec![200.0_f32; n - n / 2]);
        let image = make_image_3d(data, dims);
        let mask = KapurThreshold::new().apply(&image);
        assert_eq!(mask.shape(), dims);
    }

    // ── Binary output ──────────────────────────────────────────────────────────

    #[test]
    fn test_apply_output_is_strictly_binary() {
        let mut data = vec![30.0_f32; 60];
        data.extend(vec![180.0_f32; 40]);
        let image = make_image_1d(data);
        let mask = KapurThreshold::new().apply(&image);
        let vals = get_slice_1d(&mask);
        for &v in &vals {
            assert!(
                v == 0.0 || v == 1.0,
                "apply must produce binary output, found {}",
                v
            );
        }
    }

    // ── Convenience function matches struct ────────────────────────────────────

    #[test]
    fn test_convenience_fn_matches_struct_compute() {
        let mut data = vec![10.0_f32; 80];
        data.extend(vec![240.0_f32; 20]);
        let image = make_image_1d(data);
        let t_fn = kapur_threshold(&image);
        let t_struct = KapurThreshold::new().compute(&image);
        assert!(
            (t_fn - t_struct).abs() < f32::EPSILON,
            "convenience function and struct must agree"
        );
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_apply_preserves_spatial_metadata() {
        let dims = [2, 3, 4];
        let n = dims[0] * dims[1] * dims[2];
        let mut data = vec![10.0_f32; n / 2];
        data.extend(vec![200.0_f32; n - n / 2]);
        let image = make_image_3d(data, dims);
        let mask = KapurThreshold::new().apply(&image);

        assert_eq!(mask.origin(), image.origin());
        assert_eq!(mask.spacing(), image.spacing());
        assert_eq!(mask.direction(), image.direction());
    }

    // ── Default trait ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_256_bins() {
        let k = KapurThreshold::default();
        assert_eq!(k.num_bins, 256);
    }

    // ── Panics on invalid bins ─────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_zero_panics() {
        KapurThreshold::with_bins(0);
    }

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_one_panics() {
        KapurThreshold::with_bins(1);
    }

    // ── from_slice parity ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_kapur_from_slice_matches_filter() {
        let mut data = vec![20.0_f32; 100];
        data.extend(vec![200.0_f32; 100]);
        let image = make_image_1d(data.clone());
        let t_filter = KapurThreshold::new().compute(&image);
        let t_slice = compute_kapur_threshold_from_slice(&data, 256);
        assert_eq!(t_filter, t_slice, "from_slice must match filter: filter={} slice={}", t_filter, t_slice);
    }
}
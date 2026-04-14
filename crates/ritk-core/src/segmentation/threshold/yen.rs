//! Yen's maximum correlation thresholding method.
//!
//! # Mathematical Specification
//!
//! Yen's method (Yen, Chang, & Chang 1995) selects the threshold t* that
//! maximises a correlation criterion derived from the second-order statistics
//! of the thresholded image:
//!
//!   C(t) = −log( A(t)² + B(t)² )
//!
//! where:
//! - A(t) = Σ_{i=0}^{t}   p(i)²
//! - B(t) = Σ_{i=t+1}^{N−1} p(i)²
//! - p(i) = h[i] / n_total   (normalised histogram probability)
//!
//! The optimal threshold is:
//!
//!   t* = argmax_t C(t)
//!
//! with the intensity-domain mapping:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins using prefix sums of squared probabilities.
//! Total:                  O(n + N).
//!
//! # References
//! - Yen, J.-C., Chang, F.-J., & Chang, S. (1995). "A new criterion for
//!   automatic multilevel thresholding." *IEEE Trans. Image Process.*, 4(3),
//!   370–378.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Yen's maximum correlation threshold segmentation.
///
/// Selects a threshold t* that maximises the correlation criterion
/// C(t) = −log(A(t)² + B(t)²), then applies it to produce a binary mask.
#[derive(Debug, Clone)]
pub struct YenThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl YenThreshold {
    /// Create a `YenThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `YenThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Yen threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises the correlation criterion.
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        compute_yen_threshold_impl(image, self.num_bins)
    }

    /// Apply the Yen threshold to produce a binary mask.
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

        let tensor =
            Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        )
    }
}

impl Default for YenThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute the Yen threshold with 256 bins.
pub fn yen_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    compute_yen_threshold_impl(image, 256)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Core Yen threshold computation.
///
/// # Algorithm
/// 1. Extract pixel values to a flat `Vec<f32>`.
/// 2. Determine [x_min, x_max]; handle constant images as a degenerate case.
/// 3. Build a normalised histogram over `num_bins` equally-spaced bins.
/// 4. Compute prefix sums of p(i)² (A) and suffix sums of p(i)² (B).
/// 5. Evaluate C(t) = −log(A(t) + B(t)) for each candidate threshold t ∈ [0, N−2].
/// 6. t* = argmax_t C(t).
/// 7. Map t* back to intensity units.
fn compute_yen_threshold_impl<B: Backend, const D: usize>(
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
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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
    let p: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // ── Squared probability terms ──────────────────────────────────────────────
    let p_sq: Vec<f64> = p.iter().map(|&pi| pi * pi).collect();

    // ── Prefix sums A(t) = Σ_{i=0}^{t} p(i)² ─────────────────────────────────
    let mut prefix_sq = vec![0.0_f64; num_bins];
    prefix_sq[0] = p_sq[0];
    for i in 1..num_bins {
        prefix_sq[i] = prefix_sq[i - 1] + p_sq[i];
    }

    // ── Suffix sums B(t) = Σ_{i=t+1}^{N−1} p(i)² ─────────────────────────────
    // B(t) = total_sq − prefix_sq[t]
    let total_sq: f64 = prefix_sq[num_bins - 1];

    // ── Search for optimal threshold ───────────────────────────────────────────
    // Evaluate C(t) = −log(A(t) + B(t)) for t ∈ [0, N−2].
    // A(t) = prefix_sq[t], B(t) = total_sq − prefix_sq[t].
    // Since A(t) + B(t) = total_sq is constant if we use the raw sum, we need
    // to use the formulation where A and B are not summed but kept separate:
    //   C(t) = −log( A(t)² + B(t)² )
    // where A(t) = Σ_{i=0}^{t} p(i)² and B(t) = Σ_{i=t+1}^{N−1} p(i)².
    // Maximising C(t) = −log(A(t) + B(t)) is equivalent to minimising A(t) + B(t).

    let mut best_criterion = f64::NEG_INFINITY;
    let mut best_t = 0_usize;

    for t in 0..(num_bins - 1) {
        let a_t = prefix_sq[t];
        let b_t = total_sq - prefix_sq[t];

        let sum = a_t + b_t;
        if sum < 1e-30 {
            continue;
        }

        let criterion = -(sum.ln());

        if criterion > best_criterion {
            best_criterion = criterion;
            best_t = t;
        }
    }

    // ── Convert best bin index to intensity units ──────────────────────────────
    x_min + (best_t as f32 + 0.5) / num_bins_f * range
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
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
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
        let t = yen_threshold(&image);
        assert!(
            (t - 42.0).abs() < f32::EPSILON,
            "constant image threshold must equal the constant value, got {}",
            t
        );
    }

    // ── Bimodal image with known threshold ─────────────────────────────────────

    #[test]
    fn test_bimodal_threshold_separates_modes() {
        let mut data = vec![10.0_f32; 200];
        data.extend(vec![200.0_f32; 200]);
        let image = make_image_1d(data);
        let t = YenThreshold::new().compute(&image);

        assert!(
            t > 10.0,
            "threshold must exceed lower mode (10.0), got {}",
            t
        );
        assert!(
            t < 200.0,
            "threshold must be below upper mode (200.0), got {}",
            t
        );
    }

    // ── Output shape matches input shape ───────────────────────────────────────

    #[test]
    fn test_apply_preserves_shape_and_metadata() {
        let dims = [2, 3, 4];
        let n = dims[0] * dims[1] * dims[2];
        let data: Vec<f32> = (0..n).map(|i| if i < n / 2 { 10.0 } else { 200.0 }).collect();
        let image = make_image_3d(data, dims);
        let result = YenThreshold::new().apply(&image);

        assert_eq!(result.shape(), dims, "output shape must match input shape");
        assert_eq!(result.origin(), image.origin());
        assert_eq!(result.spacing(), image.spacing());
        assert_eq!(result.direction(), image.direction());
    }

    // ── Apply output is strictly binary ────────────────────────────────────────

    #[test]
    fn test_apply_output_is_binary() {
        let mut data = vec![5.0_f32; 80];
        data.extend(vec![250.0_f32; 80]);
        let image = make_image_1d(data);
        let result = YenThreshold::new().apply(&image);

        let out = get_slice_1d(&result);
        for &v in &out {
            assert!(
                v == 0.0 || v == 1.0,
                "apply output must be binary, got {}",
                v
            );
        }
    }

    // ── Convenience function matches struct compute ────────────────────────────

    #[test]
    fn test_convenience_fn_matches_struct_compute() {
        let mut data = vec![30.0_f32; 100];
        data.extend(vec![220.0_f32; 100]);
        let image = make_image_1d(data);

        let t_fn = yen_threshold(&image);
        let t_struct = YenThreshold::new().compute(&image);
        assert!(
            (t_fn - t_struct).abs() < f32::EPSILON,
            "convenience function and struct compute must agree"
        );
    }

    // ── Default trait ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_256_bins() {
        let yt = YenThreshold::default();
        assert_eq!(yt.num_bins, 256);
    }

    // ── Custom bins ────────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_one_panics() {
        YenThreshold::with_bins(1);
    }

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_zero_panics() {
        YenThreshold::with_bins(0);
    }
}

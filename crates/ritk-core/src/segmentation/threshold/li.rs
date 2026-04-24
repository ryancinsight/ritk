//! Li's minimum cross-entropy thresholding (Li & Tam 1998).
//!
//! # Mathematical Specification
//!
//! Li's method iteratively minimizes the cross-entropy between the original
//! image and its thresholded version. The iteration scheme converges to the
//! threshold that minimizes the Kullback–Leibler divergence of the two-class
//! model from the original intensity distribution.
//!
//! ## Algorithm
//!
//! 1. Compute a normalized histogram h\[i\] over N bins.
//! 2. Initialize: t₀ = μ (global mean intensity in bin-index space).
//! 3. Iterate:
//!      μ_b(t) = Σ_{i=0}^{⌊t⌋}   i·h\[i\] / Σ_{i=0}^{⌊t⌋}   h\[i\]
//!      μ_f(t) = Σ_{i=⌊t⌋+1}^{N-1} i·h\[i\] / Σ_{i=⌊t⌋+1}^{N-1} h\[i\]
//!      t_{n+1} = (μ_b + μ_f) / 2
//! 4. Converge when |t_{n+1} − t_n| < tolerance (1e-6) or max_iterations reached.
//! 5. Convert the converged bin index to intensity units:
//!      t*_intensity = x_min + t* / (N − 1) · range
//!
//! # Complexity
//!
//! Histogram construction: O(n) voxels.
//! Each iteration:          O(N) bins.
//! Total:                   O(n + k·N), k = number of iterations until convergence.
//!
//! # References
//!
//! - Li, C.H. & Tam, P.K.S. (1998). "An iterative algorithm for minimum
//!   cross entropy thresholding." *Pattern Recognition Letters*, 19(8), 771–776.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Li's minimum cross-entropy thresholding.
///
/// Iteratively refines a threshold by computing the midpoint of the
/// foreground and background conditional means until convergence.
#[derive(Debug, Clone)]
pub struct LiThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
    /// Maximum number of iterations before forced termination. Default 1000.
    pub max_iterations: usize,
}

impl LiThreshold {
    /// Create a `LiThreshold` with 256 histogram bins and 1000 max iterations.
    pub fn new() -> Self {
        Self {
            num_bins: 256,
            max_iterations: 1000,
        }
    }

    /// Compute the optimal Li threshold for `image`.
    ///
    /// Returns the intensity value t* that minimizes the cross-entropy
    /// between the image and its binary thresholded version.
    /// For a constant image, returns the image's uniform intensity.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        compute_li_threshold_impl(image, self.num_bins, self.max_iterations)
    }

    /// Apply the Li threshold to produce a binary mask.
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

impl Default for LiThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute the Li threshold with default parameters (256 bins, 1000 iterations).
pub fn li_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    compute_li_threshold_impl(image, 256, 1000)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Compute the Li threshold directly from a flat `&[f32]` slice.
///
/// Zero-copy variant: accepts pre-extracted slice, eliminating `clone().into_data()`.
pub fn compute_li_threshold_from_slice(slice: &[f32], num_bins: usize, max_iterations: usize) -> f32 {
    let n = slice.len();
    if n == 0 { return 0.0; }
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (x_max - x_min).abs() < f32::EPSILON { return x_min; }
    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f64;
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f as f32).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let global_mean: f64 = (0..num_bins).map(|i| i as f64 * h[i]).sum();
    let mut t = global_mean;
    let tolerance = 1e-6_f64;
    for _ in 0..max_iterations {
        let t_floor = (t.floor() as usize).min(num_bins - 1);
        let mut w_b = 0.0_f64; let mut sum_b = 0.0_f64;
        for i in 0..=t_floor { w_b += h[i]; sum_b += i as f64 * h[i]; }
        let mut w_f = 0.0_f64; let mut sum_f = 0.0_f64;
        for i in (t_floor + 1)..num_bins { w_f += h[i]; sum_f += i as f64 * h[i]; }
        if w_b < 1e-12 || w_f < 1e-12 { break; }
        let mu_b = sum_b / w_b; let mu_f = sum_f / w_f;
        let t_new = (mu_b + mu_f) / 2.0;
        if (t_new - t).abs() < tolerance { t = t_new; break; }
        t = t_new;
    }
    x_min + (t as f32) / num_bins_f as f32 * range
}

/// Delegates to [`compute_li_threshold_from_slice`] after extracting a slice
/// from the image tensor.
fn compute_li_threshold_impl<B: Backend, const D: usize>(
    image: &Image<B, D>,
    num_bins: usize,
    max_iterations: usize,
) -> f32 {
    let tensor_data = image.data().clone().into_data();
    let slice = tensor_data.as_slice::<f32>().expect("f32 image tensor data");
    compute_li_threshold_from_slice(slice, num_bins, max_iterations)
}


// ── Tests ──────────────────────────────────────────────────────────────────────

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

    // ── Degenerate / constant image ────────────────────────────────────────────

    #[test]
    fn test_constant_image_returns_constant_value() {
        let image = make_image_1d(vec![42.0; 100]);
        let t = LiThreshold::new().compute(&image);
        assert!(
            (t - 42.0).abs() < 1e-3,
            "constant image threshold must equal the constant value, got {}",
            t
        );
    }

    // ── Bimodal image with known threshold ─────────────────────────────────────

    #[test]
    fn test_bimodal_threshold_between_modes() {
        // 50 voxels at 20.0 and 50 voxels at 200.0.
        // The converged threshold must lie strictly between the two modes.
        let mut data = vec![20.0_f32; 50];
        data.extend(vec![200.0_f32; 50]);
        let image = make_image_1d(data);

        let t = li_threshold(&image);

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

    // ── Output shape matches input shape ───────────────────────────────────────

    #[test]
    fn test_apply_output_shape_matches_input() {
        let dims = [4, 5, 6];
        let total = dims.iter().product::<usize>();
        let data: Vec<f32> = (0..total).map(|i| (i % 2) as f32 * 100.0).collect();
        let image = make_image_3d(data, dims);

        let mask = LiThreshold::new().apply(&image);
        assert_eq!(mask.shape(), dims);
    }

    // ── Apply produces binary output ───────────────────────────────────────────

    #[test]
    fn test_apply_output_is_binary() {
        let mut data = vec![10.0_f32; 50];
        data.extend(vec![180.0_f32; 50]);
        let image = make_image_1d(data);

        let mask = LiThreshold::new().apply(&image);
        let vals = get_slice_1d(&mask);
        for &v in &vals {
            assert!(v == 0.0 || v == 1.0, "mask must be binary, found {}", v);
        }
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_apply_preserves_spatial_metadata() {
        let data = vec![10.0_f32; 30];
        let mut data2 = data.clone();
        data2.extend(vec![200.0_f32; 30]);
        let image = make_image_1d(data2);

        let mask = LiThreshold::new().apply(&image);
        assert_eq!(mask.origin(), image.origin());
        assert_eq!(mask.spacing(), image.spacing());
        assert_eq!(mask.direction(), image.direction());
    }

    // ── Convenience function agrees with struct ────────────────────────────────

    #[test]
    fn test_convenience_fn_matches_struct_compute() {
        let mut data = vec![30.0_f32; 40];
        data.extend(vec![170.0_f32; 60]);
        let image = make_image_1d(data);

        let t_fn = li_threshold(&image);
        let t_struct = LiThreshold::new().compute(&image);
        assert!(
            (t_fn - t_struct).abs() < 1e-9,
            "convenience fn and struct must agree: {} vs {}",
            t_fn,
            t_struct
        );
    }

    // ── Default delegates to new ───────────────────────────────────────────────

    #[test]
    fn test_default_is_256_bins_1000_iters() {
        let d = LiThreshold::default();
        assert_eq!(d.num_bins, 256);
        assert_eq!(d.max_iterations, 1000);
    }

    // ── from_slice parity ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_li_from_slice_matches_filter() {
        let mut data = vec![20.0_f32; 100];
        data.extend(vec![200.0_f32; 100]);
        let image = make_image_1d(data.clone());
        let t_filter = LiThreshold::new().compute(&image);
        let t_slice = compute_li_threshold_from_slice(&data, 256, 1000);
        assert_eq!(t_filter, t_slice, "from_slice must match filter: filter={} slice={}", t_filter, t_slice);
    }
}
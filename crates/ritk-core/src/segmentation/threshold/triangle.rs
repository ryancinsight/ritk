//! Triangle thresholding method (Zack, Rogers & Latt 1977).
//!
//! # Mathematical Specification
//!
//! The triangle algorithm selects a threshold by maximising the perpendicular
//! distance from each histogram bin to a line drawn between the histogram peak
//! and the lowest-count tail.
//!
//! Given a normalised histogram h\[0..N−1\]:
//!
//! 1. Find the peak bin p = argmax h\[i\].
//! 2. Identify the tail bin t as the bin farthest from p (either 0 or N−1)
//!    that has the minimum count in that direction.
//! 3. For each bin i between p and t, compute the perpendicular distance d(i)
//!    from the point (i, h\[i\]) to the line segment (p, h\[p\])→(t, h\[t\]):
//!
//!      d(i) = |A·i + B·h\[i\] + C| / √(A² + B²)
//!
//!    where the line Ax + By + C = 0 passes through (p, h\[p\]) and (t, h\[t\]):
//!      A = h\[t\] − h\[p\]
//!      B = p − t
//!      C = t·h\[p\] − p·h\[t\]
//!
//! 4. t* = argmax_i d(i).
//! 5. Convert to intensity: t*_intensity = x_min + t* · (x_max − x_min) / (N − 1).
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins.
//! Total:                  O(n + N).
//!
//! # References
//! - Zack G.W., Rogers W.E., Latt S.A. (1977). "Automatic measurement of
//!   sister chromatid exchange frequency." *J. Histochem. Cytochem.* 25(7):741–753.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Triangle thresholding segmentation.
///
/// Selects a threshold by maximising the perpendicular distance from each
/// histogram bin to the line connecting the histogram peak and the lowest tail.
#[derive(Debug, Clone)]
pub struct TriangleThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl TriangleThreshold {
    /// Create a `TriangleThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `TriangleThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal triangle threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises the perpendicular distance
    /// to the peak–tail line. For a constant image, returns the image's uniform
    /// intensity (degenerate case).
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        compute_triangle_threshold_impl(image, self.num_bins)
    }

    /// Apply the triangle threshold to produce a binary mask.
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

impl Default for TriangleThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute the triangle threshold with 256 bins.
pub fn triangle_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    compute_triangle_threshold_impl(image, 256)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Core triangle threshold computation.
///
/// # Algorithm
/// 1. Extract pixel values to a flat `Vec<f32>`.
/// 2. Determine \[x_min, x_max\]; handle constant images as a degenerate case.
/// 3. Build a histogram over `num_bins` equally-spaced bins.
/// 4. Find the peak bin (highest count) and the tail bin (farthest from peak
///    at either end of the histogram).
/// 5. For each bin between peak and tail, compute the perpendicular distance
///    to the peak–tail line.
/// 6. t* = argmax distance.
/// 7. Convert t* to intensity units.
fn compute_triangle_threshold_impl<B: Backend, const D: usize>(
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

    // Degenerate case: constant image.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f64;

    // ── Build histogram ────────────────────────────────────────────────────────
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) as f64 / range as f64 * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }

    // ── Find peak bin ──────────────────────────────────────────────────────────
    let peak_bin = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // ── Identify tail bin ──────────────────────────────────────────────────────
    // The tail is the end of the histogram farthest from the peak.
    // If the peak is in the left half, the tail is the rightmost non-zero bin
    // (or N−1). Otherwise, the tail is the leftmost non-zero bin (or 0).
    let tail_bin = if peak_bin <= num_bins / 2 {
        // Tail on the right side.
        counts
            .iter()
            .rposition(|&c| c > 0)
            .unwrap_or(num_bins - 1)
    } else {
        // Tail on the left side.
        counts.iter().position(|&c| c > 0).unwrap_or(0)
    };

    // Degenerate: peak and tail coincide.
    if peak_bin == tail_bin {
        return x_min + peak_bin as f32 / num_bins_f as f32 * range;
    }

    // ── Line equation coefficients ─────────────────────────────────────────────
    // Line through (peak_bin, counts[peak_bin]) and (tail_bin, counts[tail_bin]).
    // Using f64 for numerical stability.
    let x1 = peak_bin as f64;
    let y1 = counts[peak_bin] as f64;
    let x2 = tail_bin as f64;
    let y2 = counts[tail_bin] as f64;

    // Line: A·x + B·y + C = 0
    let a = y2 - y1;
    let b = x1 - x2;
    let c = x2 * y1 - x1 * y2;
    let norm = (a * a + b * b).sqrt();

    // ── Search for maximum perpendicular distance ──────────────────────────────
    let (start, end) = if peak_bin < tail_bin {
        (peak_bin + 1, tail_bin)
    } else {
        (tail_bin + 1, peak_bin)
    };

    let mut best_dist = 0.0_f64;
    let mut best_bin = start;

    for i in start..end {
        let xi = i as f64;
        let yi = counts[i] as f64;
        let dist = (a * xi + b * yi + c).abs() / norm;

        if dist > best_dist {
            best_dist = dist;
            best_bin = i;
        }
    }

    // ── Convert bin index to intensity ─────────────────────────────────────────
    x_min + best_bin as f32 / num_bins_f as f32 * range
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

    // ── Degenerate case ────────────────────────────────────────────────────────

    #[test]
    fn test_constant_image_returns_constant_value() {
        let data = vec![42.0f32; 100];
        let image = make_image_1d(data);
        let t = triangle_threshold(&image);
        assert!(
            (t - 42.0).abs() < f32::EPSILON,
            "constant image must return its value, got {}",
            t
        );
    }

    // ── Bimodal distribution ───────────────────────────────────────────────────

    #[test]
    fn test_bimodal_threshold_between_modes() {
        // Bimodal: strong peak at 30.0 (200 voxels), smaller peak at 200.0 (20 voxels).
        // Triangle algorithm should place the threshold between the two modes.
        let mut data = vec![30.0f32; 200];
        data.extend(vec![200.0f32; 20]);
        let image = make_image_1d(data);
        let t = TriangleThreshold::new().compute(&image);

        assert!(
            t > 30.0,
            "threshold must exceed lower mode (30.0), got {}",
            t
        );
        assert!(
            t < 200.0,
            "threshold must be below upper mode (200.0), got {}",
            t
        );
    }

    // ── Output shape preservation ──────────────────────────────────────────────

    #[test]
    fn test_apply_output_shape_matches_input() {
        let dims = [2, 3, 4];
        let n: usize = dims.iter().product();
        let mut data = vec![10.0f32; n / 2];
        data.extend(vec![200.0f32; n - n / 2]);
        let image = make_image_3d(data, dims);
        let result = TriangleThreshold::new().apply(&image);
        assert_eq!(result.shape(), dims);
    }

    // ── Apply produces binary mask ─────────────────────────────────────────────

    #[test]
    fn test_apply_output_is_binary() {
        let mut data = vec![10.0f32; 60];
        data.extend(vec![200.0f32; 40]);
        let image = make_image_1d(data);
        let result = TriangleThreshold::new().apply(&image);
        let out = get_slice_1d(&result);
        for &v in &out {
            assert!(
                v == 0.0 || v == 1.0,
                "output must be binary (0 or 1), got {}",
                v
            );
        }
    }

    // ── Convenience function matches struct ────────────────────────────────────

    #[test]
    fn test_convenience_fn_matches_struct() {
        let mut data = vec![25.0f32; 80];
        data.extend(vec![180.0f32; 20]);
        let image = make_image_1d(data);
        let t_fn = triangle_threshold(&image);
        let t_struct = TriangleThreshold::new().compute(&image);
        assert!(
            (t_fn - t_struct).abs() < f32::EPSILON,
            "convenience fn and struct must agree"
        );
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_apply_preserves_spatial_metadata() {
        let dims = [2, 3, 4];
        let n: usize = dims.iter().product();
        let mut data = vec![10.0f32; n / 2];
        data.extend(vec![200.0f32; n - n / 2]);
        let image = make_image_3d(data, dims);
        let result = TriangleThreshold::new().apply(&image);
        assert_eq!(result.origin(), image.origin());
        assert_eq!(result.spacing(), image.spacing());
        assert_eq!(result.direction(), image.direction());
    }

    // ── Default is 256 bins ────────────────────────────────────────────────────

    #[test]
    fn test_default_is_256_bins() {
        let t = TriangleThreshold::default();
        assert_eq!(t.num_bins, 256);
    }

    // ── Panics on invalid bin count ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "num_bins must be ≥ 2")]
    fn test_with_bins_one_panics() {
        TriangleThreshold::with_bins(1);
    }
}

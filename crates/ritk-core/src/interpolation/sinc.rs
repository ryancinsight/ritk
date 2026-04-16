//! Lanczos-windowed Sinc interpolation implementation.
//!
//! This module provides mathematically rigorous Sinc interpolation with Lanczos windowing
//! for bandlimited signal reconstruction following the Shannon-Nyquist sampling theorem.
//!
//! # Mathematical Foundation
//!
//! The ideal Sinc interpolator reconstructs a continuous signal from discrete samples:
//! ```text
//! f(x) = Σ f[k] · sinc(x - k)
//! ```
//! where `sinc(x) = sin(πx) / (πx)`.
//!
//! However, the Sinc function has infinite support, making it computationally intractable.
//! The Lanczos window provides an optimal finite support approximation:
//!
//! ```text
//! L_a(x) = sinc(x) · sinc(x/a)   for |x| < a
//!        = 0                       otherwise
//! ```
//!
//! where `a` is the window size (kernel radius). Common values are a=3, 4, or 5.
//!
//! # Properties
//!
//! - **Bandlimited reconstruction**: Optimal for signals sampled above Nyquist rate
//! - **Separable**: Multi-dimensional interpolation via tensor product of 1D kernels
//! - **C² continuity**: Twice differentiable (smoother than linear, less smooth than cubic B-spline)
//! - **Ringing artifacts**: May introduce Gibbs phenomenon near sharp edges
//!
//! # References
//!
//! - Shannon, C. E. (1949). Communication in the presence of noise. *Proc. IRE*, 37(1), 10-21.
//! - Lanczos, C. (1956). *Applied Analysis*. Prentice-Hall.
//! - Turkowski, K. (1990). Filters for common resampling tasks. *Graphics Gems I*, 147-165.

use super::trait_::Interpolator;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Lanczos-windowed Sinc interpolator.
///
/// Implements high-quality bandlimited interpolation using the Lanczos kernel.
/// The kernel window size `a` controls the trade-off between quality and performance.
///
/// # Type Parameters
///
/// * `const A: usize` - Window size (kernel radius). Must be >= 2. Typical values: 3, 4, 5.
///
/// # Examples
///
/// ```ignore
/// use ritk_core::interpolation::LanczosInterpolator;
/// use burn_ndarray::NdArray;
/// use burn::tensor::Tensor;
///
/// type Backend = NdArray<f32>;
/// let device = Default::default();
///
/// // Create a 3-tap Lanczos interpolator (window size = 3)
/// let interpolator = LanczosInterpolator::<3>::new();
///
/// // Interpolate from 3D volume
/// let data = Tensor::<Backend, 3>::zeros([64, 64, 64], &device);
/// let indices = Tensor::<Backend, 2>::from_floats([[32.0, 32.0, 32.0]], &device);
/// let values = interpolator.interpolate(&data, indices);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LanczosInterpolator<const A: usize = 3> {
    _private: (),
}

impl<const A: usize> LanczosInterpolator<A> {
    /// Create a new Lanczos interpolator with window size `A`.
    ///
    /// # Panics
    ///
    /// Panics if `A < 2` (minimum required for meaningful interpolation).
    pub fn new() -> Self {
        assert!(A >= 2, "Lanczos window size must be >= 2, got {}", A);
        Self { _private: () }
    }

    /// Get the window size (kernel radius).
    pub fn window_size(&self) -> usize {
        A
    }

    /// Get the kernel support size (diameter).
    ///
    /// For a Lanczos-a kernel, support is `2a` pixels.
    pub fn support(&self) -> usize {
        2 * A
    }
}

impl<const A: usize> Default for LanczosInterpolator<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the Lanczos kernel value at position x.
///
/// The Lanczos kernel is defined as:
/// ```text
/// L_a(x) = sinc(x) · sinc(x/a)   if 0 < |x| < a
///        = 1                      if x = 0
///        = 0                      otherwise
/// ```
///
/// where `sinc(x) = sin(πx) / (πx)` with the limit at x=0 taken as 1.
///
/// # Arguments
///
/// * `x` - The position at which to evaluate the kernel (in pixels).
///
/// # Returns
///
/// The kernel weight at position `x`.
///
/// # Mathematical Justification
///
/// This function implements the cardinal basis function for bandlimited interpolation.
/// For a signal sampled at the Nyquist rate or higher, Lanczos interpolation provides
/// exact reconstruction of the original continuous signal.
#[inline]
fn lanczos_kernel<const A: usize>(x: f32) -> f32 {
    let abs_x = x.abs();

    // Outside the window support
    if abs_x >= A as f32 {
        return 0.0;
    }

    // At the origin, sinc(0) = 1 by limit
    if abs_x < 1e-7 {
        return 1.0;
    }

    // Compute sinc(x) = sin(πx) / (πx)
    let pi_x = std::f32::consts::PI * x;
    let sinc_x = pi_x.sin() / pi_x;

    // Compute sinc(x/a) = sin(πx/a) / (πx/a)
    let pi_x_over_a = std::f32::consts::PI * x / A as f32;
    let sinc_x_over_a = pi_x_over_a.sin() / pi_x_over_a;

    // Lanczos kernel is the product
    sinc_x * sinc_x_over_a
}

/// Precompute Lanczos kernel weights for a single dimension.
///
/// Given a continuous coordinate, computes the `(2A)` weights and integer offsets
/// for all contributing samples along one dimension.
///
/// # Arguments
///
/// * `coord` - Continuous coordinate (may be fractional)
/// * `dim_size` - Size of the dimension (for bounds checking)
///
/// # Returns
///
/// Vector of (integer_index, weight) pairs for non-zero weights.
fn compute_lanczos_weights<const A: usize>(coord: f32, dim_size: usize) -> Vec<(i64, f32)> {
    let center = coord.floor() as i64;
    let frac = coord - center as f32;

    let mut weights = Vec::with_capacity(2 * A);

    // Sample from center - A + 1 to center + A
    // This covers all positions where the kernel is non-zero
    for k in -(A as i64 - 1)..=(A as i64) {
        let idx = center + k;
        let x = k as f32 - frac;

        // Bounds check: only include valid indices
        if idx >= 0 && (idx as usize) < dim_size {
            let weight = lanczos_kernel::<A>(x);
            if weight.abs() > 1e-10 {
                weights.push((idx, weight));
            }
        }
    }

    weights
}

/// Interpolate a single point in a 3D volume using Lanczos kernel.
///
/// Uses separable tensor product of 1D Lanczos kernels for efficiency:
/// ```text
/// L(x, y, z) = Σᵢ Σⱼ Σₖ L(x-i) · L(y-j) · L(z-k) · f[i, j, k]
/// ```
///
/// # Arguments
///
/// * `data` - 3D tensor [Z, Y, X]
/// * `coords` - Continuous coordinates [x, y, z]
/// * `dims` - Dimensions [d0, d1, d2] = [Z, Y, X]
/// * `device` - Device for tensor allocation
///
/// # Returns
///
/// Interpolated value at the specified coordinates.
fn interpolate_point_3d<B: Backend, const A: usize>(
    data: &Tensor<B, 3>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
) -> f32 {
    let (x, y, z) = (coords[0], coords[1], coords[2]);
    let (d2, d1, d0) = (dims[2], dims[1], dims[0]); // X, Y, Z sizes

    // Precompute weights for each dimension
    let weights_x = compute_lanczos_weights::<A>(x, d2);
    let weights_y = compute_lanczos_weights::<A>(y, d1);
    let weights_z = compute_lanczos_weights::<A>(z, d0);

    // Early exit if no valid weights (should not happen for in-bounds coordinates)
    if weights_x.is_empty() || weights_y.is_empty() || weights_z.is_empty() {
        return 0.0;
    }

    // Flatten the data for efficient gathering
    let flat_data = data.clone().reshape([d0 * d1 * d2]);

    // Accumulate the weighted sum
    // L(x,y,z) = Σᵢ Σⱼ Σₖ L_x(i) · L_y(j) · L_z(k) · f[i,j,k]
    let mut result = 0.0;
    let mut weight_sum = 0.0;

    for (iz, wz) in &weights_z {
        for (iy, wy) in &weights_y {
            for (ix, wx) in &weights_x {
                // Combined weight from separable kernel
                let w = wz * wy * wx;

                // Linear index: data is [Z, Y, X] layout
                // idx = iz * (d1 * d2) + iy * d2 + ix
                let idx = (*iz as usize) * (d1 * d2) + (*iy as usize) * d2 + (*ix as usize);

                // Gather the value
                let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
                    burn::tensor::TensorData::new(vec![idx as i32], burn::tensor::Shape::new([1])),
                    device,
                );
                let val_tensor: Tensor<B, 1> = flat_data.clone().gather(0, idx_tensor);
                let val = val_tensor.into_data().as_slice::<f32>().unwrap()[0];

                result += w * val;
                weight_sum += w;
            }
        }
    }

    // Normalize by weight sum to handle boundary effects
    // This ensures reconstruction fidelity even when some kernel samples
    // fall outside the image bounds
    if weight_sum.abs() > 1e-10 {
        result / weight_sum
    } else {
        0.0
    }
}

/// Interpolate a single point in a 2D image using Lanczos kernel.
///
/// Uses separable tensor product of 1D Lanczos kernels.
///
/// # Arguments
///
/// * `data` - 2D tensor [Y, X]
/// * `coords` - Continuous coordinates [x, y]
/// * `dims` - Dimensions [d0, d1] = [Y, X]
/// * `device` - Device for tensor allocation
///
/// # Returns
///
/// Interpolated value at the specified coordinates.
fn interpolate_point_2d<B: Backend, const A: usize>(
    data: &Tensor<B, 2>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
) -> f32 {
    let (x, y) = (coords[0], coords[1]);
    let (d1, d0) = (dims[1], dims[0]); // X, Y sizes

    // Precompute weights for each dimension
    let weights_x = compute_lanczos_weights::<A>(x, d1);
    let weights_y = compute_lanczos_weights::<A>(y, d0);

    if weights_x.is_empty() || weights_y.is_empty() {
        return 0.0;
    }

    let flat_data = data.clone().reshape([d0 * d1]);

    let mut result = 0.0;
    let mut weight_sum = 0.0;

    for (iy, wy) in &weights_y {
        for (ix, wx) in &weights_x {
            let w = wy * wx;

            // Linear index: data is [Y, X] layout
            let idx = (*iy as usize) * d1 + (*ix as usize);

            let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(vec![idx as i32], burn::tensor::Shape::new([1])),
                device,
            );
            let val_tensor: Tensor<B, 1> = flat_data.clone().gather(0, idx_tensor);
            let val = val_tensor.into_data().as_slice::<f32>().unwrap()[0];

            result += w * val;
            weight_sum += w;
        }
    }

    if weight_sum.abs() > 1e-10 {
        result / weight_sum
    } else {
        0.0
    }
}

impl<B: Backend, const A: usize> Interpolator<B> for LanczosInterpolator<A> {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = indices.device();
        let [n_points, rank] = indices.dims();

        assert_eq!(rank, D, "Indices rank must match data dimensionality");
        assert!(
            D == 2 || D == 3,
            "Lanczos interpolation only supports 2D and 3D tensors"
        );

        let shape = data.shape();
        let dims: Vec<usize> = shape.dims.into();

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        // Process each point
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords_start = i * D;
            let coords: Vec<f32> = (0..D).map(|d| indices_slice[coords_start + d]).collect();

            let value = match D {
                3 => {
                    // SAFETY: We checked D == 3 above, so this transmute is safe
                    let data_3d: &Tensor<B, 3> = unsafe { std::mem::transmute(data as *const _) };
                    interpolate_point_3d::<B, A>(data_3d, &coords, &dims, &device)
                }
                2 => {
                    // SAFETY: We checked D == 2 above, so this transmute is safe
                    let data_2d: &Tensor<B, 2> = unsafe { std::mem::transmute(data as *const _) };
                    interpolate_point_2d::<B, A>(data_2d, &coords, &dims, &device)
                }
                _ => unreachable!("Already asserted D == 2 || D == 3"),
            };

            results.push(value);
        }

        Tensor::from_data(
            burn::tensor::TensorData::new(results, burn::tensor::Shape::new([n_points])),
            &device,
        )
    }
}

/// Type alias for the commonly-used Lanczos-3 interpolator.
pub type SincInterpolator = LanczosInterpolator<3>;

/// Type alias for Lanczos-4 interpolator (higher quality, more expensive).
pub type Lanczos4Interpolator = LanczosInterpolator<4>;

/// Type alias for Lanczos-5 interpolator (highest quality, most expensive).
pub type Lanczos5Interpolator = LanczosInterpolator<5>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_lanczos_kernel_at_origin() {
        // At x=0, kernel should be exactly 1
        let k = lanczos_kernel::<3>(0.0);
        assert!((k - 1.0).abs() < 1e-6, "Expected 1.0 at origin, got {}", k);
    }

    #[test]
    fn test_lanczos_kernel_outside_support() {
        // Outside window size, kernel should be 0
        let k3 = lanczos_kernel::<3>(3.5);
        assert!(k3.abs() < 1e-6, "Expected 0 outside support, got {}", k3);

        let k4 = lanczos_kernel::<4>(5.0);
        assert!(k4.abs() < 1e-6, "Expected 0 outside support, got {}", k4);
    }

    #[test]
    fn test_lanczos_kernel_symmetry() {
        // Kernel should be symmetric around origin
        for x in &[0.1, 0.5, 1.0, 1.5, 2.0] {
            let k_pos = lanczos_kernel::<3>(*x);
            let k_neg = lanczos_kernel::<3>(-*x);
            assert!(
                (k_pos - k_neg).abs() < 1e-6,
                "Kernel not symmetric at x={}: {} vs {}",
                x,
                k_pos,
                k_neg
            );
        }
    }

    #[test]
    fn test_lanczos_kernel_zeros() {
        // Lanczos kernel should have zeros at integer positions (except origin)
        for n in 1..=2 {
            let k = lanczos_kernel::<3>(n as f32);
            assert!(k.abs() < 1e-6, "Expected zero at x={}, got {}", n, k);
        }
    }

    #[test]
    fn test_lanczos_weights_bounds() {
        let weights = compute_lanczos_weights::<3>(5.5, 10);
        for (idx, _w) in &weights {
            assert!(*idx >= 0, "Negative index in weights");
            assert!((*idx as usize) < 10, "Index out of bounds in weights");
        }
    }

    #[test]
    fn test_sinc_interpolator_2d_at_grid_points() {
        let device = Default::default();

        // Create a simple 4x4 image with known values
        let data_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec.clone(), burn::tensor::Shape::new([4, 4])),
            &device,
        );

        let interpolator = SincInterpolator::new();

        // At integer coordinates, should return exact values
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device,
        );

        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        // At (0,0): value should be 0
        assert!(
            (slice[0] - 0.0).abs() < 0.1,
            "Expected ~0.0, got {}",
            slice[0]
        );
        // At (1,0): value should be 1
        assert!(
            (slice[1] - 1.0).abs() < 0.1,
            "Expected ~1.0, got {}",
            slice[1]
        );
        // At (0,1): value should be 4
        assert!(
            (slice[2] - 4.0).abs() < 0.1,
            "Expected ~4.0, got {}",
            slice[2]
        );
        // At (1,1): value should be 5
        assert!(
            (slice[3] - 5.0).abs() < 0.1,
            "Expected ~5.0, got {}",
            slice[3]
        );
    }

    #[test]
    fn test_sinc_interpolator_3d_at_grid_points() {
        let device = Default::default();

        // Create a 2x2x2 volume
        let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
        let data = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data_vec.clone(), burn::tensor::Shape::new([2, 2, 2])),
            &device,
        );

        let interpolator = SincInterpolator::new();

        // Test at corner points
        let indices =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], &device);

        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        // At (0,0,0): should be ~0
        assert!(
            (slice[0] - 0.0).abs() < 0.1,
            "Expected ~0.0, got {}",
            slice[0]
        );
        // At (1,1,1): should be ~111
        assert!(
            (slice[1] - 111.0).abs() < 0.1,
            "Expected ~111.0, got {}",
            slice[1]
        );
    }

    #[test]
    fn test_sinc_interpolator_constant_image() {
        let device = Default::default();

        // Constant image: all values are 42.0
        let data_vec: Vec<f32> = vec![42.0; 64];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([8, 8])),
            &device,
        );

        let interpolator = SincInterpolator::new();

        // Interpolate at various positions
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.5, 0.5], // Center of first quadrant
                [3.7, 2.3], // Arbitrary position
                [7.0, 7.0], // Near edge
            ],
            &device,
        );

        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        // For a constant image, interpolation should return the constant value
        for (i, &val) in slice.iter().enumerate() {
            assert!(
                (val - 42.0).abs() < 0.1,
                "Expected ~42.0 at index {}, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_sinc_interpolator_bandlimited_signal() {
        // Test reconstruction of a bandlimited signal (cosine)
        // Sinc interpolation should perfectly reconstruct signals below Nyquist

        let device = Default::default();
        let n = 32;

        // Generate samples of cos(2πx/8) - frequency well below Nyquist (Nyquist = 0.5 cycles/pixel)
        let period = 8.0;
        let data_vec: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * (i as f32) / period).cos())
            .collect();

        let data = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(data_vec.clone(), burn::tensor::Shape::new([n])),
            &device,
        );

        let interpolator = SincInterpolator::new();

        // Reshape to 2D for interpolator (1D case not directly supported, use [1, N])
        let data_2d = data.clone().reshape([1, n]);
        let x_test = 7.5f32; // Half-pixel offset

        let indices = Tensor::<TestBackend, 2>::from_floats([[x_test, 0.0]], &device);
        let result = interpolator.interpolate(&data_2d, indices);
        let interpolated = result.into_data().as_slice::<f32>().unwrap()[0];

        // Expected value
        let expected = (2.0 * std::f32::consts::PI * x_test / period).cos();

        // Sinc interpolation should closely approximate the true value
        assert!(
            (interpolated - expected).abs() < 0.2,
            "Expected {:.4}, got {:.4}",
            expected,
            interpolated
        );
    }

    #[test]
    fn test_lanczos_interpolator_various_window_sizes() {
        let device = Default::default();

        let data_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([4, 4])),
            &device,
        );

        // Test with different window sizes
        let interp3 = LanczosInterpolator::<3>::new();
        let interp4 = LanczosInterpolator::<4>::new();
        let interp5 = LanczosInterpolator::<5>::new();

        let indices = Tensor::<TestBackend, 2>::from_floats([[1.5, 1.5]], &device);

        let r3 = interp3
            .interpolate(&data, indices.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let r4 = interp4
            .interpolate(&data, indices.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let r5 = interp5
            .interpolate(&data, indices)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];

        // All should give reasonable results (not NaN, not wildly different)
        assert!(r3.is_finite(), "Lanczos-3 produced non-finite result");
        assert!(r4.is_finite(), "Lanczos-4 produced non-finite result");
        assert!(r5.is_finite(), "Lanczos-5 produced non-finite result");

        // Results should be in the valid range
        let min_val = 0.0f32;
        let max_val = 15.0f32;
        assert!(
            r3 >= min_val && r3 <= max_val,
            "Lanczos-3 result {} out of range",
            r3
        );
        assert!(
            r4 >= min_val && r4 <= max_val,
            "Lanczos-4 result {} out of range",
            r4
        );
        assert!(
            r5 >= min_val && r5 <= max_val,
            "Lanczos-5 result {} out of range",
            r5
        );
    }

    #[test]
    #[should_panic(expected = "Lanczos window size must be >= 2")]
    fn test_lanczos_interpolator_invalid_window_size() {
        let _ = LanczosInterpolator::<1>::new();
    }
}

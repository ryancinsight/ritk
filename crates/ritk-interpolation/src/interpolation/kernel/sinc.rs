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

use ritk_core::interpolation::Interpolator;
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
pub(crate) fn lanczos_kernel<const A: usize>(x: f32) -> f32 {
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
pub(crate) fn compute_lanczos_weights<const A: usize>(
    coord: f32,
    dim_size: usize,
) -> Vec<(i64, f32)> {
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
/// * `flat_slice` - Pre-flattened CPU slice of the volume data (layout [Z·Y·X])
/// * `coords` - Continuous coordinates [x, y, z]
/// * `dims` - Dimensions [d0, d1, d2] = [Z, Y, X]
///
/// # Returns
///
/// Interpolated value at the specified coordinates.
fn interpolate_point_3d_flat<const A: usize>(
    flat_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
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

                let val = flat_slice[idx];

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
/// * `flat_slice` - Pre-flattened CPU slice of the image data (layout [Y·X])
/// * `coords` - Continuous coordinates [x, y]
/// * `dims` - Dimensions [d0, d1] = [Y, X]
///
/// # Returns
///
/// Interpolated value at the specified coordinates.
fn interpolate_point_2d_flat<const A: usize>(
    flat_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
) -> f32 {
    let (x, y) = (coords[0], coords[1]);
    let (d1, d0) = (dims[1], dims[0]); // X, Y sizes

    // Precompute weights for each dimension
    let weights_x = compute_lanczos_weights::<A>(x, d1);
    let weights_y = compute_lanczos_weights::<A>(y, d0);

    if weights_x.is_empty() || weights_y.is_empty() {
        return 0.0;
    }

    let mut result = 0.0;
    let mut weight_sum = 0.0;

    for (iy, wy) in &weights_y {
        for (ix, wx) in &weights_x {
            let w = wy * wx;

            // Linear index: data is [Y, X] layout
            let idx = (*iy as usize) * d1 + (*ix as usize);

            let val = flat_slice[idx];

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
        let dims: Vec<usize> = shape.dims;

        // Pre-flatten once — O(1) reshape shared across all points instead of O(volume) per point.
        let n_elements: usize = dims.iter().product();
        let flat_data: Tensor<B, 1> = data.clone().reshape([n_elements]);

        // Extract volume data to CPU once before the point loop so inner loops index a plain
        // &[f32] slice — eliminates the (2A)^D tensor clones/dispatches per query point.
        let flat_cpu = flat_data.into_data();
        let flat_slice: &[f32] = flat_cpu
            .as_slice::<f32>()
            .expect("sinc: volume data must be contiguous f32");

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        // Process each point
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords_start = i * D;
            // Zero-copy slice into the already-extracted indices buffer — no per-point heap allocation.
            let coords = &indices_slice[coords_start..coords_start + D];

            let value = match D {
                3 => interpolate_point_3d_flat::<A>(flat_slice, coords, &dims),
                2 => interpolate_point_2d_flat::<A>(flat_slice, coords, &dims),
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
#[path = "tests_sinc.rs"]
mod tests;

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

use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;

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
/// use ritk_image::tensor::Tensor;
///
/// let interpolator = LanczosInterpolator::<3>::new();
///
/// // Interpolate from 3D volume
/// let data = Tensor::<f32, _>::zeros([64, 64, 64]);
/// let indices = Tensor::<f32, _>::from_slice([1, 3], &[32.0, 32.0, 32.0]);
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
#[inline]
pub(crate) fn lanczos_kernel<const A: usize>(x: f32) -> f32 {
    let abs_x = x.abs();

    if abs_x >= A as f32 {
        return 0.0;
    }

    if abs_x < 1e-7 {
        return 1.0;
    }

    let pi_x = std::f32::consts::PI * x;
    let sinc_x = pi_x.sin() / pi_x;

    let pi_x_over_a = std::f32::consts::PI * x / A as f32;
    let sinc_x_over_a = pi_x_over_a.sin() / pi_x_over_a;

    sinc_x * sinc_x_over_a
}

/// Precompute Lanczos kernel weights for a single dimension.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LanczosWeights {
    /// Stack-allocated taps `(clamped_index, weight)`. Support is up to A=8 (16 taps).
    pub taps: [(i64, f32); 16],
    /// Actual number of taps (`2 * A`).
    pub len: usize,
}

#[inline]
pub(crate) fn compute_lanczos_weights<const A: usize>(
    coord: f32,
    dim_size: usize,
) -> LanczosWeights {
    let center = coord.floor() as i64;
    let frac = coord - center as f32;
    let max_idx = dim_size as i64 - 1;
    let len = 2 * A;

    let mut taps = [(0, 0.0f32); 16];
    debug_assert!(
        len <= 16,
        "Lanczos window size A={} exceeds maximum support of 8",
        A
    );

    for (i, k) in (-(A as i64 - 1)..=(A as i64)).enumerate() {
        if i >= len || i >= 16 {
            break;
        }
        let idx = (center + k).clamp(0, max_idx);
        let x = k as f32 - frac;
        taps[i] = (idx, lanczos_kernel::<A>(x));
    }

    LanczosWeights { taps, len }
}

/// Interpolate a single point in a 3D volume using Lanczos kernel.
fn interpolate_point_3d_flat<const A: usize>(
    flat_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
) -> f32 {
    let (x, y, z) = (coords[0], coords[1], coords[2]);
    let (d2, d1, d0) = (dims[2], dims[1], dims[0]);

    let weights_x = compute_lanczos_weights::<A>(x, d2);
    let weights_y = compute_lanczos_weights::<A>(y, d1);
    let weights_z = compute_lanczos_weights::<A>(z, d0);

    let mut result = 0.0;

    for i in 0..weights_z.len {
        let (iz, wz) = weights_z.taps[i];
        for j in 0..weights_y.len {
            let (iy, wy) = weights_y.taps[j];
            for k in 0..weights_x.len {
                let (ix, wx) = weights_x.taps[k];
                let w = wz * wy * wx;
                let idx = (iz as usize) * (d1 * d2) + (iy as usize) * d2 + (ix as usize);
                result += w * flat_slice[idx];
            }
        }
    }

    result
}

/// Interpolate a single point in a 2D image using Lanczos kernel.
fn interpolate_point_2d_flat<const A: usize>(
    flat_slice: &[f32],
    coords: &[f32],
    dims: &[usize],
) -> f32 {
    let (x, y) = (coords[0], coords[1]);
    let (d1, d0) = (dims[1], dims[0]);

    let weights_x = compute_lanczos_weights::<A>(x, d1);
    let weights_y = compute_lanczos_weights::<A>(y, d0);

    let mut result = 0.0;

    for i in 0..weights_y.len {
        let (iy, wy) = weights_y.taps[i];
        for j in 0..weights_x.len {
            let (ix, wx) = weights_x.taps[j];
            let w = wy * wx;
            let idx = (iy as usize) * d1 + (ix as usize);
            result += w * flat_slice[idx];
        }
    }

    result
}

impl<B, const A: usize> Interpolator<B> for LanczosInterpolator<A>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = data.shape().to_vec();
        let rank = shape.len();
        assert!(
            matches!(rank, 2 | 3),
            "Lanczos interpolation only supports 2D and 3D data"
        );

        let idx_shape = indices.shape();
        assert_eq!(idx_shape.len(), 2, "indices must be a 2D tensor [N, rank]");
        let n_points = idx_shape[0];
        let idx_rank = idx_shape[1];
        assert_eq!(idx_rank, rank, "indices rank must match data rank");

        let data_contig = data.to_contiguous();
        let flat_slice = data_contig.as_slice();
        let idx_contig = indices.to_contiguous();
        let idx_slice = idx_contig.as_slice();

        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords = &idx_slice[i * rank..(i + 1) * rank];
            let value = match rank {
                3 => interpolate_point_3d_flat::<A>(flat_slice, coords, &shape),
                2 => interpolate_point_2d_flat::<A>(flat_slice, coords, &shape),
                _ => unreachable!(),
            };
            results.push(value);
        }

        Tensor::from_slice([n_points], &results)
    }
}

/// Type alias for the commonly-used Lanczos-3 interpolator.
pub type SincInterpolator = LanczosInterpolator<3>;

/// Type alias for Lanczos-4 interpolator (higher quality, more expensive).
pub type Lanczos4Interpolator = LanczosInterpolator<4>;

/// Type alias for Lanczos-5 interpolator (highest quality, most expensive).
pub type Lanczos5Interpolator = LanczosInterpolator<5>;

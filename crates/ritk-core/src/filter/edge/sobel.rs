//! 3-D Sobel gradient filter via separable convolution.
//!
//! # Mathematical Specification
//!
//! The 3-D Sobel operator estimates spatial derivatives using separable
//! convolution kernels that combine a derivative operator with smoothing.
//! For each axis direction, the Sobel kernel is the outer product of three
//! 1-D kernels:
//!
//!   d = \[-1, 0, 1\]  (derivative)
//!   s = \[ 1, 2, 1\]  (smoothing)
//!
//! For the x-derivative: K\_x = s ⊗ s ⊗ d  (smooth z, smooth y, derivative x)
//! For the y-derivative: K\_y = s ⊗ d ⊗ s  (smooth z, derivative y, smooth x)
//! For the z-derivative: K\_z = d ⊗ s ⊗ s  (derivative z, smooth y, smooth x)
//!
//! Each 3×3×3 kernel is applied via three sequential 1-D convolutions
//! with replicate (clamp) boundary padding.
//!
//! ## Normalization
//!
//! The raw convolution output is normalized to approximate the true spatial
//! gradient in physical units. The normalization factor for each component is:
//!
//!   factor = 2 · h · 4 · 4 = 32 · h
//!
//! where h is the physical spacing along the derivative axis. The factor of 2·h
//! accounts for the central-difference step size (the derivative kernel
//! \[-1, 0, 1\] computes f(i+1) − f(i−1), spanning 2 voxels), and each factor
//! of 4 is the sum of one smoothing kernel \[1, 2, 1\].
//!
//! ## Proof sketch (linear ramp)
//!
//! Let I(z, y, x) = x with unit spacing. At any interior voxel:
//!   1. Derivative along x: I(x+1) − I(x−1) = 2
//!   2. Smooth along y: \[1,2,1\] · \[2,2,2\] = 8
//!   3. Smooth along z: \[1,2,1\] · \[8,8,8\] = 32
//!   4. Normalize: 32 / (32 · 1.0) = 1.0  ✓  (true gradient of I = x is 1)
//!
//! ## Gradient Magnitude
//!
//!   |∇I| = √(G\_z² + G\_y² + G\_x²)
//!
//! ## Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to
//! \[0, dim\_size − 1\]. This yields one-sided differences at boundaries
//! (with halved magnitude relative to central differences).
//!
//! # Reference
//!
//! Zucker, S. W. & Hummel, R. A. (1981). "A three-dimensional edge operator."
//! *IEEE Trans. Pattern Analysis and Machine Intelligence*, 3(3), 324–331.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// 3-D Sobel gradient filter.
///
/// Computes spatial derivatives using the 3-D Sobel operator, which combines
/// central-difference derivative estimation with binomial smoothing along
/// the two orthogonal axes. The output is normalized to physical gradient
/// units (intensity per unit spacing).
///
/// ## Kernel structure
///
/// For derivative axis `a` with orthogonal axes `b`, `c`:
///
/// ```text
/// K_a[db][dc][da] = s[db] · s[dc] · d[da]
///   where d = [-1, 0, 1], s = [1, 2, 1]
/// ```
///
/// ## Normalization factor derivation
///
/// | Component        | Factor | Source                                      |
/// |------------------|--------|---------------------------------------------|
/// | Central diff     | 2·h    | \[-1,0,1\] spans 2 voxels of spacing h      |
/// | Smoothing axis 1 | 4      | sum(\[1,2,1\])                               |
/// | Smoothing axis 2 | 4      | sum(\[1,2,1\])                               |
/// | **Total**        | 32·h   |                                             |
#[derive(Debug, Clone)]
pub struct SobelFilter {
    /// Physical voxel spacing \[sz, sy, sx\].
    pub spacing: [f64; 3],
}

impl SobelFilter {
    /// Create a filter with the given physical spacing \[sz, sy, sx\].
    pub fn new(spacing: [f64; 3]) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing \[1.0, 1.0, 1.0\].
    pub fn unit() -> Self {
        Self {
            spacing: [1.0, 1.0, 1.0],
        }
    }

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I| = √(G\_z² + G\_y² + G\_x²).
    /// The output has the same shape and physical metadata (origin, spacing,
    /// direction) as the input.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = sobel_components(&vals, dims, self.spacing);
        let mag: Vec<f32> = gz
            .iter()
            .zip(gy.iter())
            .zip(gx.iter())
            .map(|((&z, &y), &x)| (z * z + y * y + x * x).sqrt())
            .collect();
        Ok(rebuild(mag, dims, image))
    }

    /// Compute the three gradient component images.
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape
    /// and physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<B, 3>,
    ) -> anyhow::Result<(Image<B, 3>, Image<B, 3>, Image<B, 3>)> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = sobel_components(&vals, dims, self.spacing);
        Ok((
            rebuild(gz, dims, image),
            rebuild(gy, dims, image),
            rebuild(gx, dims, image),
        ))
    }
}

// ── sobel_components ─────────────────────────────────────────────────────────────

/// Compute Sobel gradient components (gz, gy, gx) via separable 1-D convolutions.
///
/// For each component:
///   1. Apply derivative kernel \[-1, 0, 1\] along the target axis.
///   2. Apply smoothing kernel \[1, 2, 1\] along each orthogonal axis.
///   3. Normalize by 32 · h\_axis.
///
/// Boundary handling: replicate (clamp) padding.
///
/// # Invariants
///
/// - Interior voxels receive second-order central-difference gradient estimates.
/// - Output lengths equal `nz × ny × nx`.
/// - For a linear field I = c·x\_a, the interior component along axis a equals c,
///   and all orthogonal components equal zero.
fn sobel_components(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
    let smooth: [f32; 3] = [1.0, 2.0, 1.0];

    // G_z: derivative along z (axis 0), smooth along y (axis 1) and x (axis 2).
    let gz = {
        let tmp = convolve_1d_axis(data, dims, 0, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 1, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        let norm = 32.0 * spacing[0] as f32;
        normalize_vec(raw, norm)
    };

    // G_y: derivative along y (axis 1), smooth along z (axis 0) and x (axis 2).
    let gy = {
        let tmp = convolve_1d_axis(data, dims, 1, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        let norm = 32.0 * spacing[1] as f32;
        normalize_vec(raw, norm)
    };

    // G_x: derivative along x (axis 2), smooth along z (axis 0) and y (axis 1).
    let gx = {
        let tmp = convolve_1d_axis(data, dims, 2, &deriv);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 1, &smooth);
        let norm = 32.0 * spacing[2] as f32;
        normalize_vec(raw, norm)
    };

    (gz, gy, gx)
}

/// Divide every element of `v` by `norm`.
#[inline]
fn normalize_vec(v: Vec<f32>, norm: f32) -> Vec<f32> {
    let inv = 1.0 / norm;
    v.into_iter().map(|x| x * inv).collect()
}

/// Apply a 3-tap 1-D convolution along the specified axis with replicate padding.
///
/// `axis`: 0 = z, 1 = y, 2 = x.
/// `kernel`: 3-element filter \[k\_{-1}, k\_0, k\_{+1}\].
///
/// Boundary indices are clamped to \[0, dim\_size − 1\] (replicate padding).
///
/// # Complexity
///
/// O(N) where N = nz × ny × nx. Each voxel performs exactly 3 multiply-adds.
fn convolve_1d_axis(data: &[f32], dims: [usize; 3], axis: usize, kernel: &[f32; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    let stride: usize = match axis {
        0 => ny * nx,
        1 => nx,
        2 => 1,
        _ => unreachable!(),
    };
    let dim_len = dims[axis];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let base = iz * ny * nx + iy * nx + ix;
                let pos = match axis {
                    0 => iz,
                    1 => iy,
                    2 => ix,
                    _ => unreachable!(),
                };

                let mut sum = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let offset = ki as isize - 1; // -1, 0, +1
                    let neighbor = (pos as isize + offset).clamp(0, dim_len as isize - 1) as usize;
                    let neighbor_flat = (base as isize
                        + (neighbor as isize - pos as isize) * stride as isize)
                        as usize;
                    sum += kv * data[neighbor_flat];
                }
                out[base] = sum;
            }
        }
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_sobel.rs"]
mod tests_sobel;

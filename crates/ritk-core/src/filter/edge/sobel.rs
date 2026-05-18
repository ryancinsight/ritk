//! 3-D Sobel gradient filter via separable convolution.
//!
//! # Mathematical Specification
//!
//! The 3-D Sobel operator estimates spatial derivatives using separable
//! convolution kernels that combine a derivative operator with smoothing.
//! For each axis direction, the Sobel kernel is the outer product of three
//! 1-D kernels:
//!
//!   d = [-1, 0, 1]  (derivative)
//!   s = [ 1, 2, 1]  (smoothing)
//!
//! For the x-derivative: K_x = s ⊗ s ⊗ d  (smooth z, smooth y, derivative x)
//! For the y-derivative: K_y = s ⊗ d ⊗ s  (smooth z, derivative y, smooth x)
//! For the z-derivative: K_z = d ⊗ s ⊗ s  (derivative z, smooth y, smooth x)
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
//! [-1, 0, 1] computes f(i+1) − f(i−1), spanning 2 voxels), and each factor
//! of 4 is the sum of one smoothing kernel [1, 2, 1].
//!
//! ## Proof sketch (linear ramp)
//!
//! Let I(z, y, x) = x with unit spacing. At any interior voxel:
//! 1. Derivative along x: I(x+1) − I(x−1) = 2
//! 2. Smooth along y: [1,2,1] · [2,2,2] = 8
//! 3. Smooth along z: [1,2,1] · [8,8,8] = 32
//! 4. Normalize: 32 / (32 · 1.0) = 1.0 ✓ (true gradient of I = x is 1)
//!
//! ## Gradient Magnitude
//!
//! |∇I| = √(G_z² + G_y² + G_x²)
//!
//! ## Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to
//! [0, dim_size − 1]. This yields one-sided differences at boundaries
//! (with halved magnitude relative to central differences).
//!
//! # SIMD boundary/interior split
//!
//! The 1-D convolution kernel `convolve_1d_axis` is split into:
//! 1. **Boundary pass** — processes the first and last voxel of each 1-D
//!    line where the -1 or +1 neighbor index would go out of bounds.
//!    Uses clamped indexing (conditionals).
//! 2. **Interior pass** — processes all remaining voxels with known-in-bounds
//!    neighbor access at uniform stride. No conditionals per iteration,
//!    enabling LLVM auto-vectorization of the FMA chain:
//!    `sum = k[-1]·data[pos-1] + k[0]·data[pos] + k[1]·data[pos+1]`.
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
/// where d = [-1, 0, 1], s = [1, 2, 1]
/// ```
///
/// ## Normalization factor derivation
///
/// | Component        | Factor | Source                                       |
/// |------------------|--------|----------------------------------------------|
/// | Central diff     | 2·h   | [-1,0,1] spans 2 voxels of spacing h         |
/// | Smoothing axis 1 | 4     | sum([1,2,1])                                 |
/// | Smoothing axis 2 | 4     | sum([1,2,1])                                 |
/// | **Total**        | 32·h  |                                              |
#[derive(Debug, Clone)]
pub struct SobelFilter {
    /// Physical voxel spacing [sz, sy, sx].
    pub spacing: [f64; 3],
}

impl SobelFilter {
    /// Create a filter with the given physical spacing [sz, sy, sx].
    pub fn new(spacing: [f64; 3]) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing [1.0, 1.0, 1.0].
    pub fn unit() -> Self {
        Self {
            spacing: [1.0, 1.0, 1.0],
        }
    }

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I| = √(G_z² + G_y² + G_x²).
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
/// 1. Apply derivative kernel [-1, 0, 1] along the target axis.
/// 2. Apply smoothing kernel [1, 2, 1] along each orthogonal axis.
/// 3. Normalize by 32 · h_axis.
///
/// Boundary handling: replicate (clamp) padding.
///
/// # Invariants
///
/// - Interior voxels receive second-order central-difference gradient estimates.
/// - Output lengths equal `nz × ny × nx`.
/// - For a linear field I = c·x_a, the interior component along axis a equals c,
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
/// `kernel`: 3-element filter [k_{-1}, k_0, k_{+1}].
///
/// Boundary indices are clamped to [0, dim_size − 1] (replicate padding).
///
/// # Boundary/interior split
///
/// The loop over each 1-D line is split into:
/// 1. **Boundary pass** — i=0 (clamped i−1) and i=len−1 (clamped i+1).
///    These use conditional neighbor-index clamping.
/// 2. **Interior pass** — i=1..len−2, where both i−1 and i+1 are
///    guaranteed in-bounds. No conditionals, uniform stride.
///    LLVM can auto-vectorize the 3-tap FMA body.
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

                if dim_len <= 1 {
                    // Degenerate: only 1 element along this axis.
                    // All neighbors clamp to pos 0, so:
                    // k[-1]·x + k[0]·x + k[1]·x = (k[-1]+k[0]+k[1])·x
                    out[base] = (kernel[0] + kernel[1] + kernel[2]) * data[base];
                    continue;
                }

                // Boundary: i = 0  →  i-1 clamped to 0
                if pos == 0 {
                    let n_prev = data[base]; // clamp(−1) → data[0]
                    let n_curr = data[base];
                    let n_next = data[base + stride]; // i+1 = 1
                    out[base] = kernel[0] * n_prev + kernel[1] * n_curr + kernel[2] * n_next;
                    continue;
                }

                // Boundary: i = dim_len − 1  →  i+1 clamped to dim_len − 1
                if pos == dim_len - 1 {
                    let n_prev = data[base - stride]; // i-1
                    let n_curr = data[base];
                    let n_next = data[base]; // clamp(dim_len) → data[dim_len-1]
                    out[base] = kernel[0] * n_prev + kernel[1] * n_curr + kernel[2] * n_next;
                    continue;
                }

                // Interior: i ∈ [1, dim_len−2]  →  i−1 and i+1 in bounds.
                // No conditionals — uniform stride, 3-tap FMA.
                let n_prev = data[base - stride];
                let n_curr = data[base];
                let n_next = data[base + stride];
                out[base] = kernel[0] * n_prev + kernel[1] * n_curr + kernel[2] * n_next;
            }
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_sobel.rs"]
mod tests_sobel;

#[cfg(test)]
mod tests_boundary_interior {
    use super::*;

    /// Naive (unsplit) 1-D convolution — the original combined-loop logic.
    fn convolve_1d_axis_naive(
        data: &[f32],
        dims: [usize; 3],
        axis: usize,
        kernel: &[f32; 3],
    ) -> Vec<f32> {
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
                        let offset = ki as isize - 1;
                        let neighbor =
                            (pos as isize + offset).clamp(0, dim_len as isize - 1) as usize;
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

    /// Differential test: boundary/interior split matches naive reference
    /// for all 3 axes, multiple sizes, both derivative and smoothing kernels.
    #[test]
    fn test_convolve_split_matches_naive() {
        let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
        let smooth: [f32; 3] = [1.0, 2.0, 1.0];

        for &dims in &[
            [4, 4, 4],
            [3, 5, 7],
            [1, 1, 16],
            [8, 1, 1],
            [1, 10, 1],
            [2, 2, 2],
        ] {
            let n = dims[0] * dims[1] * dims[2];
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            for axis in 0..3 {
                for kernel in [&deriv, &smooth] {
                    let out_split = convolve_1d_axis(&data, dims, axis, kernel);
                    let out_naive = convolve_1d_axis_naive(&data, dims, axis, kernel);
                    for i in 0..n {
                        assert!(
                            (out_split[i] - out_naive[i]).abs() < 1e-6,
                            "convolve mismatch: dims={dims:?} axis={axis} kernel={:?} \
                             idx={i} split={} naive={}",
                            kernel,
                            out_split[i],
                            out_naive[i]
                        );
                    }
                }
            }
        }
    }

    /// Single-element axis: convolution with replicate padding clamps all 3 taps
    /// to the same value, so the output is (k[-1]+k[0]+k[1])·x.
    /// For derivative [-1,0,1]: sum = 0 → output = 0.
    /// For smoothing [1,2,1]: sum = 4 → output = 4·x.
    #[test]
    fn test_convolve_single_element_axis() {
        let dims = [1, 1, 1];
        let data = vec![42.0];
        let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
        let smooth: [f32; 3] = [1.0, 2.0, 1.0];
        let out_d = convolve_1d_axis(&data, dims, 2, &deriv);
        let out_s = convolve_1d_axis(&data, dims, 2, &smooth);
        assert_eq!(out_d[0], 0.0, "derivative of single element must be zero");
        assert_eq!(
            out_s[0],
            42.0 * 4.0,
            "smoothing of single element = kernel_sum * x = 4 * 42 = 168"
        );
    }
}

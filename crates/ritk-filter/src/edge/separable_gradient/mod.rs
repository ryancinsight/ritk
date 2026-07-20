//! Shared infrastructure for 3-D separable gradient filters (Sobel, Prewitt).
//!
//! # Architecture
//!
//! Both Sobel and Prewitt decompose their 3×3×3 kernels into three sequential
//! 1-D convolutions: one derivative pass and two smoothing passes. The only
//! difference is the smoothing kernel and normalization factor.
//!
//! This module provides:
//! - [`GradientKernel`] — a ZST trait encoding the smoothing kernel and
//!   normalization factor at the type level.
//! - [`SeparableGradientFilter<K>`] — a generic filter struct parameterized
//!   by the kernel ZST, monomorphizing to zero-cost Sobel/Prewitt variants.
//! - [`convolve_1d_axis`] — the canonical 3-tap 1-D convolution with
//!   replicate padding, shared by all gradient filters.
//!
//! # SIMD boundary/interior split
//!
//! The 1-D convolution kernel is split into:
//! 1. **Boundary pass** — processes the first and last voxel of each 1-D
//!    line where the -1 or +1 neighbor index would go out of bounds.
//!    Uses clamped indexing (conditionals).
//! 2. **Interior pass** — processes all remaining voxels with known-in-bounds
//!    neighbor access at uniform stride. No conditionals per iteration,
//!    enabling LLVM auto-vectorization of the FMA chain.

mod kernel;
pub use kernel::{GradientKernel, PrewittKernel, SobelKernel};

use moirai;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec, rebuild};

type GradientImages<B> = (Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>);

// ── Generic filter ────────────────────────────────────────────────────────────

/// 3-D separable gradient filter parameterized by smoothing kernel type.
///
/// Computes spatial derivatives via the outer product of a derivative kernel
/// `d = [-1, 0, 1]` and the smoothing kernel defined by `K::SMOOTH`,
/// applied as three sequential 1-D convolutions with replicate boundary
/// padding. Output is normalized to physical gradient units.
///
/// # Invariants
///
/// - Interior voxels receive second-order central-difference gradient estimates.
/// - Output shape equals input shape.
/// - For a linear field `I = c·x_a`, the interior component along axis `a`
///   equals `c`, and all orthogonal components equal zero.
#[derive(Debug, Clone)]
pub struct SeparableGradientFilter<K: GradientKernel> {
    /// Physical voxel spacing [sz, sy, sx].
    pub spacing: Spacing<3>,
    _kernel: core::marker::PhantomData<K>,
}

impl<K: GradientKernel> SeparableGradientFilter<K> {
    /// Create a filter with the given physical spacing [sz, sy, sx].
    pub fn new(spacing: Spacing<3>) -> Self {
        Self {
            spacing,
            _kernel: core::marker::PhantomData,
        }
    }

    /// Create a filter with unit spacing [1.0, 1.0, 1.0].
    pub fn unit() -> Self {
        Self {
            spacing: Spacing::uniform(1.0),
            _kernel: core::marker::PhantomData,
        }
    }

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I| = √(G_z² + G_y² + G_x²).
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let mag = gradient_magnitude_vec::<K>(&vals, dims, &self.spacing);
        Ok(rebuild(mag, dims, image))
    }

    /// Coeus-native sister of [`SeparableGradientFilter::apply`].
    ///
    /// Runs the identical separable-convolution gradient magnitude (replicate
    /// boundary) via the shared `gradient_magnitude_vec` host core on the
    /// image's contiguous host buffer, so the result is bitwise-identical to the
    /// Coeus path. No tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<BC>(
        &self,
        image: &ritk_image::Image<f32, BC, 3>,
    ) -> anyhow::Result<ritk_image::Image<f32, BC, 3>>
    where
        BC: coeus_core::ComputeBackend + Default,
        BC::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let mag = gradient_magnitude_vec::<K>(&vals, dims, &self.spacing);
        ritk_tensor_ops::native::rebuild_image(mag, dims, image, &BC::default())
    }

    /// Compute the three gradient component images.
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape
    /// and physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<f32, B, 3>,
    ) -> anyhow::Result<GradientImages<B>> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = gradient_components::<K>(&vals, dims, &self.spacing);
        Ok((
            rebuild(gz, dims, image),
            rebuild(gy, dims, image),
            rebuild(gx, dims, image),
        ))
    }
}

// ── Component computation ─────────────────────────────────────────────────────

/// Derivative kernel shared by all separable gradient filters.
const DERIV: [f32; 3] = [-1.0, 0.0, 1.0];

/// Compute gradient components (gz, gy, gx) via separable 1-D convolutions.
///
/// For each component:
/// 1. Apply derivative kernel `[-1, 0, 1]` along the target axis.
/// 2. Apply `K::SMOOTH` along each orthogonal axis.
/// 3. Normalize by `K::NORM_BASE · h_axis`.
///
/// Boundary handling: replicate (clamp) padding.
fn gradient_components<K: GradientKernel>(
    data: &[f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let smooth = K::SMOOTH;

    // G_z: derivative along z (axis 0), smooth along y (axis 1) and x (axis 2).
    let gz = {
        let tmp = convolve_1d_axis(data, dims, 0, &DERIV);
        let tmp = convolve_1d_axis(&tmp, dims, 1, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        normalize_vec(raw, K::NORM_BASE * spacing[0] as f32)
    };

    // G_y: derivative along y (axis 1), smooth along z (axis 0) and x (axis 2).
    let gy = {
        let tmp = convolve_1d_axis(data, dims, 1, &DERIV);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 2, &smooth);
        normalize_vec(raw, K::NORM_BASE * spacing[1] as f32)
    };

    // G_x: derivative along x (axis 2), smooth along z (axis 0) and y (axis 1).
    let gx = {
        let tmp = convolve_1d_axis(data, dims, 2, &DERIV);
        let tmp = convolve_1d_axis(&tmp, dims, 0, &smooth);
        let raw = convolve_1d_axis(&tmp, dims, 1, &smooth);
        normalize_vec(raw, K::NORM_BASE * spacing[2] as f32)
    };

    (gz, gy, gx)
}

/// Gradient magnitude `√(gz² + gy² + gx²)` from the separable components.
///
/// Single host realization shared by the Coeus [`SeparableGradientFilter::apply`]
/// path and the Coeus-native [`SeparableGradientFilter::apply_native`] path.
pub(crate) fn gradient_magnitude_vec<K: GradientKernel>(
    data: &[f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
) -> Vec<f32> {
    let (gz, gy, gx) = gradient_components::<K>(data, dims, spacing);
    gz.iter()
        .zip(gy.iter())
        .zip(gx.iter())
        .map(|((&z, &y), &x)| (z * z + y * y + x * x).sqrt())
        .collect()
}

/// Divide every element of `v` by `norm`.
#[inline]
fn normalize_vec(v: Vec<f32>, norm: f32) -> Vec<f32> {
    let inv = 1.0 / norm;
    v.into_iter().map(|x| x * inv).collect()
}

// ── 1-D convolution ──────────────────────────────────────────────────────────

/// Apply a 3-tap 1-D convolution along the specified axis with replicate padding.
///
/// `axis`: 0 = z, 1 = y, 2 = x.
/// `kernel`: 3-element filter `[k_{-1}, k_0, k_{+1}]`.
///
/// Boundary indices are clamped to `[0, dim_size − 1]` (replicate padding).
///
/// # Boundary/interior split
///
/// The loop over each 1-D line is split into:
/// 1. **Boundary pass** — `i=0` (clamped `i−1`) and `i=len−1` (clamped `i+1`).
///    These use conditional neighbor-index clamping.
/// 2. **Interior pass** — `i=1..len−2`, where both `i−1` and `i+1` are
///    guaranteed in-bounds. No conditionals, uniform stride.
///    LLVM can auto-vectorize the 3-tap FMA body.
///
/// # Complexity
///
/// O(N) where N = nz × ny × nx. Each voxel performs exactly 3 multiply-adds.
pub fn convolve_1d_axis(
    data: &[f32],
    dims: [usize; 3],
    axis: usize,
    kernel: &[f32; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let stride: usize = match axis {
        0 => ny * nx,
        1 => nx,
        2 => 1,
        _ => unreachable!(),
    };
    let dim_len = dims[axis];

    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |base| {
        let iz = base / (ny * nx);
        let iy = (base / nx) % ny;
        let ix = base % nx;
        let pos = match axis {
            0 => iz,
            1 => iy,
            2 => ix,
            _ => unreachable!(),
        };
        if dim_len <= 1 {
            return (kernel[0] + kernel[1] + kernel[2]) * data[base];
        }
        if pos == 0 {
            kernel[0] * data[base] + kernel[1] * data[base] + kernel[2] * data[base + stride]
        } else if pos == dim_len - 1 {
            kernel[0] * data[base - stride] + kernel[1] * data[base] + kernel[2] * data[base]
        } else {
            kernel[0] * data[base - stride]
                + kernel[1] * data[base]
                + kernel[2] * data[base + stride]
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_separable_gradient;

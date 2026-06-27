//! Shared pixel-buffer I/O helpers for tensor-backed RITK image operations.
//!
//! # Design Rationale (SSOT / DRY)
//!
//! Every scalar-path filter and statistics routine must perform the same two operations:
//!
//! 1. **Extract** voxel data from tensor-backed images into contiguous host
//!    memory for CPU-side iterator kernels.
//! 2. **Rebuild** outputs from modified buffers while preserving the caller's
//!    dimensional contract and spatial metadata where applicable.
//!
//! Prior to this module, both helpers were defined independently in 10–13 leaf
//! filter files (e.g. `arithmetic.rs`, `rescale.rs`, `bilateral.rs`, `sobel.rs`).
//! Those private copies are identical in semantics; they differed only in their
//! error message strings.
//!
//! This module is the **single authoritative implementation**. Leaf filters must
//! `use ritk_tensor_ops::{extract_vec, rebuild}` and delete their local copies.
//!
//! # Genericity
//!
//! Both functions are generic over `const D: usize` so they work for 2-D, 3-D,
//! and future N-D images without additional copies. The `rebuild` function
//! additionally accepts a `const D: usize` parameter so the returned image has
//! the correct static dimensionality.
//!
//! # Performance notes
//!
//! The legacy `Image<B, D>` helpers cross the Burn tensor ↔ CPU memory boundary
//! and are therefore inherently O(N) in both time and space. The `coeus` feature
//! adds borrowed extraction for contiguous Coeus tensors and Coeus-backed images
//! so read-only kernels can avoid a copy.

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use num_traits::Float;
use ritk_image::Image;
use std::ops::AddAssign;

#[cfg(feature = "coeus")]
pub mod coeus;

// ── extract_vec ───────────────────────────────────────────────────────────────

/// Copy voxel data out of a `D`-dimensional image into a flat `Vec<f32>`.
///
/// Returns `(voxels, shape)` where `shape` is `[usize; D]`.
///
/// # Errors
/// Returns `Err` when the backend element type cannot be cast to `f32` (e.g. a
/// non-float backend was used).
///
/// # Invariants
/// - `voxels.len() == shape[0] * shape[1] * … * shape[D-1]`
/// - The flat layout is row-major (C-order), matching Burn's default memory
///   layout.
///
/// # Example
/// ```ignore
/// let (vals, dims) = extract_vec(&image)?;
/// let result: Vec<f32> = vals.iter().map(|&v| v.abs()).collect();
/// Ok(rebuild(result, dims, &image))
/// ```
#[inline]
pub fn extract_vec<B: Backend, const D: usize>(
    image: &Image<B, D>,
) -> anyhow::Result<(Vec<f32>, [usize; D])> {
    let dims = image.shape();
    let vals = image
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("filter ops: cannot convert tensor to f32 Vec: {:?}", e))?;
    Ok((vals, dims))
}

// ── extract_vec_infallible ────────────────────────────────────────────────────

/// Infallible variant of [`extract_vec`] for backends that are guaranteed to
/// carry `f32` data (e.g. `NdArray<f32>`, `Wgpu<f32>`).
///
/// Panics with a clear message when the conversion fails. Only use this variant
/// inside arithmetic filters where the public API documents that `f32` is
/// required and returning `Err` would be an internal invariant violation.
#[inline]
pub fn extract_vec_infallible<B: Backend, const D: usize>(
    image: &Image<B, D>,
) -> (Vec<f32>, [usize; D]) {
    let dims = image.shape();
    let vals = image
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("filter ops: extract_vec_infallible requires an f32 backend tensor");
    (vals, dims)
}

// ── rebuild ───────────────────────────────────────────────────────────────────

/// Construct a new `Image<B, D>` from a flat voxel buffer, reusing the spatial
/// metadata of `src`.
///
/// # Arguments
/// - `vals`  — Flat voxel values in row-major order; length must equal the
///   product of `dims`.
/// - `dims`  — Output shape `[usize; D]`.
/// - `src`   — Reference image from which origin, spacing, and direction are
///   copied.
///
/// # Panics
/// Does not panic; the only possible failure is a Burn backend OOM, which is
/// unrecoverable by the filter layer.
///
/// # Invariant
/// `vals.len() == dims[0] * dims[1] * … * dims[D-1]`
#[inline]
fn build_tensor<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    let td = TensorData::new(vals, Shape::new(dims));
    Tensor::<B, D>::from_data(td, device)
}

#[inline]
pub fn rebuild<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let tensor = build_tensor::<B, D>(vals, dims, &device);
    Image::new(tensor, *src.origin(), *src.spacing(), *src.direction())
}

#[inline]
pub fn rebuild_with_origin<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    new_origin: ritk_spatial::Point<D>,
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let tensor = build_tensor::<B, D>(vals, dims, &device);
    Image::new(tensor, new_origin, *src.spacing(), *src.direction())
}

#[inline]
pub fn rebuild_with_metadata<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    new_origin: ritk_spatial::Point<D>,
    new_spacing: ritk_spatial::Spacing<D>,
    new_direction: ritk_spatial::Direction<D>,
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let tensor = build_tensor::<B, D>(vals, dims, &device);
    Image::new(tensor, new_origin, new_spacing, new_direction)
}

// ── gaussian_kernel ─────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of type `T`.
///
/// The kernel is symmetric, centred, and sums to `T::one()` (probability-
/// preserving convolution). If `sigma <= T::zero()`, returns `vec![T::one()]`
/// (identity kernel). When `radius` is `None`, the radius defaults to
/// `⌈3σ⌉`, which captures >99.7% of the Gaussian mass.
///
/// # Formula
///
/// ```text
/// w_i = exp(−d² / (2σ²)) / Z,   d = i − radius,   Z = Σ w_i
/// ```
///
/// # Invariants
///
/// - `kernel.len() == 2 * radius + 1`
/// - `kernel[radius]` is the peak value
/// - `kernel[i] == kernel[len − 1 − i]` (symmetry)
/// - `Σ kernel[i] == 1.0` within floating-point rounding
///
/// # Type parameters
///
/// `T: Float + AddAssign + Default` — supports `f32` and `f64`. Monomorphisation
/// emits zero-cost specialisations identical to hand-written concrete versions.
///
/// # Evidence tier
///
/// Property-tested: normalisation, symmetry, peak-at-centre, length.
pub fn gaussian_kernel<T>(sigma: T, radius: Option<usize>) -> Vec<T>
where
    T: Float + AddAssign + Default,
{
    let zero = T::zero();
    let one = T::one();

    if sigma <= zero {
        return vec![one];
    }

    let r = radius.unwrap_or_else(|| (T::from(3.0).unwrap() * sigma).ceil().to_usize().unwrap());
    // SAFETY: 2.0 is exactly representable in every IEEE-754 float type.
    let two_sigma2 = T::from(2.0).unwrap() * sigma * sigma; // 2σ²
    let len = 2 * r + 1;

    let mut kernel = Vec::with_capacity(len);
    let mut sum = zero;
    for i in 0..len {
        let d = T::from(i).unwrap() - T::from(r).unwrap();
        let w = (-d * d / two_sigma2).exp();
        kernel.push(w);
        sum += w;
    }

    let inv_sum = one / sum;
    for w in &mut kernel {
        *w = *w * inv_sum;
    }

    kernel
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_tensor_ops.rs"]
mod tests;

#[cfg(test)]
mod tests_coeus;

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

use eunomia::FloatElement;
use ritk_image::tensor::{ComputeBackend, Scalar};
use ritk_image::Image;
use std::ops::{AddAssign, Neg};

pub mod native;

// ── extract_vec ───────────────────────────────────────────────────────────────

/// Copy voxel data out of a `D`-dimensional image into a flat `Vec<T>`.
///
/// Returns `(voxels, shape)` where `shape` is `[usize; D]`.
///
/// # Errors
/// Returns `Err` when the backend element type cannot be accessed (e.g. a
/// non-CPU-addressable backend was used without transfer).
///
/// # Invariants
/// - `voxels.len() == shape[0] * shape[1] * … * shape[D-1]`
/// - The flat layout is row-major (C-order).
///
/// # Example
/// ```ignore
/// let (vals, dims) = extract_vec(&image)?;
/// let result: Vec<f32> = vals.iter().map(|&v| v.abs()).collect();
/// Ok(rebuild(result, dims, &image))
/// ```
#[inline]
pub fn extract_vec<T, B, const D: usize>(
    image: &Image<T, B, D>,
) -> anyhow::Result<(Vec<T>, [usize; D])>
where
    T: Scalar + std::fmt::Debug + Clone,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let dims = image.shape();
    let vals = image.try_data_vec()?;
    Ok((vals, dims))
}

// ── extract_vec_infallible ────────────────────────────────────────────────────

/// Infallible variant of [`extract_vec`] for backends that are guaranteed to
/// carry the expected scalar type.
///
/// Panics with a clear message when the conversion fails.
#[inline]
pub fn extract_vec_infallible<T, B, const D: usize>(
    image: &Image<T, B, D>,
) -> (Vec<T>, [usize; D])
where
    T: Scalar + std::fmt::Debug + Clone,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    let dims = image.shape();
    let vals = image
        .try_data_vec()
        .expect("filter ops: extract_vec_infallible requires a CPU-addressable tensor");
    (vals, dims)
}

// ── rebuild ───────────────────────────────────────────────────────────────────

/// Construct a new `Image<T, B, D>` from a flat voxel buffer, reusing the
/// spatial metadata of `src`.
///
/// # Arguments
/// - `vals`  — Flat voxel values in row-major order; length must equal the
///   product of `dims`.
/// - `dims`  — Output shape `[usize; D]`.
/// - `src`   — Reference image from which origin, spacing, and direction are
///   copied.
///
/// # Invariant
/// `vals.len() == dims[0] * dims[1] * … * dims[D-1]`
#[inline]
pub fn rebuild<T, B, const D: usize>(
    vals: Vec<T>,
    dims: [usize; D],
    src: &Image<T, B, D>,
) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    Image::from_flat_on(
        vals,
        dims,
        *src.origin(),
        *src.spacing(),
        *src.direction(),
        &backend,
    )
}

#[inline]
pub fn rebuild_with_origin<T, B, const D: usize>(
    vals: Vec<T>,
    dims: [usize; D],
    new_origin: ritk_spatial::Point<D>,
    src: &Image<T, B, D>,
) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    Image::from_flat_on(
        vals,
        dims,
        new_origin,
        *src.spacing(),
        *src.direction(),
        &backend,
    )
}

#[inline]
pub fn rebuild_with_metadata<T, B, const D: usize>(
    vals: Vec<T>,
    dims: [usize; D],
    new_origin: ritk_spatial::Point<D>,
    new_spacing: ritk_spatial::Spacing<D>,
    new_direction: ritk_spatial::Direction<D>,
    src: &Image<T, B, D>,
) -> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    let backend = B::default();
    Image::from_flat_on(vals, dims, new_origin, new_spacing, new_direction, &backend)
}

// ── gaussian_kernel ─────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of type `T`.
///
/// The kernel is symmetric, centred, and sums to `T::ONE` (probability-
/// preserving convolution). If `sigma <= T::ZERO`, returns `vec![T::ONE]`
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
/// `T: FloatElement + AddAssign + Neg + Default` — supports `f32` and `f64`. Monomorphisation
/// emits zero-cost specialisations identical to hand-written concrete versions.
///
/// # Evidence tier
///
/// Property-tested: normalisation, symmetry, peak-at-centre, length.
pub fn gaussian_kernel<T>(sigma: T, radius: Option<usize>) -> Vec<T>
where
    T: FloatElement + AddAssign + Neg<Output = T> + Default,
{
    let zero = T::ZERO;
    let one = T::ONE;

    if sigma <= zero {
        return vec![one];
    }

    let r = radius.unwrap_or_else(|| (T::from_f64(3.0) * sigma).ceil().to_f64() as usize);
    // SAFETY: 2.0 is exactly representable in every IEEE-754 float type.
    let two_sigma2 = T::from_f64(2.0) * sigma * sigma; // 2σ²
    let len = 2 * r + 1;

    let mut kernel = Vec::with_capacity(len);
    let mut sum = zero;
    for i in 0..len {
        let d = T::from_f64(i as f64) - T::from_f64(r as f64);
        let w = (-d * d / two_sigma2).exp();
        kernel.push(w);
        sum += w;
    }

    let inv_sum = one / sum;
    for w in &mut kernel {
        *w *= inv_sum;
    }

    kernel
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_tensor_ops.rs"]
mod tests;

#[cfg(test)]
mod tests_native;

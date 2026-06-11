//! Shared pixel-buffer I/O helpers for filter implementations.
//!
//! # Design Rationale (SSOT / DRY)
//!
//! Every scalar-path filter in `ritk-core` must perform the same two operations:
//!
//! 1. **Extract** the voxel data from a `Tensor<B, D>` into a contiguous `Vec<f32>`
//!    that can be processed by CPU-side iterators.
//! 2. **Rebuild** the output `Image<B, D>` from the modified `Vec<f32>`, preserving
//!    all spatial metadata (origin, spacing, direction) from the source image.
//!
//! Prior to this module, both helpers were defined independently in 10–13 leaf
//! filter files (e.g. `arithmetic.rs`, `rescale.rs`, `bilateral.rs`, `sobel.rs`).
//! Those private copies are identical in semantics; they differed only in their
//! error message strings.
//!
//! This module is the **single authoritative implementation**. Leaf filters must
//! `use crate::filter::ops::{extract_vec, rebuild}` and delete their local copies.
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
//! These helpers cross the Burn tensor ↔ CPU memory boundary and are therefore
//! inherently O(N) in both time and space. They are appropriate for scalar-path
//! operations that cannot be expressed as a single Burn kernel dispatch. For
//! purely tensor-native operations (element-wise arithmetic, reductions) callers
//! must prefer `Tensor::map` / `Tensor::elementwise_*` to avoid the round-trip.

use ritk_image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use num_traits::Float;
use std::ops::AddAssign;

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
pub fn rebuild<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, D>::from_data(td, &device);
    Image::new(tensor, *src.origin(), *src.spacing(), *src.direction())
}

#[inline]
pub fn rebuild_with_origin<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    new_origin: crate::spatial::Point<D>,
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, D>::from_data(td, &device);
    Image::new(tensor, new_origin, *src.spacing(), *src.direction())
}

#[inline]
pub fn rebuild_with_metadata<B: Backend, const D: usize>(
    vals: Vec<f32>,
    dims: [usize; D],
    new_origin: crate::spatial::Point<D>,
    new_spacing: crate::spatial::Spacing<D>,
    new_direction: crate::spatial::Direction<D>,
    src: &Image<B, D>,
) -> Image<B, D> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, D>::from_data(td, &device);
    Image::new(tensor, new_origin, new_spacing, new_direction)
}

// ── gaussian_kernel_1d ──────────────────────────────────────────────────────

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
/// Property-tested: normalisation, symmetry, peak-at-centre, length (see tests
/// in `filter::ops`, `level_set::helpers`, `level_set::geodesic_active_contour`).
pub fn gaussian_kernel_1d<T>(sigma: T, radius: Option<usize>) -> Vec<T>
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image_3d(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        use burn::tensor::Shape;
        let device = Default::default();
        let t = burn::tensor::Tensor::<B, 3>::from_data(
            TensorData::new(data, Shape::new(shape)),
            &device,
        );
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── extract_vec ────────────────────────────────────────────────────────

    /// Round-trip: extract_vec then rebuild must reproduce the original image.
    ///
    /// # Derivation
    /// extract_vec(I) = (v, d)  and  rebuild(v, d, I) = I' must satisfy:
    ///   ∀ i: I'(i) = I(i)    (element-wise equality within f32 precision)
    ///   shape(I') = shape(I)
    #[test]
    fn extract_and_rebuild_roundtrip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        let img = make_image_3d(data.clone(), [2, 3, 4]);

        let (vals, dims) = extract_vec(&img).unwrap();
        assert_eq!(vals, data, "extracted values must equal original data");
        assert_eq!(dims, [2, 3, 4], "extracted dims must match image shape");

        let rebuilt = rebuild(vals, dims, &img);
        let got = rebuilt
            .data()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        assert_eq!(got, data, "rebuilt image must reproduce original data");
    }

    /// Spatial metadata is preserved through extract → rebuild.
    #[test]
    fn rebuild_preserves_metadata() {
        let sp = Spacing::new([2.5, 1.0, 0.5]);
        let orig = Point::new([10.0, 20.0, 30.0]);
        let device = Default::default();
        let t = burn::tensor::Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 6], burn::tensor::Shape::new([1usize, 2, 3])),
            &device,
        );
        let img = Image::new(t, orig, sp, Direction::identity());

        let (vals, dims) = extract_vec(&img).unwrap();
        let rebuilt = rebuild(vals, dims, &img);

        assert_eq!(rebuilt.origin(), img.origin(), "origin must be preserved");
        assert_eq!(
            rebuilt.spacing(),
            img.spacing(),
            "spacing must be preserved"
        );
    }

    /// extract_vec_infallible produces the same result as extract_vec.
    #[test]
    fn extract_vec_infallible_matches_fallible() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = make_image_3d(data.clone(), [2, 2, 2]);

        let (v1, d1) = extract_vec(&img).unwrap();
        let (v2, d2) = extract_vec_infallible(&img);

        assert_eq!(v1, v2, "infallible variant must return same values");
        assert_eq!(d1, d2, "infallible variant must return same dims");
    }

    /// rebuild constructs an image with the expected shape.
    #[test]
    fn rebuild_output_has_correct_shape() {
        let data = vec![1.0_f32; 3 * 4 * 5];
        let img = make_image_3d(data.clone(), [3, 4, 5]);
        let (vals, dims) = extract_vec(&img).unwrap();
        let out = rebuild(vals, dims, &img);
        assert_eq!(
            out.shape(),
            img.shape(),
            "rebuilt shape must match source shape"
        );
    }

    // ── gaussian_kernel_1d ──────────────────────────────────────────────

    /// Kernel sums to 1.0 (f64 variant).
    #[test]
    fn gaussian_kernel_1d_f64_sums_to_one() {
        let kernel = super::gaussian_kernel_1d(2.0_f64, None);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
    }

    /// Kernel sums to 1.0 (f32 variant).
    #[test]
    fn gaussian_kernel_1d_f32_sums_to_one() {
        let kernel = super::gaussian_kernel_1d(2.0_f32, None);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    /// Zero sigma returns identity kernel.
    #[test]
    fn gaussian_kernel_1d_zero_sigma_is_identity() {
        let kernel = super::gaussian_kernel_1d(0.0_f64, None);
        assert_eq!(kernel, vec![1.0_f64]);
    }

    /// Explicit radius overrides the default.
    #[test]
    fn gaussian_kernel_1d_explicit_radius() {
        let kernel = super::gaussian_kernel_1d(1.0_f64, Some(5));
        assert_eq!(kernel.len(), 11); // 2 * 5 + 1
    }

    /// Centre-to-adjacent ratio verifies the exponent denominator is exactly 2σ².
    ///
    /// # Derivation
    /// For d=1 from centre: w₁/w₀ = exp(-1 / (2σ²)).
    /// With σ=2.0: expected = exp(-1/8) ≈ 0.882497.
    /// The previous defect (`1 + σ²` = 5) would produce exp(-1/5) ≈ 0.818731 — a ~7% error.
    #[test]
    fn gaussian_kernel_1d_exponent_denominator_is_two_sigma_squared() {
        let sigma = 2.0_f64;
        let kernel = super::gaussian_kernel_1d(sigma, Some(4));
        let centre = 4_usize; // r = 4, centre = index 4
        let expected_ratio = (-1.0_f64 / (2.0 * sigma * sigma)).exp();
        let actual_ratio = kernel[centre - 1] / kernel[centre];
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-12,
            "ratio kernel[r-1]/kernel[r] = {actual_ratio:.9}, expected exp(-1/(2σ²)) = {expected_ratio:.9}"
        );
    }

    /// Kernel is symmetric.
    #[test]
    fn gaussian_kernel_1d_is_symmetric() {
        let kernel = super::gaussian_kernel_1d(2.0_f64, None);
        let n = kernel.len();
        for i in 0..n {
            assert!(
                (kernel[i] - kernel[n - 1 - i]).abs() < 1e-15,
                "asymmetry at i={i}: {} vs {}",
                kernel[i],
                kernel[n - 1 - i]
            );
        }
    }

    /// Peak is at the centre.
    #[test]
    fn gaussian_kernel_1d_peak_at_centre() {
        let kernel = super::gaussian_kernel_1d(1.0_f64, Some(3));
        let center = kernel.len() / 2;
        for (i, &w) in kernel.iter().enumerate() {
            if i != center {
                assert!(
                    kernel[center] >= w,
                    "centre {} < kernel[{i}] = {w}",
                    kernel[center]
                );
            }
        }
    }
}

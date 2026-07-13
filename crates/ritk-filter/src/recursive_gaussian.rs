//! Recursive Gaussian filter using the Deriche 4th-order IIR approximation
//! (matching ITK `RecursiveGaussianImageFilter` / SimpleITK
//! `SmoothingRecursiveGaussian`).
//!
//! # Mathematical Specification
//!
//! Implements a separable 4th-order recursive (IIR) Gaussian filter via the
//! Deriche (1993) approximation. For each dimension the 1-D Gaussian is the SUM
//! of a causal (forward) and an anticausal (backward) pass over the same input:
//!
//! Causal:     y_c\[n\] = Σ_{k=0..3} n_k·x\[n−k\] − Σ_{k=1..4} d_k·y_c\[n−k\]
//! Anticausal: y_a\[n\] = Σ_{k=1..4} m_k·x\[n+k\] − Σ_{k=1..4} d_k·y_a\[n+k\]
//! Output:     y\[n\] = y_c\[n\] + y_a\[n\]
//!
//! The coefficients (`n_k`, `d_k`, `m_k`) depend only on σ in pixel units and
//! are DC-normalised so the smoothing has unit gain (see `iir::DericheCoefficients`).
//! The interior is float-exact to SimpleITK; boundaries use constant (replicate)
//! extension.
//!
//! Derivative orders use the ITK/SimpleITK separable structure — the
//! corresponding-order Deriche recursion along the differentiated axis and the
//! zero-order (smoothing) recursion along the others, combined:
//!
//! - **Order 0 (smoothing)**: Two-pass IIR as described above.
//! - **Order 1 (gradient magnitude)**: for each axis `d`, first-order Deriche
//!   along `d` and zero-order along the others; `|∇I| = √(Σ_d (∂I/∂x_d / s_d)²)`.
//!   Float-exact to `GradientMagnitudeRecursiveGaussian`.
//! - **Order 2 (Laplacian)**: for each axis `d`, second-order Deriche along `d`
//!   and zero-order along the others; `∇²I = Σ_d ∂²I/∂x_d² / s_d²`. Float-exact
//!   to `LaplacianRecursiveGaussian`.
//!
//! Physical spacing is respected: `pixel_sigma = sigma / spacing[dim]`, and the
//! order-`k` term is divided by `spacing[d]^k` for the physical derivative.
//!
//! # Complexity
//!
//! O(N) per dimension where N is the number of voxels along that axis,
//! applied separably across all D dimensions. Total: O(D · N_total).
//!
//! # References
//!
//! - Young, I.T. & van Vliet, L.J. (1995). Recursive implementation of the
//!   Gaussian filter. *Signal Processing* 44(2), pp. 139–151.
//! - van Vliet, L.J., Young, I.T., Verbeek, P.W. (1998). Recursive Gaussian
//!   derivative filters. *Proc. 14th ICPR*, pp. 509–514.

use crate::edge::GaussianSigma;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use serde::{Deserialize, Serialize};

#[path = "iir.rs"]
mod iir;
use iir::*;

// ── Scale normalization enum ─────────────────────────────────────────────────

/// Whether to multiply the output by σ^order for comparable cross-scale magnitudes.
///
/// - `Skip`: no scale normalization (default for `RecursiveGaussianFilter::new`).
/// - `Normalize`: multiply output by σ^order so that responses are comparable
///   across scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScaleNormalization {
    /// No scale normalization.
    #[default]
    Skip,
    /// Multiply output by σ^order for cross-scale comparability.
    Normalize,
}

// ── Derivative order enum ─────────────────────────────────────────────────────

/// Derivative order for the recursive Gaussian filter.
///
/// Selects which derivative of the Gaussian kernel to approximate:
/// - `Zero`: Gaussian smoothing (zeroth derivative).
/// - `First`: First derivative — smoothing + gradient magnitude.
/// - `Second`: Second derivative — smoothing + Laplacian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOrder {
    /// Zeroth derivative — Gaussian smoothing.
    Zero,
    /// First derivative of Gaussian.
    First,
    /// Second derivative of Gaussian.
    Second,
}

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Recursive Gaussian filter using a 3rd-order Young–van Vliet IIR
/// approximation.
///
/// Applies a separable recursive Gaussian (or its first/second derivative)
/// along each spatial dimension of a 3-D image, respecting physical spacing.
#[derive(Debug, Clone)]
pub struct RecursiveGaussianFilter {
    /// Standard deviation of the Gaussian in physical units (mm).
    sigma: GaussianSigma,
    /// Which derivative order to approximate.
    derivative_order: DerivativeOrder,
    /// Scale normalization policy.
    scale_normalization: ScaleNormalization,
}

impl RecursiveGaussianFilter {
    /// Create a new recursive Gaussian filter with the given sigma (physical
    /// units).
    ///
    /// Defaults to smoothing (order 0), no scale normalization.
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma: GaussianSigma::new_unchecked(sigma),
            derivative_order: DerivativeOrder::Zero,
            scale_normalization: ScaleNormalization::Skip,
        }
    }

    /// Set the derivative order.
    pub fn with_derivative_order(mut self, order: DerivativeOrder) -> Self {
        self.derivative_order = order;
        self
    }

    /// Set scale normalization policy.
    pub fn with_scale_normalization(mut self, policy: ScaleNormalization) -> Self {
        self.scale_normalization = policy;
        self
    }

    /// Apply the recursive Gaussian filter to a 3-D image.
    ///
    /// Processing is separable: the 1-D IIR filter is applied along each of
    /// the three spatial axes in sequence.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let spacing = image.spacing();
        let out = self.filter_vals(vals, dims, [spacing[0], spacing[1], spacing[2]]);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`RecursiveGaussianFilter::apply`].
    ///
    /// Runs the identical separable Deriche IIR recursion (order 0/1/2 plus scale
    /// normalization; constant/replicate boundary) via the shared
    /// [`RecursiveGaussianFilter::filter_vals`] host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let spacing = image.spacing();
        let out = self.filter_vals(vals, dims, [spacing[0], spacing[1], spacing[2]]);
        ritk_tensor_ops::native::rebuild_image(out, dims, image, &B::default())
    }

    /// Shared host core: separable Deriche recursion with the configured
    /// derivative order and scale normalization, on a flat z-major buffer.
    ///
    /// Both the Burn [`apply`](Self::apply) and Coeus-native
    /// [`apply_native`](Self::apply_native) paths call this single realization;
    /// order 0 smooths all axes, orders 1 and 2 apply the first/second-order
    /// Deriche recursion along each axis (zero-order along the others) and
    /// combine — magnitude for order 1, sum for order 2.
    fn filter_vals(&self, mut vals: Vec<f32>, dims: [usize; 3], sp: [f64; 3]) -> Vec<f32> {
        let sigma = self.sigma.get();
        vals = match self.derivative_order {
            DerivativeOrder::Zero => {
                for (dim, &s) in sp.iter().enumerate() {
                    let pixel_sigma = sigma / s;
                    if pixel_sigma < 0.2 {
                        continue;
                    }
                    let coeffs = DericheCoefficients::from_sigma(pixel_sigma);
                    vals = apply_deriche_1d(&vals, dims, dim, &coeffs, pixel_sigma);
                }
                vals
            }
            DerivativeOrder::First => gradient_magnitude_rg_vals(&vals, dims, sp, sigma),
            DerivativeOrder::Second => laplacian_rg_vals(&vals, dims, sp, sigma),
        };

        // Scale normalization: multiply by σ^order
        if let ScaleNormalization::Normalize = self.scale_normalization {
            let scale_factor = match self.derivative_order {
                DerivativeOrder::Zero => 1.0,
                DerivativeOrder::First => sigma,
                DerivativeOrder::Second => sigma * sigma,
            };
            if (scale_factor - 1.0).abs() > 1e-12 {
                let sf = scale_factor as f32;
                for v in &mut vals {
                    *v *= sf;
                }
            }
        }

        vals
    }
}
// ── Recursive-Gaussian derivative operators (ITK structure) ───────────────────

/// `∇²(G_σ * I) = Σ_d ∂²/∂x_d²(G_σ * I)` as a flat z-major buffer. For each axis
/// `d` the volume is filtered with the **second-order** Deriche recursion along
/// `d` and the **zero-order** (smoothing) recursion along the others; each
/// per-axis second derivative is divided by `spacing[d]²` (physical units) and
/// summed. Float-exact to ITK/SimpleITK `LaplacianRecursiveGaussian`.
pub(crate) fn laplacian_rg_vals(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    sigma: f64,
) -> Vec<f32> {
    let mut laplacian = vec![0.0f32; vals.len()];
    for d in 0..3 {
        // First axis reads `vals` directly (no upfront clone); each subsequent
        // pass consumes the previous result, so exactly one buffer is live.
        let mut temp: Option<Vec<f32>> = None;
        for (ax, &s) in spacing.iter().enumerate() {
            let pixel_sigma = sigma / s;
            let coeffs = if ax == d {
                DericheCoefficients::second_order(pixel_sigma)
            } else {
                DericheCoefficients::from_sigma(pixel_sigma)
            };
            let src = temp.as_deref().unwrap_or(vals);
            temp = Some(apply_deriche_1d(src, dims, ax, &coeffs, pixel_sigma));
        }
        let temp = temp.expect("invariant: 3 axes always filtered");
        let inv_s2 = (1.0 / (spacing[d] * spacing[d])) as f32;
        for (acc, t) in laplacian.iter_mut().zip(temp.iter()) {
            *acc += t * inv_s2;
        }
    }
    laplacian
}

/// `|∇(G_σ * I)| = √(Σ_d (∂/∂x_d(G_σ * I))²)` as a flat z-major buffer. For each
/// axis `d` the volume is filtered with the **first-order** Deriche recursion
/// along `d` and the **zero-order** recursion along the others; each per-axis
/// first derivative is divided by `spacing[d]` (physical units), squared and
/// accumulated, then the square root is taken. Float-exact to ITK/SimpleITK
/// `GradientMagnitudeRecursiveGaussian`.
pub(crate) fn gradient_magnitude_rg_vals(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    sigma: f64,
) -> Vec<f32> {
    let mut sum_sq = vec![0.0f32; vals.len()];
    for d in 0..3 {
        // First axis reads `vals` directly (no upfront clone).
        let mut temp: Option<Vec<f32>> = None;
        for (ax, &s) in spacing.iter().enumerate() {
            let pixel_sigma = sigma / s;
            let coeffs = if ax == d {
                DericheCoefficients::first_order(pixel_sigma)
            } else {
                DericheCoefficients::from_sigma(pixel_sigma)
            };
            let src = temp.as_deref().unwrap_or(vals);
            temp = Some(apply_deriche_1d(src, dims, ax, &coeffs, pixel_sigma));
        }
        let temp = temp.expect("invariant: 3 axes always filtered");
        let inv_s = (1.0 / spacing[d]) as f32;
        for (acc, t) in sum_sq.iter_mut().zip(temp.iter()) {
            let g = t * inv_s;
            *acc += g * g;
        }
    }
    for v in &mut sum_sq {
        *v = v.sqrt();
    }
    sum_sq
}

/// Compute `∇²(G_σ * I)` matching ITK/SimpleITK `LaplacianRecursiveGaussian`
/// (float-exact). See `laplacian_rg_vals`.
///
/// # Errors
/// Returns `Err` if the tensor data cannot be extracted as `f32`.
pub fn laplacian_recursive_gaussian<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    let sp = image.spacing();
    let out = laplacian_rg_vals(&vals, dims, [sp[0], sp[1], sp[2]], sigma);
    Ok(rebuild(out, dims, image))
}

/// Compute `|∇(G_σ * I)|` matching ITK/SimpleITK `GradientMagnitudeRecursiveGaussian`
/// (float-exact). See `gradient_magnitude_rg_vals`.
///
/// # Errors
/// Returns `Err` if the tensor data cannot be extracted as `f32`.
pub fn gradient_magnitude_recursive_gaussian<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    let sp = image.spacing();
    let out = gradient_magnitude_rg_vals(&vals, dims, [sp[0], sp[1], sp[2]], sigma);
    Ok(rebuild(out, dims, image))
}

/// Single-axis recursive (Deriche) Gaussian or its directional derivative along
/// one axis only (the other axes are untouched) — matching ITK/SimpleITK
/// `RecursiveGaussian(sigma, normalizeAcrossScale=false, order, direction)`
/// (float-exact). `direction` is a ritk axis index (`0 = z, 1 = y, 2 = x`); note
/// SimpleITK's `direction` is in `(x, y, z)` order, so sitk direction `0` (x) is
/// ritk axis `2`. Unlike `LaplacianRecursiveGaussian`, the derivative is NOT
/// divided by the voxel spacing (it matches the raw ITK single-axis filter).
///
/// # Errors
/// Returns `Err` if the data cannot be read as `f32` or `direction > 2`.
pub fn recursive_gaussian_directional<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
    order: DerivativeOrder,
    direction: usize,
) -> anyhow::Result<Image<B, 3>> {
    if direction > 2 {
        anyhow::bail!(
            "recursive_gaussian_directional: direction must be 0, 1, or 2, got {direction}"
        );
    }
    let (vals, dims) = extract_vec(image)?;
    let spacing = image.spacing();
    let pixel_sigma = sigma / spacing[direction];
    let coeffs = match order {
        DerivativeOrder::Zero => DericheCoefficients::from_sigma(pixel_sigma),
        DerivativeOrder::First => DericheCoefficients::first_order(pixel_sigma),
        DerivativeOrder::Second => DericheCoefficients::second_order(pixel_sigma),
    };
    let out = apply_deriche_1d(&vals, dims, direction, &coeffs, pixel_sigma);
    Ok(rebuild(out, dims, image))
}

/// Compute the three first-order recursive-Gaussian gradient components on raw
/// buffers, returned in **ritk axis order** `[∂/∂(axis0=z), ∂/∂(axis1=y),
/// ∂/∂(axis2=x)]`, each divided once by its axis spacing.
///
/// Component `k` is order-0 (smoothing) Deriche along the two non-`k` axes then
/// order-1 (derivative) Deriche along axis `k`. Operating directly on `Vec<f32>`
/// buffers avoids round-tripping each intermediate through an [`Image`] (tensor
/// rebuild) and re-extracting it, so a full vector gradient does one tensor
/// extraction instead of nine — float-identical to chaining
/// [`recursive_gaussian_directional`], but with the per-pass `Image`
/// alloc/rebuild eliminated.
///
/// # Errors
/// Returns `Err` if the tensor data cannot be read as `f32`.
pub fn gradient_recursive_gaussian_components<B: Backend>(
    image: &Image<B, 3>,
    sigma: f64,
) -> anyhow::Result<[Vec<f32>; 3]> {
    let (vals, dims) = extract_vec(image)?;
    let spacing = image.spacing();

    let pass = |buf: &[f32], axis: usize, order: DerivativeOrder| -> Vec<f32> {
        let pixel_sigma = sigma / spacing[axis];
        let coeffs = match order {
            DerivativeOrder::Zero => DericheCoefficients::from_sigma(pixel_sigma),
            DerivativeOrder::First => DericheCoefficients::first_order(pixel_sigma),
            DerivativeOrder::Second => DericheCoefficients::second_order(pixel_sigma),
        };
        apply_deriche_1d(buf, dims, axis, &coeffs, pixel_sigma)
    };

    let component = |axis_k: usize| -> Vec<f32> {
        let others: [usize; 2] = match axis_k {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let mut buf = pass(&vals, others[0], DerivativeOrder::Zero);
        buf = pass(&buf, others[1], DerivativeOrder::Zero);
        buf = pass(&buf, axis_k, DerivativeOrder::First);
        let inv = 1.0_f32 / spacing[axis_k] as f32;
        for v in buf.iter_mut() {
            *v *= inv;
        }
        buf
    };

    Ok([component(0), component(1), component(2)])
}

/// Separable zero-order recursive (Deriche) Gaussian smoothing with a per-axis
/// physical `sigmas[d]` (broadcast from the last element). This is the blur
/// ITK/SimpleITK `UnsharpMask` uses (`SmoothingRecursiveGaussian`), as opposed
/// to the discrete Gaussian; it is float-exact to `SmoothingRecursiveGaussian`.
///
/// # Errors
/// Returns `Err` if the tensor data cannot be extracted as `f32`.
pub fn smoothing_recursive_gaussian<B: Backend>(
    image: &Image<B, 3>,
    sigmas: &[f64],
) -> anyhow::Result<Image<B, 3>> {
    let (mut vals, dims) = extract_vec(image)?;
    let spacing = image.spacing();
    let last = sigmas.last().copied().unwrap_or(0.0);
    for dim in 0..3 {
        let sigma = sigmas.get(dim).copied().unwrap_or(last);
        let pixel_sigma = sigma / spacing[dim];
        if pixel_sigma < 0.2 {
            continue;
        }
        let coeffs = DericheCoefficients::from_sigma(pixel_sigma);
        vals = apply_deriche_1d(&vals, dims, dim, &coeffs, pixel_sigma);
    }
    Ok(rebuild(vals, dims, image))
}

/// Compute all 6 independent Hessian components at every voxel using the
/// Deriche IIR recursion — matching ITK `HessianRecursiveGaussianImageFilter`.
///
/// For each axis `d` (z=0, y=1, x=2):
/// - H_{dd}: `second_order` Deriche along axis `d`, `zero_order` along the
///   other two, divided by `spacing[d]²`.
/// - H_{di} (d<i): `first_order` along `d`, `first_order` along `i`,
///   `zero_order` along the remaining axis, divided by `spacing[d]·spacing[i]`.
///
/// Output layout per voxel: `[Hzz, Hzy, Hzx, Hyy, Hyx, Hxx]` — identical to
/// `compute_hessian` in `vesselness/hessian/mod.rs` so callers are drop-in.
///
/// Evidence tier: matches ITK source `itkHessianRecursiveGaussianImageFilter.hxx`
/// structure; verified against LoG (sum of diagonal = Laplacian via IIR) and
/// differential tests against FD Hessian at σ=3.0 (large σ where both converge).
pub(crate) fn compute_hessian_iir(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    sigma: f64,
) -> Vec<[f32; 6]> {
    let n = vals.len();

    // Helper: apply one IIR pass along `axis` with chosen `order`.
    let pass = |buf: &[f32], axis: usize, order: DerivativeOrder| -> Vec<f32> {
        let pixel_sigma = sigma / spacing[axis];
        let coeffs = match order {
            DerivativeOrder::Zero => DericheCoefficients::from_sigma(pixel_sigma),
            DerivativeOrder::First => DericheCoefficients::first_order(pixel_sigma),
            DerivativeOrder::Second => DericheCoefficients::second_order(pixel_sigma),
        };
        apply_deriche_1d(buf, dims, axis, &coeffs, pixel_sigma)
    };

    // Diagonal H_{dd}: second_order along d, zero_order along others.
    // Divided by spacing[d]² for physical units.
    let hessian_diag = |d: usize| -> Vec<f32> {
        let others: [usize; 2] = match d {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let mut buf = pass(vals, others[0], DerivativeOrder::Zero);
        buf = pass(&buf, others[1], DerivativeOrder::Zero);
        buf = pass(&buf, d, DerivativeOrder::Second);
        let inv = (1.0 / (spacing[d] * spacing[d])) as f32;
        for v in buf.iter_mut() {
            *v *= inv;
        }
        buf
    };

    // Cross H_{di} (d < i): first_order along d, first_order along i, zero_order along k.
    // Divided by spacing[d]·spacing[i] for physical units.
    let hessian_cross = |d: usize, i: usize| -> Vec<f32> {
        // k = the remaining axis; formula works for (0,1)→k=2, (0,2)→k=1, (1,2)→k=0.
        let k = 3 - d - i;
        let mut buf = pass(vals, k, DerivativeOrder::Zero);
        buf = pass(&buf, d, DerivativeOrder::First);
        buf = pass(&buf, i, DerivativeOrder::First);
        let inv = (1.0 / (spacing[d] * spacing[i])) as f32;
        for v in buf.iter_mut() {
            *v *= inv;
        }
        buf
    };

    let hzz = hessian_diag(0);
    let hzy = hessian_cross(0, 1);
    let hzx = hessian_cross(0, 2);
    let hyy = hessian_diag(1);
    let hyx = hessian_cross(1, 2);
    let hxx = hessian_diag(2);

    // Pack into [Hzz, Hzy, Hzx, Hyy, Hyx, Hxx] per voxel.
    let mut out = vec![[0.0_f32; 6]; n];
    for i in 0..n {
        out[i] = [hzz[i], hzy[i], hzx[i], hyy[i], hyx[i], hxx[i]];
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_recursive_gaussian.rs"]
mod tests_recursive_gaussian;

#[cfg(test)]
#[path = "tests_recursive_gaussian_native.rs"]
mod tests_recursive_gaussian_native;

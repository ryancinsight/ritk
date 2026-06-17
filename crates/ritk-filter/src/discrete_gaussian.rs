//! Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
//!
//! # Mathematical Specification
//!
//! Given variance v_d (physical units^2) and voxel spacing h_d, the pixel
//! variance is t = v_d / h_d^2. The kernel is ITK's *discrete* Gaussian
//! (`GaussianOperator`), the discrete analog of the Gaussian (Lindeberg 1990):
//!   g\[k\] = e^{-t} · I_{|k|}(t)   for k in {-r,...,r}
//! where I_n is the modified Bessel function of the first kind. One-sided
//! coefficients are accumulated until the mass g\[0\] + 2·Sum_{i>=1} g\[i\]
//! reaches 1 - maximum_error (radius capped at 32), then normalised by that sum.
//! This is NOT a sampled continuous Gaussian — it is float-exact to SimpleITK.
//!
//! # Boundary Conditions
//! Replicate (edge) padding is used for all convolutions. This preserves the
//! invariant conv(constant_image, normalized_kernel) = constant_image for
//! ALL voxels, including boundaries.
//!
//! # ITK Parity
//!   - Parameterized by variance (sigma^2), not sigma.
//!   - maximum_error truncation criterion (default 0.01).
//!   - use_image_spacing flag (default true).
//!
//! # References
//! - Young et al. (1995). Fundamentals of Image Processing. Delft UT.
//! - ITK Software Guide 2nd Ed., Sec 6.3.1.

use crate::edge::GaussianSigma;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Controls whether Gaussian variance is specified in physical or pixel units.
///
/// - `Physical` (default): variance is in physical (world) units; converted to
///   pixel units using image spacing (`sigma_pixel = sqrt(v) / h_d`).
/// - `Voxel`: variance is already in pixel (voxel) units; no conversion applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpacingMode {
    /// Ignore image spacing; treat variance as voxel-space variance.
    Voxel,
    /// Use physical spacing to convert variance to pixel sigma (ITK default).
    #[default]
    Physical,
}

impl std::fmt::Display for SpacingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpacingMode::Physical => f.write_str("physical"),
            SpacingMode::Voxel => f.write_str("voxel"),
        }
    }
}

impl std::str::FromStr for SpacingMode {
    type Err = String;

    /// Parse a spacing mode string.
    ///
    /// Accepts `"physical"` / `"true"` (legacy) and `"voxel"` / `"pixel"` /
    /// `"false"` (legacy).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "physical" | "true" => Ok(SpacingMode::Physical),
            "voxel" | "pixel" | "false" => Ok(SpacingMode::Voxel),
            _ => Err(format!("expected 'physical' or 'voxel', got '{s}'")),
        }
    }
}

/// Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
///
/// Applies separable 1-D Gaussian convolution along each image dimension.
/// Replicate (edge) padding preserves the uniform-image identity invariant.
pub struct DiscreteGaussianFilter<B: Backend> {
    variance: Vec<f64>,
    maximum_error: f64,
    spacing_mode: SpacingMode,
    _b: PhantomData<fn() -> B>,
}

impl<B: Backend> DiscreteGaussianFilter<B> {
    /// Create with per-dimension Gaussian sigmas (physical units).
    ///
    /// Variance is computed internally as `sigma²` for each dimension.
    /// Panics if `sigmas` is empty.
    pub fn new(sigmas: Vec<GaussianSigma>) -> Self {
        assert!(!sigmas.is_empty(), "sigmas list must not be empty");
        let variance: Vec<f64> = sigmas.iter().map(|s| s.get().powi(2)).collect();
        Self {
            variance,
            maximum_error: 0.01,
            spacing_mode: SpacingMode::Physical,
            _b: PhantomData,
        }
    }

    /// Create a filter with the same variance applied to all image dimensions.
    ///
    /// Equivalent to `new(vec![variance])` when the single variance value is
    /// broadcast to every dimension by `variance_for_dim`.
    pub fn new_isotropic(variance: f64) -> Self {
        assert!(variance >= 0.0, "variance must be >= 0, got {variance}");
        Self {
            variance: vec![variance],
            maximum_error: 0.01,
            spacing_mode: SpacingMode::Physical,
            _b: PhantomData,
        }
    }

    /// Set maximum_error (default 0.01). Panics if not in (0,1).
    pub fn with_maximum_error(mut self, maximum_error: f64) -> Self {
        assert!(
            maximum_error > 0.0 && maximum_error < 1.0,
            "maximum_error must be in (0, 1), got {maximum_error}"
        );
        self.maximum_error = maximum_error;
        self
    }

    /// Set spacing mode (default `Physical`).
    pub fn with_spacing_mode(mut self, mode: SpacingMode) -> Self {
        self.spacing_mode = mode;
        self
    }

    /// Apply the filter; spatial metadata preserved.
    pub fn apply<const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (tensor, origin, spacing, direction) = image.clone().into_parts();
        let smoothed = self.apply_inner(tensor, &spacing);
        Image::new(smoothed, origin, spacing, direction)
    }

    #[inline]
    fn variance_for_dim(&self, d: usize) -> f64 {
        if d < self.variance.len() {
            self.variance[d]
        } else {
            *self
                .variance
                .last()
                .expect("variance schedule must not be empty")
        }
    }

    #[inline]
    fn pixel_sigma_for_dim<const D: usize>(
        &self,
        d: usize,
        spacing: &ritk_spatial::Spacing<D>,
    ) -> f64 {
        let v = self.variance_for_dim(d);
        match self.spacing_mode {
            SpacingMode::Physical => {
                let h = spacing[d];
                v.max(0.0).sqrt() / h.max(1e-12)
            }
            SpacingMode::Voxel => v.max(0.0).sqrt(),
        }
    }

    /// Build ITK's discrete Gaussian kernel (`GaussianOperator`): the symmetric,
    /// normalised coefficients `g[k] = e^{-t}·I_{|k|}(t)` where `t` is the pixel
    /// variance `σ_pixel²` and `I_n` is the modified Bessel function (the discrete
    /// analog of the Gaussian, Lindeberg 1990). One-sided coefficients are
    /// accumulated until the running mass `g[0] + 2·Σ_{i≥1} g[i]` reaches
    /// `1 − maximum_error`, capped at `MAX_KERNEL_RADIUS` (ITK default 32). This
    /// is *not* a sampled continuous Gaussian — it is float-exact to SimpleITK
    /// `DiscreteGaussian`.
    #[inline]
    fn build_kernel(&self, pixel_variance: f64) -> Vec<f32> {
        let t = pixel_variance;
        let et = (-t).exp();
        let cap = 1.0 - self.maximum_error;

        // One-sided coefficients [g0, g1, g2, …].
        let mut coeff = vec![et * modified_bessel_i0(t)];
        let mut sum = coeff[0];
        let g1 = et * modified_bessel_i1(t);
        coeff.push(g1);
        sum += 2.0 * g1;
        let mut i = 2;
        while sum < cap {
            let c = et * modified_bessel_i(i, t);
            if c <= 0.0 {
                break;
            }
            coeff.push(c);
            sum += 2.0 * c;
            if i >= Self::MAX_KERNEL_RADIUS {
                break;
            }
            i += 1;
        }

        // Symmetric, normalised: [g_n, …, g_1, g_0, g_1, …, g_n] / sum.
        let radius = coeff.len() - 1;
        let inv = 1.0 / sum;
        let mut k = Vec::with_capacity(2 * radius + 1);
        k.extend(coeff[1..].iter().rev().map(|&c| (c * inv) as f32));
        k.extend(coeff.iter().map(|&c| (c * inv) as f32));
        k
    }

    /// Minimum pixel variance below which the kernel is the identity impulse.
    const VARIANCE_MIN: f64 = 1e-18;
    /// Maximum one-sided kernel radius (ITK `GaussianOperator` default).
    const MAX_KERNEL_RADIUS: usize = 32;

    #[inline]
    fn apply_inner<const D: usize>(
        &self,
        data: Tensor<B, D>,
        spacing: &ritk_spatial::Spacing<D>,
    ) -> Tensor<B, D> {
        let device = data.device();
        let dims: [usize; D] = data.shape().dims();

        // Build kernels for each dimension (None = skip, identity).
        let mut kernels: [Option<Vec<f32>>; D] = std::array::from_fn(|_| None);
        for (d, kernel_slot) in kernels.iter_mut().enumerate() {
            let sigma = self.pixel_sigma_for_dim::<D>(d, spacing);
            let pixel_variance = sigma * sigma;
            if pixel_variance >= Self::VARIANCE_MIN {
                let kernel = self.build_kernel(pixel_variance);
                // A single-tap kernel is the identity — skip it.
                if kernel.len() > 1 {
                    *kernel_slot = Some(kernel);
                }
            }
        }

        // Check if any dimension needs filtering.
        if kernels.iter().all(|k| k.is_none()) {
            return data;
        }

        // Extract flat Vec<f32>.
        let flat: Vec<f32> = data.into_data().into_vec::<f32>().expect("f32 tensor data");

        // Apply separable convolution on the flat array.
        let result = convolve_separable(flat, dims, &kernels);

        // Reconstruct tensor.
        Tensor::<B, D>::from_data(TensorData::new(result, Shape::new(dims)), &device)
    }
}

// ── Modified Bessel functions (ITK GaussianOperator) ──────────────────────────
// Abramowitz & Stegun 9.8.1–9.8.4 polynomial approximations (the exact forms
// ITK uses), so the discrete Gaussian kernel is float-exact to SimpleITK.

/// Modified Bessel function of the first kind, order 0.
fn modified_bessel_i0(y: f64) -> f64 {
    let d = y.abs();
    if d < 3.75 {
        let m = (y / 3.75) * (y / 3.75);
        1.0 + m
            * (3.5156229
                + m * (3.0899424
                    + m * (1.2067492 + m * (0.2659732 + m * (0.0360768 + m * 0.0045813)))))
    } else {
        let m = 3.75 / d;
        (d.exp() / d.sqrt())
            * (0.39894228
                + m * (0.01328592
                    + m * (0.00225319
                        + m * (-0.00157565
                            + m * (0.00916281
                                + m * (-0.02057706
                                    + m * (0.02635537 + m * (-0.01647633 + m * 0.00392377))))))))
    }
}

/// Modified Bessel function of the first kind, order 1.
fn modified_bessel_i1(y: f64) -> f64 {
    let d = y.abs();
    let acc = if d < 3.75 {
        let m = (y / 3.75) * (y / 3.75);
        d * (0.5
            + m * (0.87890594
                + m * (0.51498869
                    + m * (0.15084934 + m * (0.02658733 + m * (0.00301532 + m * 0.00032411))))))
    } else {
        let m = 3.75 / d;
        let a = 0.02282967 + m * (-0.02895312 + m * (0.01787654 - m * 0.00420059));
        let a = 0.39894228
            + m * (-0.03988024 + m * (-0.00362018 + m * (0.00163801 + m * (-0.01031555 + m * a))));
        a * (d.exp() / d.sqrt())
    };
    if y < 0.0 {
        -acc
    } else {
        acc
    }
}

/// Modified Bessel function of the first kind, order `n ≥ 2`, via Miller's
/// downward recurrence seeded from `j = 2·(n + √(40n))` and renormalised by
/// `I0` (Numerical Recipes / ITK `ModifiedBesselI`).
fn modified_bessel_i(n: usize, y: f64) -> f64 {
    if n == 0 {
        return modified_bessel_i0(y);
    }
    if n == 1 {
        return modified_bessel_i1(y);
    }
    if y == 0.0 {
        return 0.0;
    }
    let tox = 2.0 / y.abs();
    let (mut bip, mut bi, mut ans) = (0.0_f64, 1.0_f64, 0.0_f64);
    // Even starting index well above n; iterate the recurrence downward.
    let mut j = 2 * (n + (40.0 * n as f64).sqrt() as usize);
    while j > 0 {
        let bim = bip + j as f64 * tox * bi;
        bip = bi;
        bi = bim;
        if bi.abs() > 1.0e10 {
            bi *= 1.0e-10;
            bip *= 1.0e-10;
            ans *= 1.0e-10;
        }
        if j == n {
            ans = bip;
        }
        j -= 1;
    }
    ans *= modified_bessel_i0(y) / bi;
    if y < 0.0 && n % 2 == 1 {
        -ans
    } else {
        ans
    }
}

// Flat-array separable convolution for DiscreteGaussianFilter.

/// Convolve a 1-D slice with replicate (edge) boundary padding.
///
/// Reconstructs the output via SAXPY accumulation: for each kernel position kj
/// (offset = kj - r from the kernel center), add w x input[clamped_pos] to the
/// corresponding output elements. Boundary regions are identified analytically to
/// eliminate per-element clamping in the interior, enabling LLVM vectorization.
///
/// # Invariant
/// conv(constant_c, normalised_kernel) = c for all positions.
#[inline]
fn conv1d_replicate(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let n_i = n as isize;
    output.fill(0.0);
    for (kj, &w) in kernel.iter().enumerate() {
        let offset = kj as isize - r; // src_pos = output_i + offset
                                      // i_start: first output index where src_pos >= 0, i.e., i >= -offset.
        let i_start = ((-offset).max(0) as usize).min(n);
        // i_end: first output index where src_pos >= n, i.e., i >= n - offset.
        let i_end = ((n_i - offset).max(0).min(n_i) as usize).min(n);
        // Left boundary [0, i_start): src_pos < 0 - replicate input[0].
        if i_start > 0 {
            let left_val = input[0] * w;
            for o in &mut output[..i_start] {
                *o += left_val;
            }
        }
        // Interior [i_start, i_end): no clamping - sequential reads, LLVM-vectorizable.
        for i in i_start..i_end {
            output[i] += input[(i as isize + offset) as usize] * w;
        }
        // Right boundary [i_end, n): src_pos >= n - replicate input[n-1].
        if i_end < n {
            let right_val = input[n - 1] * w;
            for o in &mut output[i_end..] {
                *o += right_val;
            }
        }
    }
}

/// Convolve flat C-order [NZ,NY,NX] array along one axis. Rayon-parallel.
fn convolve3d_dim(
    src: &[f32],
    dst: &mut [f32],
    nz: usize,
    ny: usize,
    nx: usize,
    dim: usize,
    kernel: &[f32],
) {
    let nyx = ny * nx;
    match dim {
        2 => {
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nx,
                |ci, o| {
                    let i = &src[ci * nx..ci * nx + o.len()];
                    conv1d_replicate(i, kernel, o);
                },
            );
        }
        1 => {
            // Parallel over Z-slabs. Within each slab reorder loops to
            // (kj, iy, ix): reads src[sy*nx..(sy+1)*nx] (contiguous row) and
            // writes os[iy*nx..(iy+1)*nx] (contiguous row) - LLVM-vectorizable.
            // Eliminates the per-slab intermediate buf allocation.
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let ny_i = ny as isize;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nyx,
                |ci, os| {
                    let is = &src[ci * nyx..ci * nyx + os.len()];
                    os.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let r_offset = kj as isize - r;
                        for iy in 0..ny {
                            let sy = ((iy as isize + r_offset).clamp(0, ny_i - 1)) as usize;
                            let src_row = &is[sy * nx..(sy + 1) * nx]; // contiguous
                            let dst_row = &mut os[iy * nx..(iy + 1) * nx]; // contiguous
                            for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                                *d += s * w;
                            }
                        }
                    }
                },
            );
        }
        0 => {
            // Parallel over output Z-slices. For each output slice iz, accumulate
            // contributions from input Z-slices via SAXPY (contiguous reads).
            // Complexity: O(nz x ksz x nyx) with nz-way Rayon parallelism.
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let nz_i = nz as isize;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nyx,
                |iz, out_slice| {
                    out_slice.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let sz = ((iz as isize + kj as isize - r).clamp(0, nz_i - 1)) as usize;
                        let src_z = &src[sz * nyx..(sz + 1) * nyx]; // contiguous Z-slice
                        for (o, &s) in out_slice.iter_mut().zip(src_z.iter()) {
                            *o += s * w;
                        }
                    }
                },
            );
        }
        _ => unreachable!(),
    }
}

fn convolve_nd_dim_serial(
    src: &[f32],
    dst: &mut [f32],
    shape: &[usize],
    dim: usize,
    kernel: &[f32],
) {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    for i in (0..d.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let line_len = shape[dim];
    let line_stride = strides[dim];
    let n_total: usize = shape.iter().product();
    let n_lines = n_total / line_len.max(1);
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let ll_i = line_len as isize;
    let mut idx = vec![0usize; d];
    let mut ob = vec![0.0f32; line_len];
    for _line in 0..n_lines {
        let base: usize = idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        for (i, ob_elem) in ob.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (kj, &w) in kernel.iter().enumerate() {
                let pos = ((i as isize + kj as isize - r).clamp(0, ll_i - 1)) as usize;
                acc += src[base + pos * line_stride] * w;
            }
            *ob_elem = acc;
        }
        for (i, &val) in ob.iter().enumerate() {
            dst[base + i * line_stride] = val;
        }
        let mut carry = true;
        for dd in (0..d).rev() {
            if dd == dim {
                continue;
            }
            if carry {
                idx[dd] += 1;
                if idx[dd] < shape[dd] {
                    carry = false;
                } else {
                    idx[dd] = 0;
                }
            }
        }
    }
}

fn convolve_separable<const D: usize>(
    mut data: Vec<f32>,
    shape: [usize; D],
    kernels: &[Option<Vec<f32>>; D],
) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut buf = vec![0.0f32; n];
    for (dim, kernel_opt) in kernels.iter().enumerate() {
        let Some(kernel) = kernel_opt else {
            continue;
        };
        if D == 3 {
            convolve3d_dim(&data, &mut buf, shape[0], shape[1], shape[2], dim, kernel);
        } else {
            convolve_nd_dim_serial(&data, &mut buf, shape.as_slice(), dim, kernel);
        }
        std::mem::swap(&mut data, &mut buf);
    }
    data
}

#[cfg(test)]
#[path = "tests_discrete_gaussian.rs"]
mod tests;

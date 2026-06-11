//! Frequency-domain regularization traits and generic deconvolution pipelines.
//!
//! Two patterns exist among the four deconvolution methods:
//!
//! **Pattern A** (single-pass): Wiener, Tikhonov
//! Pad → FFT → frequency-domain update rule → IFFT → crop.
//! Only the update rule differs between methods.
//!
//! **Pattern B** (iterative): Landweber, Richardson-Lucy
//! Initialize → loop { convolve → residual/ratio → convolve → update estimate }.
//! The inner-loop update logic differs (additive vs multiplicative).
//!
//! This module defines:
//! - [`Regularization`] — frequency-domain update rule for Pattern A
//! - `apply_single_pass` — generic single-pass pipeline
//! - `apply_iterative` — generic iterative pipeline

use super::helpers::{
    convolve, decode_coords, encode_flat, ifft_and_crop, pad_and_fft, pad_dims, pad_total,
};
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

// ── Pattern A: Frequency-domain regularization ─────────────────────────────

/// Frequency-domain update rule for single-pass deconvolution.
///
/// Given the FFT of the image `G` and the FFT of the PSF `H`,
/// implementations modify `img_padded` in place to contain the
/// restored frequency-domain signal.
pub trait Regularization {
    /// Apply the frequency-domain update rule.
    ///
    /// `img_padded` contains `G(ω)` on entry and `U(ω)` on exit.
    /// `ker_padded` contains `H(ω)`.
    /// `pad_dims` gives the shape of the padded arrays.
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        pad_dims: &[usize],
    );
}

/// Wiener filter: `U = G·H* / (|H|² + K)`.
///
/// `K` is the noise-to-signal power ratio.
pub struct WienerRule {
    /// Noise-to-signal power ratio K = Pn / Ps.
    pub noise_to_signal: f32,
}

impl Regularization for WienerRule {
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        _pad_dims: &[usize],
    ) {
        let k = self.noise_to_signal;
        for (g, &h) in img_padded.iter_mut().zip(ker_padded.iter()) {
            let denom = h.norm_sqr() + k;
            if denom < 1e-20 {
                *g = Complex::new(0.0, 0.0);
            } else {
                let scale = 1.0 / denom;
                *g = Complex::new(
                    (g.re * h.re + g.im * h.im) * scale,
                    (g.im * h.re - g.re * h.im) * scale,
                );
            }
        }
    }
}

/// Tikhonov filter: `U = G·H* / (|H|² + λ|L|²)`.
///
/// `|L|²` is the squared eigenvalue of the discrete Laplacian operator,
/// which depends on dimensionality (2-D vs 3-D) and frequency coordinates.
pub struct TikhonovRule {
    /// Regularization parameter λ.
    pub lambda: f32,
}

impl Regularization for TikhonovRule {
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        pad_dims: &[usize],
    ) {
        let lambda = self.lambda;
        match pad_dims.len() {
            2 => {
                let pad: [usize; 2] = [pad_dims[0], pad_dims[1]];
                apply_tikhonov::<2>(img_padded, ker_padded, &pad, lambda);
            }
            3 => {
                let pad: [usize; 3] = [pad_dims[0], pad_dims[1], pad_dims[2]];
                apply_tikhonov::<3>(img_padded, ker_padded, &pad, lambda);
            }
            _ => unreachable!("only 2-D and 3-D deconvolution are supported"),
        }
    }
}

/// Const-generic Tikhonov update: `|L(ω)|² = (2D − 2Σcos(ωd))²`.
///
/// The discrete Laplacian eigenvalue for dimensionality `D` is
/// `|L(ω)|² = (2D − 2cos(ω₀) − 2cos(ω₁) − … − 2cos(ω_{D−1}))²`.
///
/// For `D = 2`: `(4 − 2cos(ωx) − 2cos(ωy))²`.
/// For `D = 3`: `(6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`.
pub(super) fn apply_tikhonov<const D: usize>(
    img_padded: &mut [Complex<f32>],
    ker_padded: &[Complex<f32>],
    pad_dims: &[usize; D],
    lambda: f32,
) {
    assert!(
        D == 2 || D == 3,
        "only 2-D and 3-D deconvolution are supported"
    );

    for (idx, g_slot) in img_padded.iter_mut().enumerate() {
        let coords = decode_coords::<D>(idx, pad_dims);

        // Accumulate Laplacian eigenvalue: 2D − 2Σcos(ωd)
        let mut l_re = 2.0 * D as f32;
        for d in 0..D {
            let n_d = pad_dims[d];
            let f_d = if coords[d] <= n_d / 2 {
                coords[d] as f32 / n_d as f32
            } else {
                (coords[d] as f32 - n_d as f32) / n_d as f32
            };
            l_re -= 2.0 * (2.0 * PI * f_d).cos();
        }

        let l_sq = l_re * l_re;
        let h = ker_padded[idx];
        let g = *g_slot;
        let denom = h.norm_sqr() + lambda * l_sq;
        if denom < 1e-20 {
            *g_slot = Complex::new(0.0, 0.0);
        } else {
            let scale = 1.0 / denom;
            *g_slot = Complex::new(
                (g.re * h.re + g.im * h.im) * scale,
                (g.im * h.re - g.re * h.im) * scale,
            );
        }
    }
}

/// 2-D Tikhonov update: `|L(ω)|² = (4 − 2cos(ωx) − 2cos(ωy))²`.
///
/// Delegates to [`apply_tikhonov::<2>`]. Retained for backward compatibility.
#[deprecated(note = "use apply_tikhonov::<2> instead")]
#[allow(dead_code)]
fn apply_tikhonov_2d(
    img_padded: &mut [Complex<f32>],
    ker_padded: &[Complex<f32>],
    pad_dims: &[usize],
    lambda: f32,
) {
    let pad: [usize; 2] = [pad_dims[0], pad_dims[1]];
    apply_tikhonov::<2>(img_padded, ker_padded, &pad, lambda);
}

/// 3-D Tikhonov update: `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`.
///
/// Delegates to [`apply_tikhonov::<3>`]. Retained for backward compatibility.
#[deprecated(note = "use apply_tikhonov::<3> instead")]
#[allow(dead_code)]
fn apply_tikhonov_3d(
    img_padded: &mut [Complex<f32>],
    ker_padded: &[Complex<f32>],
    pad_dims: &[usize],
    lambda: f32,
) {
    let pad: [usize; 3] = [pad_dims[0], pad_dims[1], pad_dims[2]];
    apply_tikhonov::<3>(img_padded, ker_padded, &pad, lambda);
}

// ── Generic pipelines ──────────────────────────────────────────────────────

/// Single-pass deconvolution: pad → FFT → regularization rule → IFFT → crop.
///
/// Used by Wiener and Tikhonov. The public `apply_2d`/`apply_3d` methods
/// delegate here with the appropriate `R` implementation and const generic
/// dimension `D`.
pub(super) fn apply_single_pass<const D: usize, R: Regularization>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    ker_vals: &[f32],
    ker_dims: &[usize; D],
    rule: R,
) -> Vec<f32> {
    let pad = pad_dims::<D>(img_dims, ker_dims);
    let pad_n = pad_total::<D>(&pad);

    let (mut img_padded, ker_padded) =
        pad_and_fft::<D>(img_vals, img_dims, ker_vals, ker_dims, &pad, pad_n);

    rule.apply_rule(&mut img_padded, &ker_padded, &pad);

    ifft_and_crop::<D>(&mut img_padded, img_dims, &pad, pad_n)
}

/// Build the spatially-reversed (transposed) kernel for iterative deconvolution.
///
/// For 2-D: `h*(-y, -x)[ky, kx] = h[kh-1-ky, kw-1-kx]`
/// For 3-D: `h*(-z, -y, -x)[kz, ky, kx] = h[kd-1-kz, kh-1-ky, kw-1-kx]`
fn reversed_kernel<const D: usize>(ker_vals: &[f32], ker_dims: &[usize; D]) -> Vec<f32> {
    let total: usize = ker_dims.iter().product();
    let mut rev = vec![0.0_f32; total];
    for (flat, &v) in ker_vals.iter().enumerate() {
        let coords = decode_coords::<D>(flat, ker_dims);
        let rcoords: [usize; D] = std::array::from_fn(|d| ker_dims[d] - 1 - coords[d]);
        let rflat = encode_flat::<D>(&rcoords, ker_dims);
        rev[rflat] = v;
    }
    rev
}

// ── Iterative algorithm types ──────────────────────────────────────────────

/// Iterative deconvolution algorithm variant.
///
/// Replaces `is_landweber: bool` to eliminate boolean blindness at call sites.
pub(super) enum IterativeAlgorithm {
    /// Landweber gradient descent: additive update `uₖ₊₁ = uₖ + α · correction`.
    Landweber {
        /// Step size α (must satisfy `0 < α < 2 / σ_max²` for convergence).
        step_size: f32,
    },
    /// Richardson-Lucy expectation-maximization: multiplicative update
    /// `uₖ₊₁ = uₖ · correction`.
    RichardsonLucy,
}

/// Kernel data and iterative configuration for [`apply_iterative`].
///
/// Groups the kernel values/dimensions with algorithm parameters to stay
/// under the 7-argument clippy limit while keeping call sites readable.
pub(super) struct IterativeParams<'a, const D: usize> {
    /// Row-major kernel values.
    pub ker_vals: &'a [f32],
    /// Kernel dimensions.
    pub ker_dims: &'a [usize; D],
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f32,
    /// Algorithm variant and associated parameters.
    pub algorithm: IterativeAlgorithm,
}

/// Iterative deconvolution: Landweber or Richardson-Lucy.
///
/// The two methods share the same outer loop structure but differ in
/// how they compute the correction and update the estimate:
///
/// - **Landweber**: residual `g − h⋆uₖ`, convolve residual with `h*`,
///   additive update `uₖ₊₁ = uₖ + α · correction`
/// - **R-L**: ratio `g / (h⋆uₖ)`, convolve ratio with `h*`,
///   multiplicative update `uₖ₊₁ = uₖ · correction`
pub(super) fn apply_iterative<const D: usize>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    params: &IterativeParams<'_, D>,
) -> Vec<f32> {
    let ker_rev = reversed_kernel::<D>(params.ker_vals, params.ker_dims);
    let mut estimate: Vec<f32> = img_vals.to_vec();
    let n = estimate.len();

    // Pre-allocate scratch buffers outside the iteration loop to avoid
    // per-iteration heap allocation. The active buffer is selected per
    // algorithm variant; the other remains allocated but unused.
    let mut residual = vec![0.0_f32; n];
    let mut ratio = vec![1.0_f32; n];

    for _iter in 0..params.max_iterations {
        let forward = convolve::<D>(&estimate, img_dims, params.ker_vals, params.ker_dims);

        match &params.algorithm {
            IterativeAlgorithm::Landweber { step_size } => {
                // Landweber: compute residual, convolve with h*, add α·correction
                let mut max_residual = 0.0_f32;
                for ((r_slot, &img), &fwd) in
                    residual.iter_mut().zip(img_vals.iter()).zip(forward.iter())
                {
                    let r = img - fwd;
                    *r_slot = r;
                    max_residual = max_residual.max(r.abs());
                }
                let correction = convolve::<D>(&residual, img_dims, &ker_rev, params.ker_dims);
                for (est, &corr) in estimate.iter_mut().zip(correction.iter()) {
                    *est += *step_size * corr;
                }
                if max_residual < params.tolerance {
                    break;
                }
            }
            IterativeAlgorithm::RichardsonLucy => {
                // Richardson-Lucy: compute ratio, convolve with h*, multiply
                let mut max_ratio = 0.0_f32;
                ratio.fill(1.0);
                for ((r_slot, &img), &fwd) in
                    ratio.iter_mut().zip(img_vals.iter()).zip(forward.iter())
                {
                    if fwd > 1e-20 {
                        let r = img / fwd;
                        *r_slot = r;
                        max_ratio = max_ratio.max((r - 1.0).abs());
                    }
                }
                let correction = convolve::<D>(&ratio, img_dims, &ker_rev, params.ker_dims);
                for (est, &corr) in estimate.iter_mut().zip(correction.iter()) {
                    *est *= corr;
                }
                if max_ratio < params.tolerance {
                    break;
                }
            }
        }
    }

    estimate
}

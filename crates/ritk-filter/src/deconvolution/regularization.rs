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
    decode_coords, encode_flat, ifft_and_crop, pad_and_fft, pad_dims, pad_total, place_corner,
    run_fft,
};
use crate::fft::convolution::{ForwardFft, InverseFft};
use eunomia::Complex;

/// Default convergence tolerance for iterative deconvolution algorithms.
pub(crate) const DEFAULT_ITERATIVE_TOLERANCE: f32 = 1e-6;

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

/// Wiener filter, matching ITK's `WienerDeconvolutionImageFilter`:
///
/// ```text
/// U(ω) = G(ω)·H*(ω) / ( |H(ω)|² + Pn/|G(ω)|² )
/// ```
///
/// where the regularisation term `Pn/|G(ω)|²` = `1/snrSquared` matches
/// ITK's `WienerDeconvolutionImageFilter::GenerateData()` exactly.
/// `Pn` is the noise power spectral density — ITK's `NoiseVariance`.
/// The regularisation is frequency-adaptive: frequencies where `|G(ω)|²` is
/// small (weak observed signal) receive stronger suppression. This differs from
/// a constant-regularisation inverse filter (that is ITK's Tikhonov — see
/// [`TikhonovRule`]).
pub struct WienerRule {
    /// Noise power spectral density `Pn` (ITK `NoiseVariance`).
    pub noise_variance: f32,
}

impl Regularization for WienerRule {
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        _pad_dims: &[usize],
    ) {
        let pn = self.noise_variance;
        for (g, &h) in img_padded.iter_mut().zip(ker_padded.iter()) {
            // SNR-based regularisation matching ITK's WienerDeconvolutionImageFilter:
            //   snrSquared = |G(ω)|² / Pn
            //   1/snrSquared = Pn / |G(ω)|²
            // Clamp denominator away from zero to suppress division-by-zero at DC
            // or background-only frequencies.
            let pf = g.norm_sqr();
            let reg = pn / pf.max(1e-20);
            let denom = h.norm_sqr() + reg;
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

/// Tikhonov filter, matching ITK's `TikhonovDeconvolutionImageFilter`:
///
/// ```text
/// U(ω) = G(ω)·H*(ω) / ( |H(ω)|² + λ )
/// ```
///
/// A constant-regularised inverse filter: `λ` trades inversion sharpness against
/// noise amplification uniformly across frequency.
pub struct TikhonovRule {
    /// Regularization parameter λ.
    pub lambda: f32,
}

impl Regularization for TikhonovRule {
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        _pad_dims: &[usize],
    ) {
        let lambda = self.lambda;
        for (g, &h) in img_padded.iter_mut().zip(ker_padded.iter()) {
            let denom = h.norm_sqr() + lambda;
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

/// Direct inverse filter, matching ITK's `InverseDeconvolutionImageFilter`:
///
/// ```text
/// U(ω) = G(ω) / H(ω) = G(ω)·H*(ω) / |H(ω)|²   if |H(ω)| >= τ, else 0
/// ```
///
/// Unlike Tikhonov (which adds a ridge term `λ`), the inverse filter divides
/// directly by the OTF and zeros any frequency whose magnitude falls below the
/// `kernel_zero_magnitude_threshold` `τ` (ITK `KernelZeroMagnitudeThreshold`,
/// default 1e-3 in ITK; SimpleITK exposes it per call). This prevents unbounded
/// noise amplification at OTF nulls without smoothing the rest of the spectrum.
pub struct InverseRule {
    /// Magnitude threshold `τ` below which an OTF frequency is treated as zero.
    pub kernel_zero_magnitude_threshold: f32,
}

impl Regularization for InverseRule {
    fn apply_rule(
        &self,
        img_padded: &mut [Complex<f32>],
        ker_padded: &[Complex<f32>],
        _pad_dims: &[usize],
    ) {
        let tau = self.kernel_zero_magnitude_threshold;
        for (g, &h) in img_padded.iter_mut().zip(ker_padded.iter()) {
            // |H(ω)| compared against τ (ITK uses the complex magnitude, not |H|²).
            if h.norm() < tau {
                *g = Complex::new(0.0, 0.0);
            } else {
                let scale = 1.0 / h.norm_sqr();
                *g = Complex::new(
                    (g.re * h.re + g.im * h.im) * scale,
                    (g.im * h.re - g.re * h.im) * scale,
                );
            }
        }
    }
}

// ── Generic pipelines ──────────────────────────────────────────────────────

/// Single-pass deconvolution: pad → FFT → regularization rule → IFFT → crop.
///
/// Used by `WienerDeconvolution::apply` and `TikhonovDeconvolution::apply`
/// via the const-generic `D` parameter.
pub(super) fn apply_single_pass<const D: usize, R: Regularization>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    ker_vals: &[f32],
    ker_dims: &[usize; D],
    rule: R,
) -> Vec<f32> {
    let pad = pad_dims::<D>(img_dims, ker_dims);
    let pad_n = pad_total::<D>(&pad);
    // ITK CropOutput convention: crop at (kernelSize[d]-1)/2 ≈ ker_dims[d]/2 per axis.
    // Place the image at the same offset so IFFT at the crop positions gives the
    // correctly aligned deconvolved estimate.
    let offset: [usize; D] = std::array::from_fn(|d| ker_dims[d] / 2);

    let (mut img_padded, ker_padded) =
        pad_and_fft::<D>(img_vals, img_dims, ker_vals, ker_dims, &pad, pad_n, &offset);

    rule.apply_rule(&mut img_padded, &ker_padded, &pad);

    ifft_and_crop::<D>(&mut img_padded, img_dims, &pad, pad_n, &offset)
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

/// Per-iteration constraint applied to the Landweber estimate.
///
/// `NonNegative` clamps every voxel to `>= 0` after each additive update,
/// matching ITK `ProjectedLandweberDeconvolutionImageFilter`'s non-negativity
/// projection; `None` is plain `LandweberDeconvolutionImageFilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LandweberProjection {
    /// No constraint — plain Landweber gradient descent.
    #[default]
    None,
    /// Clamp the estimate to `>= 0` after each iteration.
    NonNegative,
}

/// Iterative deconvolution algorithm variant.
///
/// Replaces `is_landweber: bool` to eliminate boolean blindness at call sites.
pub enum IterativeAlgorithm {
    /// Landweber gradient descent: additive update `uₖ₊₁ = uₖ + α · correction`,
    /// optionally projected onto a constraint set each iteration.
    Landweber {
        /// Step size α (must satisfy `0 < α < 2 / σ_max²` for convergence).
        step_size: f32,
        /// Per-iteration projection constraint.
        projection: LandweberProjection,
    },
    /// Richardson-Lucy expectation-maximization: multiplicative update
    /// `uₖ₊₁ = uₖ · correction`.
    RichardsonLucy,
}

/// Kernel data and iterative configuration for [`apply_iterative`].
///
/// Groups the kernel values/dimensions with algorithm parameters to stay
/// under the 7-argument clippy limit while keeping call sites readable.
pub struct IterativeParams<'a, const D: usize> {
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

/// Pre-allocated scratch buffers and cached kernel FFTs for iterative deconvolution.
pub struct DeconvolutionScratch<const D: usize> {
    img_dims: [usize; D],
    ker_dims: [usize; D],
    pad_dims: [usize; D],
    pad_n: usize,
    img_pad: Vec<Complex<f32>>,
    ker_fft: Vec<Complex<f32>>,
    ker_rev_fft: Vec<Complex<f32>>,
    pub(super) forward: Vec<f32>,
    pub(super) correction: Vec<f32>,
    pub(super) residual: Vec<f32>,
    pub(super) ratio: Vec<f32>,
}

impl<const D: usize> DeconvolutionScratch<D> {
    /// Create a new deconvolution scratch structure.
    pub fn new() -> Self {
        Self {
            img_dims: [0; D],
            ker_dims: [0; D],
            pad_dims: [0; D],
            pad_n: 0,
            img_pad: Vec::new(),
            ker_fft: Vec::new(),
            ker_rev_fft: Vec::new(),
            forward: Vec::new(),
            correction: Vec::new(),
            residual: Vec::new(),
            ratio: Vec::new(),
        }
    }

    /// Initialize the scratch space for specific dimensions and kernels.
    ///
    /// Pre-pads and computes the forward FFT of the kernel and the reversed kernel,
    /// storing them inside `self.ker_fft` and `self.ker_rev_fft`.
    pub fn init(
        &mut self,
        img_dims: [usize; D],
        ker_vals: &[f32],
        ker_dims: [usize; D],
        ker_rev: &[f32],
    ) {
        self.img_dims = img_dims;
        self.ker_dims = ker_dims;
        self.pad_dims = pad_dims::<D>(&img_dims, &ker_dims);
        self.pad_n = pad_total::<D>(&self.pad_dims);

        self.img_pad.resize(self.pad_n, Complex::new(0.0, 0.0));
        self.ker_fft.resize(self.pad_n, Complex::new(0.0, 0.0));
        self.ker_rev_fft.resize(self.pad_n, Complex::new(0.0, 0.0));

        let out_n: usize = img_dims.iter().product();
        self.forward.resize(out_n, 0.0);
        self.correction.resize(out_n, 0.0);
        self.residual.resize(out_n, 0.0);
        self.ratio.resize(out_n, 1.0);

        // Precompute ker_fft
        self.img_pad.fill(Complex::new(0.0, 0.0));
        place_corner::<D>(&mut self.img_pad, ker_vals, &self.ker_dims, &self.pad_dims);
        run_fft::<D, ForwardFft>(&mut self.img_pad, &self.pad_dims);
        self.ker_fft.copy_from_slice(&self.img_pad);

        // Precompute ker_rev_fft
        self.img_pad.fill(Complex::new(0.0, 0.0));
        place_corner::<D>(&mut self.img_pad, ker_rev, &self.ker_dims, &self.pad_dims);
        run_fft::<D, ForwardFft>(&mut self.img_pad, &self.pad_dims);
        self.ker_rev_fft.copy_from_slice(&self.img_pad);
    }

    /// Perform FFT-based convolution of `image` with the forward kernel.
    ///
    /// Writes the result directly into `self.forward` with zero allocations.
    pub fn convolve_forward(&mut self, image: &[f32]) {
        self.img_pad.fill(Complex::new(0.0, 0.0));
        place_corner::<D>(&mut self.img_pad, image, &self.img_dims, &self.pad_dims);

        run_fft::<D, ForwardFft>(&mut self.img_pad, &self.pad_dims);

        for (a, &b) in self.img_pad.iter_mut().zip(self.ker_fft.iter()) {
            *a = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        run_fft::<D, InverseFft>(&mut self.img_pad, &self.pad_dims);

        let scale = 1.0_f32 / self.pad_n as f32;
        for (flat, r) in self.forward.iter_mut().enumerate() {
            let coords = decode_coords::<D>(flat, &self.img_dims);
            let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + self.ker_dims[d] / 2);
            let pflat = encode_flat::<D>(&pcoords, &self.pad_dims);
            *r = self.img_pad[pflat].re * scale;
        }
    }

    /// Perform FFT-based convolution of `self.residual` with the reversed kernel.
    ///
    /// Writes the result directly into `self.correction` with zero allocations.
    pub fn convolve_backward_residual(&mut self) {
        self.img_pad.fill(Complex::new(0.0, 0.0));
        place_corner::<D>(
            &mut self.img_pad,
            &self.residual,
            &self.img_dims,
            &self.pad_dims,
        );

        run_fft::<D, ForwardFft>(&mut self.img_pad, &self.pad_dims);

        for (a, &b) in self.img_pad.iter_mut().zip(self.ker_rev_fft.iter()) {
            *a = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        run_fft::<D, InverseFft>(&mut self.img_pad, &self.pad_dims);

        let scale = 1.0_f32 / self.pad_n as f32;
        for (flat, r) in self.correction.iter_mut().enumerate() {
            let coords = decode_coords::<D>(flat, &self.img_dims);
            let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + self.ker_dims[d] / 2);
            let pflat = encode_flat::<D>(&pcoords, &self.pad_dims);
            *r = self.img_pad[pflat].re * scale;
        }
    }

    /// Perform FFT-based convolution of `self.ratio` with the reversed kernel.
    ///
    /// Writes the result directly into `self.correction` with zero allocations.
    pub fn convolve_backward_ratio(&mut self) {
        self.img_pad.fill(Complex::new(0.0, 0.0));
        place_corner::<D>(
            &mut self.img_pad,
            &self.ratio,
            &self.img_dims,
            &self.pad_dims,
        );

        run_fft::<D, ForwardFft>(&mut self.img_pad, &self.pad_dims);

        for (a, &b) in self.img_pad.iter_mut().zip(self.ker_rev_fft.iter()) {
            *a = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        run_fft::<D, InverseFft>(&mut self.img_pad, &self.pad_dims);

        let scale = 1.0_f32 / self.pad_n as f32;
        for (flat, r) in self.correction.iter_mut().enumerate() {
            let coords = decode_coords::<D>(flat, &self.img_dims);
            let pcoords: [usize; D] = std::array::from_fn(|d| coords[d] + self.ker_dims[d] / 2);
            let pflat = encode_flat::<D>(&pcoords, &self.pad_dims);
            *r = self.img_pad[pflat].re * scale;
        }
    }
}

impl<const D: usize> Default for DeconvolutionScratch<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterative deconvolution: Landweber or Richardson-Lucy using scratch storage.
pub fn apply_iterative_with_scratch<const D: usize>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    params: &IterativeParams<'_, D>,
    scratch: &mut DeconvolutionScratch<D>,
) -> Vec<f32> {
    let ker_rev = reversed_kernel::<D>(params.ker_vals, params.ker_dims);
    scratch.init(*img_dims, params.ker_vals, *params.ker_dims, &ker_rev);

    let mut estimate = img_vals.to_vec();

    for _iter in 0..params.max_iterations {
        scratch.convolve_forward(&estimate);

        match &params.algorithm {
            IterativeAlgorithm::Landweber {
                step_size,
                projection,
            } => {
                let mut max_residual = 0.0_f32;
                for ((r_slot, &img), &fwd) in scratch
                    .residual
                    .iter_mut()
                    .zip(img_vals.iter())
                    .zip(scratch.forward.iter())
                {
                    let r = img - fwd;
                    *r_slot = r;
                    max_residual = max_residual.max(r.abs());
                }

                scratch.convolve_backward_residual();

                for (est, &corr) in estimate.iter_mut().zip(scratch.correction.iter()) {
                    *est += *step_size * corr;
                }

                if let LandweberProjection::NonNegative = projection {
                    for est in estimate.iter_mut() {
                        if *est < 0.0 {
                            *est = 0.0;
                        }
                    }
                }
                if max_residual < params.tolerance {
                    break;
                }
            }
            IterativeAlgorithm::RichardsonLucy => {
                let mut max_ratio = 0.0_f32;
                scratch.ratio.fill(1.0);
                for ((r_slot, &img), &fwd) in scratch
                    .ratio
                    .iter_mut()
                    .zip(img_vals.iter())
                    .zip(scratch.forward.iter())
                {
                    if fwd > 1e-20 {
                        let r = img / fwd;
                        *r_slot = r;
                        max_ratio = max_ratio.max((r - 1.0).abs());
                    }
                }

                scratch.convolve_backward_ratio();

                for (est, &corr) in estimate.iter_mut().zip(scratch.correction.iter()) {
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

/// Iterative deconvolution: Landweber or Richardson-Lucy.
pub(super) fn apply_iterative<const D: usize>(
    img_vals: &[f32],
    img_dims: &[usize; D],
    params: &IterativeParams<'_, D>,
) -> Vec<f32> {
    let mut scratch = DeconvolutionScratch::new();
    apply_iterative_with_scratch(img_vals, img_dims, params, &mut scratch)
}

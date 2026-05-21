//! Wiener deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Given `g = h ∗ u + n`, the Wiener filter minimises MSE by:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / (|H(ω)|² + K)
//! ```
//!
//! where `K = Pn / Ps` is the noise-to-signal power ratio.
//! When `K = 0`, this reduces to direct inverse filtering (noisy but exact).

use crate::filter::fft::convolution::{fft2d, fft3d, FftDir};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};

/// Wiener deconvolution filter (minimum mean-square error restoration).
///
/// Restores a degraded image `g = h ∗ u + n` given the PSF kernel `h` and
/// an estimate of the noise-to-signal power ratio `K = Pn / Ps`.
///
/// In the frequency domain:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + K)
/// ```
///
/// When `K = 0`, this reduces to direct inverse filtering (noisy).
/// When `K → ∞`, the output tends to zero (overly smooth).
///
/// # Use cases
/// - Motion blur correction
/// - Out-of-focus (defocus) restoration
/// - Medical image deconvolution with known PSF
///
/// # Complexity
/// O(N log N) for FFT-based execution.
pub struct WienerDeconvolution {
    /// Noise-to-signal power ratio K = Pn / Ps (default: 0.01).
    pub noise_to_signal: f32,
}

impl WienerDeconvolution {
    /// Create a new Wiener deconvolution filter with the given noise-to-signal ratio.
    pub fn new(noise_to_signal: f32) -> Self {
        Self { noise_to_signal }
    }

    /// Apply Wiener deconvolution to a 2-D image with a 2-D PSF kernel.
    ///
    /// Both image and kernel are padded to "full" convolution size,
    /// processed in frequency domain, then cropped back to original size.
    pub fn apply_2d<B: Backend>(
        &self,
        image: &Image<B, 2>,
        kernel: &Image<B, 2>,
    ) -> Result<Image<B, 2>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let [ih, iw] = img_dims;
        let [kh, kw] = ker_dims;

        let pad_h = (ih + kh - 1).next_power_of_two();
        let pad_w = (iw + kw - 1).next_power_of_two();
        let pad_n = pad_h * pad_w;

        let mut img_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for y in 0..ih {
            for x in 0..iw {
                img_padded[y * pad_w + x] = Complex::new(img_vals[y * iw + x], 0.0);
            }
        }

        // Zero-phase kernel centering: kernel[(kh/2, kw/2)] → padded[(0,0)]
        let mut ker_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for ky in 0..kh {
            for kx in 0..kw {
                let py = (ky + pad_h - kh / 2) % pad_h;
                let px = (kx + pad_w - kw / 2) % pad_w;
                ker_padded[py * pad_w + px] = Complex::new(ker_vals[ky * kw + kx], 0.0);
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        fft2d(&mut img_padded, pad_h, pad_w, &mut planner, FftDir::Forward);
        fft2d(&mut ker_padded, pad_h, pad_w, &mut planner, FftDir::Forward);

        let k = self.noise_to_signal;
        for i in 0..pad_n {
            let h = ker_padded[i];
            let g = img_padded[i];
            let denom = h.norm_sqr() + k;
            if denom < 1e-20 {
                img_padded[i] = Complex::new(0.0, 0.0);
            } else {
                let scale = 1.0 / denom;
                img_padded[i] = Complex::new(
                    (g.re * h.re + g.im * h.im) * scale,
                    (g.im * h.re - g.re * h.im) * scale,
                );
            }
        }

        fft2d(&mut img_padded, pad_h, pad_w, &mut planner, FftDir::Inverse);

        let scale = 1.0_f32 / pad_n as f32;
        let mut out_vals = vec![0.0_f32; ih * iw];
        for y in 0..ih {
            for x in 0..iw {
                out_vals[y * iw + x] = img_padded[y * pad_w + x].re * scale;
            }
        }

        Ok(rebuild(out_vals, img_dims, image))
    }

    /// Apply Wiener deconvolution to a 3-D image with a 3-D PSF kernel.
    ///
    /// Identical to `apply_2d` but operates on `[depth, rows, cols]` volumes.
    pub fn apply_3d<B: Backend>(
        &self,
        image: &Image<B, 3>,
        kernel: &Image<B, 3>,
    ) -> Result<Image<B, 3>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let [id, ih, iw] = img_dims;
        let [kd, kh, kw] = ker_dims;

        let pad_d = (id + kd - 1).next_power_of_two();
        let pad_h = (ih + kh - 1).next_power_of_two();
        let pad_w = (iw + kw - 1).next_power_of_two();
        let pad_n = pad_d * pad_h * pad_w;
        let pad_slice = pad_h * pad_w;

        let mut img_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..id {
            for y in 0..ih {
                for x in 0..iw {
                    img_padded[z * pad_slice + y * pad_w + x] =
                        Complex::new(img_vals[z * ih * iw + y * iw + x], 0.0);
                }
            }
        }

        // Zero-phase kernel centering
        let mut ker_padded = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for kz in 0..kd {
            for ky in 0..kh {
                for kx in 0..kw {
                    let pz = (kz + pad_d - kd / 2) % pad_d;
                    let py = (ky + pad_h - kh / 2) % pad_h;
                    let px = (kx + pad_w - kw / 2) % pad_w;
                    ker_padded[pz * pad_slice + py * pad_w + px] =
                        Complex::new(ker_vals[kz * kh * kw + ky * kw + kx], 0.0);
                }
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        fft3d(
            &mut img_padded,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );
        fft3d(
            &mut ker_padded,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );

        let k = self.noise_to_signal;
        for i in 0..pad_n {
            let h = ker_padded[i];
            let g = img_padded[i];
            let denom = h.norm_sqr() + k;
            if denom < 1e-20 {
                img_padded[i] = Complex::new(0.0, 0.0);
            } else {
                let scale = 1.0 / denom;
                img_padded[i] = Complex::new(
                    (g.re * h.re + g.im * h.im) * scale,
                    (g.im * h.re - g.re * h.im) * scale,
                );
            }
        }

        fft3d(
            &mut img_padded,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Inverse,
        );

        let scale = 1.0_f32 / pad_n as f32;
        let mut out_vals = vec![0.0_f32; id * ih * iw];
        for z in 0..id {
            for y in 0..ih {
                for x in 0..iw {
                    out_vals[z * ih * iw + y * iw + x] =
                        img_padded[z * pad_slice + y * pad_w + x].re * scale;
                }
            }
        }

        Ok(rebuild(out_vals, img_dims, image))
    }
}

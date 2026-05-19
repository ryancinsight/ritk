//! Tikhonov-regularized deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Minimises `||g − h ∗ u||² + λ||L ∗ u||²` where L is the Laplacian operator.
//!
//! In the frequency domain:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ · |L(ω)|²)
//! ```
//!
//! 2-D Laplacian: `|L(ω)|² = (4 − 2cos(ωx) − 2cos(ωy))²`
//! 3-D Laplacian: `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`
//!
//! Higher λ → smoother output (higher regularization strength).

use crate::filter::fft::convolution::{fft2d, fft3d, FftDir};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Tikhonov-regularized deconvolution (ridge regression in frequency domain).
///
/// Minimizes `||g − h ∗ u||² + λ||L ∗ u||²` where L is the Laplacian operator.
///
/// In the frequency domain:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ · |L(ω)|²)
/// ```
///
/// where `|L(ω)|² = (4 − 2cos(ωx) − 2cos(ωy))²` for 2-D discrete Laplacian
/// and `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²` for 3-D.
///
/// # Comparison to Wiener
/// Tikhonov uses a smoothness prior (λ|Lu|²) rather than a noise-to-signal
/// ratio. It tends to produce smoother restorations.
///
/// # Complexity
/// O(N log N).
pub struct TikhonovDeconvolution {
    /// Regularization parameter λ (default: 0.01).
    pub lambda: f32,
}

impl TikhonovDeconvolution {
    /// Create a new Tikhonov deconvolution filter with the given regularization parameter.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Apply Tikhonov deconvolution to a 2-D image.
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

        let lambda = self.lambda;
        for row in 0..pad_h {
            let fy = if row <= pad_h / 2 {
                row as f32 / pad_h as f32
            } else {
                (row as f32 - pad_h as f32) / pad_h as f32
            };
            let wy = 2.0 * PI * fy;
            for col in 0..pad_w {
                let idx = row * pad_w + col;
                let fx = col as f32 / pad_w as f32;
                let wx = 2.0 * PI * fx;
                // 2-D discrete Laplacian eigenvalue squared
                let l_re = 4.0 - 2.0 * wx.cos() - 2.0 * wy.cos();
                let l_sq = l_re * l_re;
                let h = ker_padded[idx];
                let g = img_padded[idx];
                let denom = h.norm_sqr() + lambda * l_sq;
                if denom < 1e-20 {
                    img_padded[idx] = Complex::new(0.0, 0.0);
                } else {
                    let scale = 1.0 / denom;
                    img_padded[idx] = Complex::new(
                        (g.re * h.re + g.im * h.im) * scale,
                        (g.im * h.re - g.re * h.im) * scale,
                    );
                }
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

    /// Apply Tikhonov deconvolution to a 3-D image.
    ///
    /// Uses the 3-D discrete Laplacian:
    /// `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`
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
        fft3d(&mut img_padded, pad_d, pad_h, pad_w, &mut planner, FftDir::Forward);
        fft3d(&mut ker_padded, pad_d, pad_h, pad_w, &mut planner, FftDir::Forward);

        let lambda = self.lambda;
        for depth in 0..pad_d {
            let fz = if depth <= pad_d / 2 {
                depth as f32 / pad_d as f32
            } else {
                (depth as f32 - pad_d as f32) / pad_d as f32
            };
            let wz = 2.0 * PI * fz;
            for row in 0..pad_h {
                let fy = if row <= pad_h / 2 {
                    row as f32 / pad_h as f32
                } else {
                    (row as f32 - pad_h as f32) / pad_h as f32
                };
                let wy = 2.0 * PI * fy;
                for col in 0..pad_w {
                    let idx = depth * pad_slice + row * pad_w + col;
                    let fx = col as f32 / pad_w as f32;
                    let wx = 2.0 * PI * fx;
                    // 3-D discrete Laplacian eigenvalue squared
                    let l_re = 6.0 - 2.0 * wx.cos() - 2.0 * wy.cos() - 2.0 * wz.cos();
                    let l_sq = l_re * l_re;
                    let h = ker_padded[idx];
                    let g = img_padded[idx];
                    let denom = h.norm_sqr() + lambda * l_sq;
                    if denom < 1e-20 {
                        img_padded[idx] = Complex::new(0.0, 0.0);
                    } else {
                        let scale = 1.0 / denom;
                        img_padded[idx] = Complex::new(
                            (g.re * h.re + g.im * h.im) * scale,
                            (g.im * h.re - g.re * h.im) * scale,
                        );
                    }
                }
            }
        }

        fft3d(&mut img_padded, pad_d, pad_h, pad_w, &mut planner, FftDir::Inverse);

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

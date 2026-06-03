//! Internal FFT-based convolution helpers for deconvolution filters.
//!
//! # Design
//!
//! `convolve_2d` and `convolve_3d` implement linear convolution via FFT with
//! "same" output cropping. Both functions are identical in structure; the only
//! variation is dimension (2D vs 3D), expressed through the function signature
//! rather than a cloned body.
//!
//! # Complexity
//!
//! O(N log N) where N = total number of samples in the zero-padded buffer.

use crate::filter::fft::convolution::{fft2d, fft3d, FftDir};
use rustfft::{num_complex::Complex, FftPlanner};

// ── 2-D ──────────────────────────────────────────────────────────────────────

/// FFT-based 2-D convolution returning a "same"-sized output.
///
/// # Arguments
/// - `image`  — row-major image slice, shape `[ih, iw]`
/// - `kernel` — row-major kernel slice, shape `[kh, kw]`
///
/// # Output
/// Row-major `Vec<f32>` of shape `[ih, iw]` (same as input).
///
/// # Invariant
/// Output length equals `ih * iw`.
pub(super) fn convolve_2d(
    image: &[f32],
    ih: usize,
    iw: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let pad_h = (ih + kh - 1).next_power_of_two();
    let pad_w = (iw + kw - 1).next_power_of_two();
    let pad_n = pad_h * pad_w;

    let mut img_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    for y in 0..ih {
        for x in 0..iw {
            img_pad[y * pad_w + x] = Complex::new(image[y * iw + x], 0.0);
        }
    }

    let mut ker_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    for ky in 0..kh {
        for kx in 0..kw {
            ker_pad[ky * pad_w + kx] = Complex::new(kernel[ky * kw + kx], 0.0);
        }
    }

    let mut planner = FftPlanner::<f32>::new();
    fft2d(&mut img_pad, pad_h, pad_w, &mut planner, FftDir::Forward);
    fft2d(&mut ker_pad, pad_h, pad_w, &mut planner, FftDir::Forward);

    for i in 0..pad_n {
        let a = img_pad[i];
        let b = ker_pad[i];
        img_pad[i] = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
    }

    fft2d(&mut img_pad, pad_h, pad_w, &mut planner, FftDir::Inverse);

    let scale = 1.0_f32 / pad_n as f32;
    let cy = kh / 2;
    let cx = kw / 2;
    let mut result = vec![0.0_f32; ih * iw];
    for y in 0..ih {
        for x in 0..iw {
            result[y * iw + x] = img_pad[(y + cy) * pad_w + (x + cx)].re * scale;
        }
    }
    result
}

// ── 3-D ──────────────────────────────────────────────────────────────────────

/// FFT-based 3-D convolution returning a "same"-sized output.
///
/// # Arguments
/// - `image`  — row-major image slice, shape `[id, ih, iw]`
/// - `kernel` — row-major kernel slice, shape `[kd, kh, kw]`
///
/// # Output
/// Row-major `Vec<f32>` of shape `[id, ih, iw]` (same as input).
///
/// # Invariant
/// Output length equals `id * ih * iw`.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolve_3d(
    image: &[f32],
    id: usize,
    ih: usize,
    iw: usize,
    kernel: &[f32],
    kd: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let pad_d = (id + kd - 1).next_power_of_two();
    let pad_h = (ih + kh - 1).next_power_of_two();
    let pad_w = (iw + kw - 1).next_power_of_two();
    let pad_n = pad_d * pad_h * pad_w;
    let pad_slice = pad_h * pad_w;

    let mut img_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    for z in 0..id {
        for y in 0..ih {
            for x in 0..iw {
                img_pad[z * pad_slice + y * pad_w + x] =
                    Complex::new(image[z * ih * iw + y * iw + x], 0.0);
            }
        }
    }

    let mut ker_pad = vec![Complex::new(0.0_f32, 0.0); pad_n];
    for kz in 0..kd {
        for ky in 0..kh {
            for kx in 0..kw {
                ker_pad[kz * pad_slice + ky * pad_w + kx] =
                    Complex::new(kernel[kz * kh * kw + ky * kw + kx], 0.0);
            }
        }
    }

    let mut planner = FftPlanner::<f32>::new();
    fft3d(
        &mut img_pad,
        pad_d,
        pad_h,
        pad_w,
        &mut planner,
        FftDir::Forward,
    );
    fft3d(
        &mut ker_pad,
        pad_d,
        pad_h,
        pad_w,
        &mut planner,
        FftDir::Forward,
    );

    for i in 0..pad_n {
        let a = img_pad[i];
        let b = ker_pad[i];
        img_pad[i] = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
    }

    fft3d(
        &mut img_pad,
        pad_d,
        pad_h,
        pad_w,
        &mut planner,
        FftDir::Inverse,
    );

    let scale = 1.0_f32 / pad_n as f32;
    let cz = kd / 2;
    let cy = kh / 2;
    let cx = kw / 2;
    let mut result = vec![0.0_f32; id * ih * iw];
    for z in 0..id {
        for y in 0..ih {
            for x in 0..iw {
                result[z * ih * iw + y * iw + x] =
                    img_pad[(z + cz) * pad_slice + (y + cy) * pad_w + (x + cx)].re * scale;
            }
        }
    }
    result
}

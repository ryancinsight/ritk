use crate::filter::fft::convolution::helpers::{fft3d, FftDir};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use rustfft::num_complex::Complex;
use std::marker::PhantomData;

// ── FftConvolution3DFilter ────────────────────────────────────────────────────

/// FFT-based 3-D convolution filter ("same" output convention).
///
/// Convolves a 3-D volume with a 3-D kernel using a full separable 3-D FFT
/// (not slice-by-slice). This correctly models 3-D blur, point-spread-function
/// deconvolution (when inverted), and volumetric filtering where the kernel
/// varies along all three axes.
///
/// # Algorithm
///
/// 1. Pad volume `[D, H, W]` and kernel `[KD, KH, KW]` to
///    `[pad_d, pad_h, pad_w]` where each padded size is the next power of two
///    greater than or equal to the sum of the corresponding dimensions.
/// 2. Place both arrays at the top-left origin of the padded buffer.
/// 3. Apply separable 3-D forward FFT: row-wise → column-wise → depth-wise.
/// 4. Multiply pointwise in the frequency domain.
/// 5. Apply separable 3-D inverse FFT; normalize by `1 / (pad_d · pad_h · pad_w)`.
/// 6. Extract the "same" output: a `[D, H, W]` window starting at
///    `(⌊KD/2⌋, ⌊KH/2⌋, ⌊KW/2⌋)`.
///
/// # Complexity
///
/// O(N log N) where N = pad_d · pad_h · pad_w.
///
/// # Output
///
/// The output has the same spatial shape as the input and preserves origin,
/// spacing, and direction metadata.
pub struct FftConvolution3DFilter<B: Backend> {
    kernel_vals: Vec<f32>,
    kernel_depth: usize,
    kernel_rows: usize,
    kernel_cols: usize,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> FftConvolution3DFilter<B> {
    /// Construct from a 3-D kernel volume.
    pub fn new(kernel: &Image<B, 3>) -> Result<Self> {
        let [kd, kh, kw] = kernel.shape();
        if kd == 0 || kh == 0 || kw == 0 {
            return Err(anyhow!(
                "FftConvolution3DFilter: kernel dimensions must be non-zero, got [{kd}, {kh}, {kw}]"
            ));
        }
        let (k_vals, _) = extract_vec(kernel)?;
        Ok(Self {
            kernel_vals: k_vals,
            kernel_depth: kd,
            kernel_rows: kh,
            kernel_cols: kw,
            _phantom: PhantomData::<fn() -> B>,
        })
    }

    /// Convolve `volume` with the stored 3-D kernel ("same" convention).
    ///
    /// # Mathematical contract
    ///
    /// For a kernel with a Dirac delta at position `(⌊KD/2⌋, ⌊KH/2⌋, ⌊KW/2⌋)`
    /// and all other entries zero, `apply(volume)` reproduces `volume` within
    /// floating-point precision.
    pub fn apply(&self, volume: &Image<B, 3>) -> Result<Image<B, 3>> {
        let [d, h, w] = volume.shape();
        let (vals, dims) = extract_vec(volume)?;

        let kd = self.kernel_depth;
        let kh = self.kernel_rows;
        let kw = self.kernel_cols;

        // Padding must be >= dim + krn − 1 to suppress circular aliasing.
        let pad_d = (d + kd - 1).next_power_of_two();
        let pad_h = (h + kh - 1).next_power_of_two();
        let pad_w = (w + kw - 1).next_power_of_two();
        let pad_n = pad_d * pad_h * pad_w;
        let slice = pad_h * pad_w; // size of one depth slice in padded buffer

        // Zero-padded volume: placed at top-left (origin).
        let mut vol_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..d {
            for r in 0..h {
                for c in 0..w {
                    vol_buf[z * slice + r * pad_w + c] =
                        Complex::new(vals[z * h * w + r * w + c], 0.0);
                }
            }
        }

        // Zero-padded kernel: placed at top-left (origin).
        let mut ker_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..kd {
            for r in 0..kh {
                for c in 0..kw {
                    ker_buf[z * slice + r * pad_w + c] =
                        Complex::new(self.kernel_vals[z * kh * kw + r * kw + c], 0.0);
                }
            }
        }

        let mut planner = rustfft::FftPlanner::<f32>::new();
        fft3d(
            &mut vol_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );
        fft3d(
            &mut ker_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );

        // Point-wise complex multiply: vol_buf[i] *= ker_buf[i].
        // (a + bi)(c + di) = (ac − bd) + (ad + bc)i
        for i in 0..pad_n {
            let a = vol_buf[i];
            let b = ker_buf[i];
            vol_buf[i] = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        fft3d(
            &mut vol_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Inverse,
        );

        // Normalize by 1/pad_n and extract "same" window at
        // (⌊KD/2⌋, ⌊KH/2⌋, ⌊KW/2⌋).
        let scale = 1.0_f32 / pad_n as f32;
        let off_d = kd / 2;
        let off_h = kh / 2;
        let off_w = kw / 2;

        let mut out = vec![0.0_f32; d * h * w];
        for z in 0..d {
            for r in 0..h {
                for c in 0..w {
                    out[z * h * w + r * w + c] =
                        vol_buf[(z + off_d) * slice + (r + off_h) * pad_w + (c + off_w)].re * scale;
                }
            }
        }

        Ok(rebuild(out, dims, volume))
    }
}

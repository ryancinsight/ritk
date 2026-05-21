//! FFT-based convolution and normalized cross-correlation filters.
//!
//! # Convolution theorem
//!
//! ```text
//! conv(f, g) = IFFT(FFT(f) · FFT(g))
//! ```
//!
//! # Algorithm (2-D "same" convolution)
//!
//! 1. Pad image `[h, w]` and kernel `[kr, kc]` to `[pad_r, pad_c]` where
//!    `pad_r = next_power_of_two(h + kr − 1)` and
//!    `pad_c = next_power_of_two(w + kc − 1)`.
//!    This padding prevents circular aliasing in the linear convolution result.
//! 2. Place both arrays at the **top-left origin** of the padded buffer (no
//!    centring shift), so the kernel phase is zero and no quadrant swap is needed.
//! 3. Apply separable 2-D forward FFT: row-wise, then column-wise.
//! 4. Multiply element-wise in the frequency domain.
//! 5. Apply separable 2-D inverse FFT; normalize by `1 / (pad_r · pad_c)`.
//! 6. Extract the "same" output: a `[h, w]` window starting at
//!    `(⌊kr/2⌋, ⌊kc/2⌋)`.
//!
//! # Proof of "same" crop offset
//!
//! With kernel placed at origin, the circular convolution at position `(r, c)` is
//!
//! ```text
//! C[r, c] = Σ_{r'=r−(kr−1)}^{r} Σ_{c'=c−(kc−1)}^{c} f[r', c'] · g[r−r', c−c']
//! ```
//!
//! The full linear convolution occupies `[0, h+kr−2] × [0, w+kc−2]`.  The
//! "same" window of size `[h, w]` that centres output pixel `(r, c)` over input
//! pixel `(r, c)` starts at `(⌊kr/2⌋, ⌊kc/2⌋)`.  For the Dirac delta
//! `δ[⌊kr/2⌋, ⌊kc/2⌋] = 1`, the crop recovers `f[r, c]` exactly.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};
use std::marker::PhantomData;

// ── FftConvolutionFilter ───────────────────────────────────────────────────────

/// FFT-based 2-D convolution filter ("same" output convention).
///
/// Stores the raw kernel values at construction time. At `apply` time the
/// padded FFTs of both image and kernel are computed, multiplied in the
/// frequency domain, and the "same"-sized output is cropped from the IFFT
/// result.  Any image size is accepted regardless of the kernel size.
///
/// # Complexity
///
/// O(N log N) where N = pad_r · pad_c.
///
/// # Output
///
/// The output has the same spatial shape as the input and preserves origin,
/// spacing, and direction metadata.
pub struct FftConvolutionFilter<B: Backend> {
    kernel_vals: Vec<f32>,
    kernel_rows: usize,
    kernel_cols: usize,
    _phantom: PhantomData<B>,
}

impl<B: Backend> FftConvolutionFilter<B> {
    /// Construct from a 2-D kernel image.
    pub fn new(kernel: &Image<B, 2>) -> Result<Self> {
        let [kr, kc] = kernel.shape();
        if kr == 0 || kc == 0 {
            return Err(anyhow!(
                "FftConvolutionFilter: kernel dimensions must be non-zero, got [{kr}, {kc}]"
            ));
        }
        let (k_vals, _) = extract_vec(kernel)?;
        Ok(Self {
            kernel_vals: k_vals,
            kernel_rows: kr,
            kernel_cols: kc,
            _phantom: PhantomData,
        })
    }

    /// Convolve `image` with the stored kernel ("same" convention).
    ///
    /// # Mathematical contract
    ///
    /// For an odd-sized kernel with `δ[kr/2, kc/2] = 1` and all other entries
    /// zero, `apply(image)` reproduces `image` within floating-point precision.
    pub fn apply(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, w] = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let kr = self.kernel_rows;
        let kc = self.kernel_cols;

        // Padding must be >= h + kr − 1 to suppress circular aliasing.
        let pad_r = (h + kr - 1).next_power_of_two();
        let pad_c = (w + kc - 1).next_power_of_two();
        let pad_n = pad_r * pad_c;

        // Zero-padded image: placed at top-left (origin).
        let mut img_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..h {
            for c in 0..w {
                img_buf[r * pad_c + c] = Complex::new(vals[r * w + c], 0.0);
            }
        }

        // Zero-padded kernel: placed at top-left (origin).
        let mut ker_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..kr {
            for c in 0..kc {
                ker_buf[r * pad_c + c] = Complex::new(self.kernel_vals[r * kc + c], 0.0);
            }
        }

        let mut planner = FftPlanner::<f32>::new();

        fft2d(&mut img_buf, pad_r, pad_c, &mut planner, FftDir::Forward);
        fft2d(&mut ker_buf, pad_r, pad_c, &mut planner, FftDir::Forward);

        // Point-wise complex multiply: img_buf[i] *= ker_buf[i].
        // (a + bi)(c + di) = (ac − bd) + (ad + bc)i
        for i in 0..pad_n {
            let a = img_buf[i];
            let b = ker_buf[i];
            img_buf[i] = Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
        }

        fft2d(&mut img_buf, pad_r, pad_c, &mut planner, FftDir::Inverse);

        // Normalize by 1/pad_n and extract "same" window at (⌊kr/2⌋, ⌊kc/2⌋).
        let scale = 1.0_f32 / pad_n as f32;
        let off_r = kr / 2;
        let off_c = kc / 2;
        let mut out = vec![0.0_f32; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = img_buf[(r + off_r) * pad_c + (c + off_c)].re * scale;
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

// ── FftNormalizedCorrelationFilter ────────────────────────────────────────────

/// FFT-based cross-correlation filter for template matching.
///
/// Computes a partial-normalized cross-correlation map between a query image
/// and a stored template.
///
/// # Mathematical specification
///
/// ```text
/// xcorr[r, c] = IFFT(FFT(I) · conj(FFT(T̂)))[r, c]
/// out[r, c]   = xcorr[r, c] / (‖T̂‖₂ · pad_n)
/// ```
///
/// where `T̂ = T − mean(T)` is the mean-subtracted template.
///
/// # Notes on normalization
///
/// Only the template L₂ norm is removed (partial normalization).  Full NCC
/// — dividing by the local image patch energy via an integral image — is
/// deferred to phase 2 of GAP-262-FLT-01.
///
/// # Output interpretation
///
/// `out[r, c]` is the cross-correlation at lag `(r, c)`.  For template
/// matching, locate the position of maximum `out[r, c]`.
pub struct FftNormalizedCorrelationFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_rows: usize,
    template_cols: usize,
    /// L₂ norm of the mean-centred template used for partial normalization.
    template_norm: f32,
    _phantom: PhantomData<B>,
}

impl<B: Backend> FftNormalizedCorrelationFilter<B> {
    /// Construct from a 2-D template image.
    ///
    /// The template is mean-subtracted: `T̂ = T − mean(T)`.
    pub fn new(template: &Image<B, 2>) -> Result<Self> {
        let [tr, tc] = template.shape();
        if tr == 0 || tc == 0 {
            return Err(anyhow!(
                "FftNormalizedCorrelationFilter: template dimensions must be non-zero, got [{tr}, {tc}]"
            ));
        }
        let (t_vals, _) = extract_vec(template)?;
        let t_mean: f32 = t_vals.iter().sum::<f32>() / (tr * tc) as f32;
        let centered: Vec<f32> = t_vals.iter().map(|&v| v - t_mean).collect();
        let template_norm = centered.iter().map(|&v| v * v).sum::<f32>().sqrt();
        Ok(Self {
            template_vals: centered,
            template_rows: tr,
            template_cols: tc,
            template_norm,
            _phantom: PhantomData,
        })
    }

    /// Compute the cross-correlation map; output has the same shape as `image`.
    pub fn apply(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, w] = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let tr = self.template_rows;
        let tc = self.template_cols;

        let pad_r = (h + tr - 1).next_power_of_two();
        let pad_c = (w + tc - 1).next_power_of_two();
        let pad_n = pad_r * pad_c;

        // Zero-padded image at origin.
        let mut img_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..h {
            for c in 0..w {
                img_buf[r * pad_c + c] = Complex::new(vals[r * w + c], 0.0);
            }
        }

        // Zero-padded template at origin.
        let mut tmpl_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..tr {
            for c in 0..tc {
                tmpl_buf[r * pad_c + c] = Complex::new(self.template_vals[r * tc + c], 0.0);
            }
        }

        let mut planner = FftPlanner::<f32>::new();

        fft2d(&mut img_buf, pad_r, pad_c, &mut planner, FftDir::Forward);
        fft2d(&mut tmpl_buf, pad_r, pad_c, &mut planner, FftDir::Forward);

        // Cross-correlation: multiply image FFT by conjugate of template FFT.
        // (a + bi) · conj(c + di) = (a + bi)(c − di) = (ac + bd) + (bc − ad)i
        for i in 0..pad_n {
            let a = img_buf[i];
            let b = tmpl_buf[i];
            img_buf[i] = Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im);
        }

        fft2d(&mut img_buf, pad_r, pad_c, &mut planner, FftDir::Inverse);

        // Partial normalization: divide by ‖T̂‖₂ · pad_n.
        let denom = if self.template_norm > 1e-10_f32 {
            self.template_norm * pad_n as f32
        } else {
            1.0_f32
        };
        let scale = 1.0_f32 / denom;
        let mut out = vec![0.0_f32; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = img_buf[r * pad_c + c].re * scale;
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

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
    _phantom: PhantomData<B>,
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
            _phantom: PhantomData,
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

        let mut planner = FftPlanner::<f32>::new();

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

// ── FftNormalizedCorrelation3DFilter ───────────────────────────────────────────

/// FFT-based 3-D normalized cross-correlation filter (partial NCC).
///
/// Computes the normalized cross-correlation between a 3-D volume and a 3-D
/// template using a full separable 3-D FFT (not slice-by-slice).
///
/// # Mathematical specification
///
/// ```text
/// xcorr[z,r,c] = IFFT(FFT(V) · conj(FFT(T̂)))[z,r,c]
/// out[z,r,c]   = xcorr[z,r,c] / (‖T̂‖₂ · pad_n)
/// ```
///
/// where `T̂ = T − mean(T)` is the mean-subtracted template.
///
/// # Notes on normalization
///
/// Only the template L₂ norm is removed (partial normalization).  Full NCC
/// — dividing by the local volume patch energy via an integral volume — is
/// deferred.
///
/// # Output interpretation
///
/// `out[z,r,c]` is the cross-correlation at lag `(z,r,c)`.  For template
/// matching, locate the position of maximum `out[z,r,c]`.
///
/// # Algorithm
///
/// 1. Mean-subtract template: `T̂ = T − mean(T)`.
/// 2. Pad volume `[D, H, W]` and template `[TD, TH, TW]` to
///    `[pad_d, pad_h, pad_w]` (next power of two).
/// 3. Apply separable 3-D forward FFT to both.
/// 4. Multiply pointwise: `FFT(V) · conj(FFT(T̂))`.
/// 5. Apply separable 3-D inverse FFT; normalize by `1 / (pad_n · ‖T̂‖₂)`.
/// 6. Extract output of size `[D, H, W]` starting at `(0, 0, 0)`
///    (zero offset — unlike convolution, NCC is not centred).
pub struct FftNormalizedCorrelation3DFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_depth: usize,
    template_rows: usize,
    template_cols: usize,
    /// L₂ norm of the mean-centred template used for partial normalization.
    template_norm: f32,
    _phantom: PhantomData<B>,
}

impl<B: Backend> FftNormalizedCorrelation3DFilter<B> {
    /// Construct from a 3-D template volume.
    ///
    /// The template is mean-subtracted: `T̂ = T − mean(T)`.
    pub fn new(template: &Image<B, 3>) -> Result<Self> {
        let [td, tr, tc] = template.shape();
        if td == 0 || tr == 0 || tc == 0 {
            return Err(anyhow!(
                "FftNormalizedCorrelation3DFilter: template dimensions must be non-zero, got [{td}, {tr}, {tc}]"
            ));
        }
        let (t_vals, _) = extract_vec(template)?;
        let n = td * tr * tc;
        let t_mean: f32 = t_vals.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = t_vals.iter().map(|&v| v - t_mean).collect();
        let template_norm = centered.iter().map(|&v| v * v).sum::<f32>().sqrt();
        Ok(Self {
            template_vals: centered,
            template_depth: td,
            template_rows: tr,
            template_cols: tc,
            template_norm,
            _phantom: PhantomData,
        })
    }

    /// Compute the 3-D cross-correlation map; output has the same shape as `volume`.
    pub fn apply(&self, volume: &Image<B, 3>) -> Result<Image<B, 3>> {
        let [d, h, w] = volume.shape();
        let (vals, dims) = extract_vec(volume)?;

        let td = self.template_depth;
        let tr = self.template_rows;
        let tc = self.template_cols;

        // Padding must be >= dim + tmpl − 1 to suppress circular aliasing.
        let pad_d = (d + td - 1).next_power_of_two();
        let pad_h = (h + tr - 1).next_power_of_two();
        let pad_w = (w + tc - 1).next_power_of_two();
        let pad_n = pad_d * pad_h * pad_w;
        let slice = pad_h * pad_w;

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

        // Zero-padded template: placed at top-left (origin).
        let mut tmpl_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..td {
            for r in 0..tr {
                for c in 0..tc {
                    tmpl_buf[z * slice + r * pad_w + c] =
                        Complex::new(self.template_vals[z * tr * tc + r * tc + c], 0.0);
                }
            }
        }

        let mut planner = FftPlanner::<f32>::new();

        fft3d(
            &mut vol_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );
        fft3d(
            &mut tmpl_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Forward,
        );

        // Cross-correlation: multiply volume FFT by conjugate of template FFT.
        // (a + bi) · conj(c + di) = (a + bi)(c − di) = (ac + bd) + (bc − ad)i
        for i in 0..pad_n {
            let a = vol_buf[i];
            let b = tmpl_buf[i];
            vol_buf[i] = Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im);
        }

        fft3d(
            &mut vol_buf,
            pad_d,
            pad_h,
            pad_w,
            &mut planner,
            FftDir::Inverse,
        );

        // Partial normalization: divide by ‖T̂‖₂ · pad_n.
        let denom = if self.template_norm > 1e-10_f32 {
            self.template_norm * pad_n as f32
        } else {
            1.0_f32
        };
        let scale = 1.0_f32 / denom;

        // Extract result at zero offset (no crop for NCC).
        let mut out = vec![0.0_f32; d * h * w];
        for z in 0..d {
            for r in 0..h {
                for c in 0..w {
                    out[z * h * w + r * w + c] = vol_buf[z * slice + r * pad_w + c].re * scale;
                }
            }
        }

        Ok(rebuild(out, dims, volume))
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// FFT transform direction.
pub(crate) enum FftDir {
    Forward,
    Inverse,
}

/// In-place separable 2-D FFT (or IFFT) on a row-major buffer of shape
/// `[rows, cols]`.
///
/// Pass 1: 1-D transform along each row (transform length = `cols`).
/// Pass 2: 1-D transform along each column via a scratch column buffer
///          (transform length = `rows`).
///
/// `rustfft`'s `process` method performs the transform in-place and allocates
/// scratch space internally.
pub(crate) fn fft2d(
    buf: &mut [Complex<f32>],
    rows: usize,
    cols: usize,
    planner: &mut FftPlanner<f32>,
    dir: FftDir,
) {
    let row_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(cols),
        FftDir::Inverse => planner.plan_fft_inverse(cols),
    };
    let col_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(rows),
        FftDir::Inverse => planner.plan_fft_inverse(rows),
    };

    // Row-wise pass.
    for r in 0..rows {
        row_fft.process(&mut buf[r * cols..(r + 1) * cols]);
    }

    // Column-wise pass via scratch buffer.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buf[r * cols + c];
        }
        col_fft.process(&mut col_buf);
        for r in 0..rows {
            buf[r * cols + c] = col_buf[r];
        }
    }
}

/// In-place separable 3-D FFT (or IFFT) on a row-major buffer of shape
/// `[depth, rows, cols]`.
///
/// Pass 1: 1-D transform along each row (transform length = `cols`).
/// Pass 2: 1-D transform along each column (transform length = `rows`).
/// Pass 3: 1-D transform along the depth axis (transform length = `depth`).
///
/// `rustfft`'s `process` method performs the transform in-place and allocates
/// scratch space internally.
pub(crate) fn fft3d(
    buf: &mut [Complex<f32>],
    depth: usize,
    rows: usize,
    cols: usize,
    planner: &mut FftPlanner<f32>,
    dir: FftDir,
) {
    let row_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(cols),
        FftDir::Inverse => planner.plan_fft_inverse(cols),
    };
    let col_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(rows),
        FftDir::Inverse => planner.plan_fft_inverse(rows),
    };
    let depth_fft = match dir {
        FftDir::Forward => planner.plan_fft_forward(depth),
        FftDir::Inverse => planner.plan_fft_inverse(depth),
    };

    let slice = rows * cols;

    // Row-wise pass: for each (depth, row), transform along cols.
    for d in 0..depth {
        for r in 0..rows {
            row_fft.process(&mut buf[d * slice + r * cols..d * slice + (r + 1) * cols]);
        }
    }

    // Column-wise pass: for each (depth, col), transform along rows.
    let mut col_buf = vec![Complex::new(0.0_f32, 0.0); rows];
    for d in 0..depth {
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = buf[d * slice + r * cols + c];
            }
            col_fft.process(&mut col_buf);
            for r in 0..rows {
                buf[d * slice + r * cols + c] = col_buf[r];
            }
        }
    }

    // Depth-wise pass: for each (row, col), transform along depth.
    let mut depth_buf = vec![Complex::new(0.0_f32, 0.0); depth];
    for r in 0..rows {
        for c in 0..cols {
            for d in 0..depth {
                depth_buf[d] = buf[d * slice + r * cols + c];
            }
            depth_fft.process(&mut depth_buf);
            for d in 0..depth {
                buf[d * slice + r * cols + c] = depth_buf[d];
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_convolution.rs"]
mod tests;

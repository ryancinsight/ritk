use crate::filter::fft::convolution::helpers::{fft2d, FftDir};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use rustfft::num_complex::Complex;
use std::marker::PhantomData;

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
/// Only the template L₂ norm is removed (partial normalization). Full NCC
/// — dividing by the local image patch energy via an integral image — is
/// deferred to phase 2 of GAP-262-FLT-01.
///
/// # Output interpretation
///
/// `out[r, c]` is the cross-correlation at lag `(r, c)`. For template
/// matching, locate the position of maximum `out[r, c]`.
pub struct FftNormalizedCorrelationFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_rows: usize,
    template_cols: usize,
    /// L₂ norm of the mean-centred template used for partial normalization.
    template_norm: f32,
    _phantom: PhantomData<fn() -> B>,
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
            _phantom: PhantomData::<fn() -> B>,
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

        let mut planner = rustfft::FftPlanner::<f32>::new();
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

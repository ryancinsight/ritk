use crate::fft::convolution::helpers::{fft2d, ForwardFft, InverseFft};
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};
use num_complex::Complex;
use std::marker::PhantomData;

/// Minimum NCC denominator; below this the correlation output is clamped to 0.
/// 3 orders of magnitude above f32 epsilon (~1.2e-7).
const NCC_DENOM_FLOOR: f32 = 1e-10;

// ── FftNormalizedCorrelationFilter ────────────────────────────────────────────

/// FFT-based normalized cross-correlation filter for template matching.
///
/// Computes the fully normalized cross-correlation map (Lewis 1995) between a
/// query image and a stored template, matching ITK's
/// `FFTNormalizedCorrelationImageFilter` in value semantics: the map equals
/// `1.0` where the template aligns with an identical image patch.
///
/// # Mathematical specification
///
/// At lag `(r, c)`, with template window `N = tr·tc` and `T̂ = T − mean(T)`,
///
/// ```text
/// num(r,c)    = Σ I(r+i, c+j) · T̂(i, j)                       (= Σ (I−Īwin)·T̂, since ΣT̂ = 0)
/// Σ I, Σ I²   = local window sum / sum-of-squares of I         (box correlation)
/// energy(r,c) = Σ I² − (Σ I)² / N                             (= Σ (I − Īwin)²)
/// out(r,c)    = num(r,c) / ( sqrt(energy(r,c)) · ‖T̂‖₂ )
/// ```
///
/// The window sums are obtained by correlating `I` and `I²` with a box of ones
/// of the template's size, all via FFT, so the cost stays `O(N log N)`. Both
/// `I` and the box are zero-padded, so windows overhanging the image edge use
/// the in-bounds support (the out-of-range contribution is 0).
///
/// # Output interpretation
///
/// `out[r, c]` is the normalized correlation at lag `(r, c)` in `[−1, 1]`. For
/// template matching, locate the position of maximum `out[r, c]`.
pub struct FftNormalizedCorrelationFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_rows: usize,
    template_cols: usize,
    /// L₂ norm of the mean-centred template ‖T̂‖₂ used in the NCC denominator.
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

    /// Compute the normalized cross-correlation map; output has the same shape
    /// as `image`.
    pub fn apply(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, w] = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let tr = self.template_rows;
        let tc = self.template_cols;
        let window_n = (tr * tc) as f32;

        let pad_r = (h + tr - 1).next_power_of_two();
        let pad_c = (w + tc - 1).next_power_of_two();
        let pad_n = pad_r * pad_c;

        // Zero-padded buffers: image I, its square I², the mean-centred template
        // T̂, and a box of ones (template footprint) for window sums.
        let mut img_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut img2_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..h {
            for c in 0..w {
                let v = vals[r * w + c];
                img_buf[r * pad_c + c] = Complex::new(v, 0.0);
                img2_buf[r * pad_c + c] = Complex::new(v * v, 0.0);
            }
        }

        let mut tmpl_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut box_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for r in 0..tr {
            for c in 0..tc {
                tmpl_buf[r * pad_c + c] = Complex::new(self.template_vals[r * tc + c], 0.0);
                box_buf[r * pad_c + c] = Complex::new(1.0, 0.0);
            }
        }

        fft2d::<ForwardFft>(&mut img_buf, pad_r, pad_c);
        fft2d::<ForwardFft>(&mut img2_buf, pad_r, pad_c);
        fft2d::<ForwardFft>(&mut tmpl_buf, pad_r, pad_c);
        fft2d::<ForwardFft>(&mut box_buf, pad_r, pad_c);

        // Three correlations share the image/template/box spectra. Correlation
        // multiplies by the conjugate of the kernel spectrum:
        // (a + bi)·conj(c + di) = (ac + bd) + (bc − ad)i.
        let corr = |a: Complex<f32>, b: Complex<f32>| {
            Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
        };
        let mut num_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I·T̂
        let mut sum_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I (window)
        let mut sumsq_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I² (window)
        for i in 0..pad_n {
            num_buf[i] = corr(img_buf[i], tmpl_buf[i]);
            sum_buf[i] = corr(img_buf[i], box_buf[i]);
            sumsq_buf[i] = corr(img2_buf[i], box_buf[i]);
        }
        fft2d::<InverseFft>(&mut num_buf, pad_r, pad_c);
        fft2d::<InverseFft>(&mut sum_buf, pad_r, pad_c);
        fft2d::<InverseFft>(&mut sumsq_buf, pad_r, pad_c);

        // Apollo's inverse FFT path is unnormalized; divide each correlation by pad_n.
        let inv_pad = 1.0_f32 / pad_n as f32;
        let t_norm = self.template_norm;
        let mut out = vec![0.0_f32; h * w];
        for r in 0..h {
            for c in 0..w {
                let idx = r * pad_c + c;
                let num = num_buf[idx].re * inv_pad;
                let lsum = sum_buf[idx].re * inv_pad;
                let lsumsq = sumsq_buf[idx].re * inv_pad;
                // Σ (I − Īwin)² = Σ I² − (Σ I)² / N, clamped against round-off.
                let energy = (lsumsq - lsum * lsum / window_n).max(0.0);
                let denom = energy.sqrt() * t_norm;
                out[r * w + c] = if denom > NCC_DENOM_FLOOR {
                    num / denom
                } else {
                    0.0
                };
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

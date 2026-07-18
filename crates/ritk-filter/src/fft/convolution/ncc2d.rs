use crate::fft::convolution::helpers::{fft2d, ForwardFft, InverseFft};
use crate::fft::convolution::padding::checked_fft_shape_2d;
use anyhow::{anyhow, Result};
use eunomia::Complex;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use std::marker::PhantomData;

/// Minimum NCC denominator; below this the correlation output is clamped to 0.
/// 3 orders of magnitude above f32 epsilon (~1.2e-7).
const NCC_DENOM_FLOOR: f32 = 1e-10;

// â”€â”€ FftNormalizedCorrelationFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// FFT-based normalized cross-correlation filter for template matching.
///
/// Computes the fully normalized cross-correlation map (Lewis 1995) between a
/// query image and a stored template, matching ITK's
/// `FFTNormalizedCorrelationImageFilter` in value semantics: the map equals
/// `1.0` where the template aligns with an identical image patch.
///
/// # Mathematical specification
///
/// At lag `(r, c)`, with template window `N = trÂ·tc` and `TÌ‚ = T âˆ’ mean(T)`,
///
/// ```text
/// num(r,c)    = Î£ I(r+i, c+j) Â· TÌ‚(i, j)                       (= Î£ (Iâˆ’Äªwin)Â·TÌ‚, since Î£TÌ‚ = 0)
/// Î£ I, Î£ IÂ²   = local window sum / sum-of-squares of I         (box correlation)
/// energy(r,c) = Î£ IÂ² âˆ’ (Î£ I)Â² / N                             (= Î£ (I âˆ’ Äªwin)Â²)
/// out(r,c)    = num(r,c) / ( sqrt(energy(r,c)) Â· â€–TÌ‚â€–â‚‚ )
/// ```
///
/// The window sums are obtained by correlating `I` and `IÂ²` with a box of ones
/// of the template's size, all via FFT, so the cost stays `O(N log N)`. Both
/// `I` and the box are zero-padded, so windows overhanging the image edge use
/// the in-bounds support (the out-of-range contribution is 0).
///
/// # Output interpretation
///
/// `out[r, c]` is the normalized correlation at lag `(r, c)` in `[âˆ’1, 1]`. For
/// template matching, locate the position of maximum `out[r, c]`.
pub struct FftNormalizedCorrelationFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_rows: usize,
    template_cols: usize,
    /// Lâ‚‚ norm of the mean-centred template â€–TÌ‚â€–â‚‚ used in the NCC denominator.
    template_norm: f32,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> FftNormalizedCorrelationFilter<B> {
    /// Construct from a 2-D template image.
    ///
    /// The template is mean-subtracted: `TÌ‚ = T âˆ’ mean(T)`.
    pub fn new(template: &Image<f32, B, 2>) -> Result<Self> {
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
    pub fn apply(&self, image: &Image<f32, B, 2>) -> Result<Image<f32, B, 2>> {
        let [h, w] = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let tr = self.template_rows;
        let tc = self.template_cols;
        let window_n = (tr * tc) as f32;

        let fft_shape = checked_fft_shape_2d([h, w], [tr, tc], "FftNormalizedCorrelationFilter")?;
        let (pad_r, pad_c, pad_n) = (fft_shape.rows, fft_shape.cols, fft_shape.len);

        // Zero-padded buffers: image I, its square IÂ², the mean-centred template
        // TÌ‚, and a box of ones (template footprint) for window sums.
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
        // (a + bi)Â·conj(c + di) = (ac + bd) + (bc âˆ’ ad)i.
        let corr = |a: Complex<f32>, b: Complex<f32>| {
            Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
        };
        let mut num_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Î£ IÂ·TÌ‚
        let mut sum_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Î£ I (window)
        let mut sumsq_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Î£ IÂ² (window)
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
                // Î£ (I âˆ’ Äªwin)Â² = Î£ IÂ² âˆ’ (Î£ I)Â² / N, clamped against round-off.
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

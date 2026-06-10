use crate::filter::fft::convolution::helpers::{fft3d, ForwardFft, InverseFft};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use rustfft::num_complex::Complex;
use std::marker::PhantomData;

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
/// Only the template L₂ norm is removed (partial normalization). Full NCC
/// — dividing by the local volume patch energy via an integral volume — is
/// deferred.
///
/// # Output interpretation
///
/// `out[z,r,c]` is the cross-correlation at lag `(z,r,c)`. For template
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
    _phantom: PhantomData<fn() -> B>,
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
            _phantom: PhantomData::<fn() -> B>,
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

        let mut planner = rustfft::FftPlanner::<f32>::new();
        fft3d::<ForwardFft>(&mut vol_buf, pad_d, pad_h, pad_w, &mut planner);
        fft3d::<ForwardFft>(&mut tmpl_buf, pad_d, pad_h, pad_w, &mut planner);

        // Cross-correlation: multiply volume FFT by conjugate of template FFT.
        // (a + bi) · conj(c + di) = (a + bi)(c − di) = (ac + bd) + (bc − ad)i
        for i in 0..pad_n {
            let a = vol_buf[i];
            let b = tmpl_buf[i];
            vol_buf[i] = Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im);
        }

        fft3d::<InverseFft>(&mut vol_buf, pad_d, pad_h, pad_w, &mut planner);

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

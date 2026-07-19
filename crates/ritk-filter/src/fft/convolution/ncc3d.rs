use crate::fft::convolution::helpers::{fft3d, ForwardFft, InverseFft};
use crate::fft::convolution::padding::checked_fft_shape_3d;
use anyhow::{anyhow, Result};
use eunomia::Complex;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use std::marker::PhantomData;

/// Minimum NCC denominator; below this the correlation output is clamped to 0.
/// 3 orders of magnitude above f32 epsilon (~1.2e-7).
const NCC_DENOM_FLOOR: f32 = 1e-10;

// ── FftNormalizedCorrelation3DFilter ───────────────────────────────────────────

/// FFT-based 3-D normalized cross-correlation filter (Lewis 1995, full NCC).
///
/// Computes the fully normalized cross-correlation between a 3-D volume and a
/// 3-D template using a full separable 3-D FFT (not slice-by-slice), matching
/// ITK's `FFTNormalizedCorrelationImageFilter` in value semantics: the map
/// equals `1.0` where the template aligns with an identical volume patch.
///
/// # Mathematical specification
///
/// At lag `(z, r, c)`, with template window `N = td·tr·tc` and
/// `TÌ‚ = T − mean(T)`,
///
/// ```text
/// num         = Σ V(z+k, r+i, c+j) · TÌ‚(k, i, j)              (= Σ (V − VÌ„win)·TÌ‚)
/// Σ V, Σ V²   = local window sum / sum-of-squares of V         (box correlation)
/// energy      = Σ V² − (Σ V)² / N                             (= Σ (V − VÌ„win)²)
/// out         = num / ( sqrt(energy) · —–TÌ‚—–₂ )
/// ```
///
/// The window sums come from correlating `V` and `V²` with a box of ones of the
/// template's size, all via FFT, keeping the cost `O(N log N)`.
///
/// # Output interpretation
///
/// `out[z,r,c]` is the normalized correlation at lag `(z,r,c)` in `[−1, 1]`. For
/// template matching, locate the position of maximum `out[z,r,c]`.
pub struct FftNormalizedCorrelation3DFilter<B: Backend> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    template_depth: usize,
    template_rows: usize,
    template_cols: usize,
    /// L₂ norm of the mean-centred template —–TÌ‚—–₂ used in the NCC denominator.
    template_norm: f32,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> FftNormalizedCorrelation3DFilter<B> {
    /// Construct from a 3-D template volume.
    ///
    /// The template is mean-subtracted: `TÌ‚ = T − mean(T)`.
    pub fn new(template: &Image<f32, B, 3>) -> Result<Self> {
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

    /// Compute the 3-D normalized cross-correlation map; output has the same
    /// shape as `volume`.
    pub fn apply(&self, volume: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let [d, h, w] = volume.shape();
        let (vals, dims) = extract_vec(volume)?;

        let td = self.template_depth;
        let tr = self.template_rows;
        let tc = self.template_cols;
        let window_n = (td * tr * tc) as f32;

        // Padding must be >= dim + tmpl − 1 to suppress circular aliasing.
        let fft_shape =
            checked_fft_shape_3d([d, h, w], [td, tr, tc], "FftNormalizedCorrelation3DFilter")?;
        let (pad_d, pad_h, pad_w, pad_n, slice) = (
            fft_shape.depth,
            fft_shape.rows,
            fft_shape.cols,
            fft_shape.len,
            fft_shape.slice_len,
        );

        // Zero-padded buffers: volume V, its square V², mean-centred template TÌ‚,
        // and a box of ones (template footprint) for window sums.
        let mut vol_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut vol2_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..d {
            for r in 0..h {
                for c in 0..w {
                    let v = vals[z * h * w + r * w + c];
                    vol_buf[z * slice + r * pad_w + c] = Complex::new(v, 0.0);
                    vol2_buf[z * slice + r * pad_w + c] = Complex::new(v * v, 0.0);
                }
            }
        }

        let mut tmpl_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut box_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for z in 0..td {
            for r in 0..tr {
                for c in 0..tc {
                    tmpl_buf[z * slice + r * pad_w + c] =
                        Complex::new(self.template_vals[z * tr * tc + r * tc + c], 0.0);
                    box_buf[z * slice + r * pad_w + c] = Complex::new(1.0, 0.0);
                }
            }
        }

        fft3d::<ForwardFft>(&mut vol_buf, pad_d, pad_h, pad_w);
        fft3d::<ForwardFft>(&mut vol2_buf, pad_d, pad_h, pad_w);
        fft3d::<ForwardFft>(&mut tmpl_buf, pad_d, pad_h, pad_w);
        fft3d::<ForwardFft>(&mut box_buf, pad_d, pad_h, pad_w);

        // Correlation multiplies by the conjugate of the kernel spectrum:
        // (a + bi)·conj(c + di) = (ac + bd) + (bc − ad)i.
        let corr = |a: Complex<f32>, b: Complex<f32>| {
            Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
        };
        let mut num_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ V·TÌ‚
        let mut sum_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ V (window)
        let mut sumsq_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ V² (window)
        for i in 0..pad_n {
            num_buf[i] = corr(vol_buf[i], tmpl_buf[i]);
            sum_buf[i] = corr(vol_buf[i], box_buf[i]);
            sumsq_buf[i] = corr(vol2_buf[i], box_buf[i]);
        }
        fft3d::<InverseFft>(&mut num_buf, pad_d, pad_h, pad_w);
        fft3d::<InverseFft>(&mut sum_buf, pad_d, pad_h, pad_w);
        fft3d::<InverseFft>(&mut sumsq_buf, pad_d, pad_h, pad_w);

        // Apollo's inverse FFT path is unnormalized; divide each correlation by pad_n.
        let inv_pad = 1.0_f32 / pad_n as f32;
        let t_norm = self.template_norm;
        let mut out = vec![0.0_f32; d * h * w];
        for z in 0..d {
            for r in 0..h {
                for c in 0..w {
                    let idx = z * slice + r * pad_w + c;
                    let num = num_buf[idx].re * inv_pad;
                    let lsum = sum_buf[idx].re * inv_pad;
                    let lsumsq = sumsq_buf[idx].re * inv_pad;
                    let energy = (lsumsq - lsum * lsum / window_n).max(0.0);
                    let denom = energy.sqrt() * t_norm;
                    out[z * h * w + r * w + c] = if denom > NCC_DENOM_FLOOR {
                        num / denom
                    } else {
                        0.0
                    };
                }
            }
        }

        Ok(rebuild(out, dims, volume))
    }
}

//! Rank-generic FFT-based fully normalized cross-correlation (Lewis 1995).
//!
//! One const-generic implementation serves both ranks: the historical 2-D and
//! 3-D names are preserved as [`pub type`](FftNormalizedCorrelationFilter)
//! aliases, so no call site moves.

use crate::fft::convolution::fft_strategy::{fft_nd, ForwardFft, InverseFft};
use crate::fft::convolution::padding::checked_fft_shape;
use anyhow::{anyhow, Result};
use eunomia::Complex;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use std::marker::PhantomData;

/// Minimum NCC denominator; below this the correlation output is clamped to 0.
/// 3 orders of magnitude above f32 epsilon (~1.2e-7).
const NCC_DENOM_FLOOR: f32 = 1e-10;

/// Public type name for a given rank, used in diagnostics.
fn type_name<const D: usize>() -> &'static str {
    match D {
        3 => "FftNormalizedCorrelation3DFilter",
        _ => "FftNormalizedCorrelationFilter",
    }
}

// ── FftNormalizedCorrelation ─────────────────────────────────────────────────

/// FFT-based fully normalized cross-correlation filter for `D`-dimensional
/// images (Lewis 1995), `D = 2` or `D = 3`.
///
/// Computes the fully normalized cross-correlation map between a query image
/// and a stored template, matching ITK's `FFTNormalizedCorrelationImageFilter`
/// in value semantics: the map equals `1.0` where the template aligns with an
/// identical image patch.
///
/// # Mathematical specification
///
/// At lag `p`, with template window `N = Π td` and `TÌ‚ = T − mean(T)`,
///
/// ```text
/// num(p)      = Σ I(p+i) · TÌ‚(i)                     (= Σ (I−Īwin)·TÌ‚, since ΣTÌ‚ = 0)
/// Σ I, Σ I²   = local window sum / sum-of-squares of I        (box correlation)
/// energy(p)   = Σ I² − (Σ I)² / N                     (= Σ (I − Īwin)²)
/// out(p)      = num(p) / ( sqrt(energy(p)) · —–TÌ‚—–₂ )
/// ```
///
/// The window sums are obtained by correlating `I` and `I²` with a box of ones
/// of the template's size, all via FFT, so the cost stays `O(N log N)`. Both
/// `I` and the box are zero-padded, so windows overhanging the image edge use
/// the in-bounds support (the out-of-range contribution is 0).
///
/// # Output interpretation
///
/// `out[p]` is the normalized correlation at lag `p` in `[−1, 1]`. For template
/// matching, locate the position of maximum `out[p]`.
pub struct FftNormalizedCorrelation<B: Backend, const D: usize> {
    /// Mean-centred template values (row-major, placed at origin).
    template_vals: Vec<f32>,
    /// Template spatial shape (row-major, outermost axis first).
    template_shape: [usize; D],
    /// L₂ norm of the mean-centred template —–TÌ‚—–₂ used in the NCC denominator.
    template_norm: f32,
    _phantom: PhantomData<fn() -> B>,
}

/// 2-D FFT normalized cross-correlation filter.
pub type FftNormalizedCorrelationFilter<B> = FftNormalizedCorrelation<B, 2>;

/// 3-D FFT normalized cross-correlation filter.
pub type FftNormalizedCorrelation3DFilter<B> = FftNormalizedCorrelation<B, 3>;

impl<B: Backend, const D: usize> FftNormalizedCorrelation<B, D> {
    /// Construct from a `D`-D template image.
    ///
    /// The template is mean-subtracted: `TÌ‚ = T − mean(T)`.
    pub fn new(template: &Image<f32, B, D>) -> Result<Self> {
        let template_shape = template.shape();
        if template_shape.contains(&0) {
            let dims = join_dims(&template_shape);
            return Err(anyhow!(
                "{}: template dimensions must be non-zero, got [{dims}]",
                type_name::<D>()
            ));
        }
        let (t_vals, _) = extract_vec(template)?;
        let n: usize = template_shape.iter().product();
        let t_mean: f32 = t_vals.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = t_vals.iter().map(|&v| v - t_mean).collect();
        let template_norm = centered.iter().map(|&v| v * v).sum::<f32>().sqrt();

        Ok(Self {
            template_vals: centered,
            template_shape,
            template_norm,
            _phantom: PhantomData::<fn() -> B>,
        })
    }

    /// Compute the normalized cross-correlation map; the output has the same
    /// shape as `image`.
    pub fn apply(&self, image: &Image<f32, B, D>) -> Result<Image<f32, B, D>> {
        let shape = image.shape();
        let (vals, dims) = extract_vec(image)?;

        let template_shape = self.template_shape;
        let window_n: f32 = template_shape.iter().product::<usize>() as f32;

        // Padding must be >= dim + tmpl − 1 on every axis to suppress circular
        // aliasing.
        let fft_shape = checked_fft_shape::<D>(shape, template_shape, type_name::<D>())?;
        let pad = fft_shape.dims;
        let pad_n = fft_shape.len;

        // Flat-index-to-padded-offset maps for the image and the template. Both
        // are placed at the padded origin (no centring shift).
        let in_strides = row_major_strides(shape);
        let pad_strides = row_major_strides(pad);
        let template_strides = row_major_strides(template_shape);
        let n: usize = shape.iter().product();

        // Zero-padded buffers: image I, its square I², the mean-centred template
        // TÌ‚, and a box of ones (template footprint) for window sums.
        let mut img_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut img2_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for index in 0..n {
            let offset = origin_offset(index, &shape, &in_strides, &pad_strides);
            let value = vals[index];
            img_buf[offset] = Complex::new(value, 0.0);
            img2_buf[offset] = Complex::new(value * value, 0.0);
        }

        let template_n: usize = template_shape.iter().product();
        let mut tmpl_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        let mut box_buf = vec![Complex::new(0.0_f32, 0.0); pad_n];
        for index in 0..template_n {
            let offset = origin_offset(index, &template_shape, &template_strides, &pad_strides);
            tmpl_buf[offset] = Complex::new(self.template_vals[index], 0.0);
            box_buf[offset] = Complex::new(1.0, 0.0);
        }

        fft_nd::<D, ForwardFft>(&mut img_buf, &pad);
        fft_nd::<D, ForwardFft>(&mut img2_buf, &pad);
        fft_nd::<D, ForwardFft>(&mut tmpl_buf, &pad);
        fft_nd::<D, ForwardFft>(&mut box_buf, &pad);

        // Three correlations share the image/template/box spectra. Correlation
        // multiplies by the conjugate of the kernel spectrum:
        // (a + bi)·conj(c + di) = (ac + bd) + (bc − ad)i.
        let corr = |a: Complex<f32>, b: Complex<f32>| {
            Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
        };
        let mut num_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I·TÌ‚
        let mut sum_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I (window)
        let mut sumsq_buf = vec![Complex::new(0.0_f32, 0.0); pad_n]; // Σ I² (window)
        for i in 0..pad_n {
            num_buf[i] = corr(img_buf[i], tmpl_buf[i]);
            sum_buf[i] = corr(img_buf[i], box_buf[i]);
            sumsq_buf[i] = corr(img2_buf[i], box_buf[i]);
        }
        fft_nd::<D, InverseFft>(&mut num_buf, &pad);
        fft_nd::<D, InverseFft>(&mut sum_buf, &pad);
        fft_nd::<D, InverseFft>(&mut sumsq_buf, &pad);

        // Apollo's inverse FFT path is unnormalized; divide each correlation by pad_n.
        let inv_pad = 1.0_f32 / pad_n as f32;
        let t_norm = self.template_norm;
        let mut out = vec![0.0_f32; n];
        for index in 0..n {
            let offset = origin_offset(index, &shape, &in_strides, &pad_strides);
            let num = num_buf[offset].re * inv_pad;
            let lsum = sum_buf[offset].re * inv_pad;
            let lsumsq = sumsq_buf[offset].re * inv_pad;
            // Σ (I − Īwin)² = Σ I² − (Σ I)² / N, clamped against round-off.
            let energy = (lsumsq - lsum * lsum / window_n).max(0.0);
            let denom = energy.sqrt() * t_norm;
            out[index] = if denom > NCC_DENOM_FLOOR {
                num / denom
            } else {
                0.0
            };
        }

        Ok(rebuild(out, dims, image))
    }
}

/// Row-major strides for a shape (last axis contiguous).
fn row_major_strides<const D: usize>(shape: [usize; D]) -> [usize; D] {
    let mut strides = [1usize; D];
    for axis in (0..D.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// Offset of flat row-major `index` of a `shape`-shaped array in a padded
/// buffer, with the array placed at the padded origin.
fn origin_offset<const D: usize>(
    index: usize,
    shape: &[usize; D],
    in_strides: &[usize; D],
    pad_strides: &[usize; D],
) -> usize {
    let mut offset = 0;
    for axis in 0..D {
        let coord = (index / in_strides[axis]) % shape[axis];
        offset += coord * pad_strides[axis];
    }
    offset
}

/// Render a shape as the `a, b, …` body of a `[…]` diagnostic.
fn join_dims(shape: &[usize]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

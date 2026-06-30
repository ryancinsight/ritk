//! Masked FFT normalized cross-correlation
//! (`itk::MaskedFFTNormalizedCorrelationImageFilter` / `sitk.MaskedFFTNormalizedCorrelation`).
//!
//! # Mathematical Specification
//!
//! Padfield's (2012) masked normalized cross-correlation. Given a fixed image `F`
//! with mask `Mf` and a moving image `T` with mask `Mt`, the NCC over every
//! relative translation is computed with a handful of FFTs at a size `≥ Nf+Nt−1`
//! (so the circular correlation equals the linear one over the valid full output):
//!
//! ```text
//! rotate T, Mt by 180°
//! overlap   = round(IFFT(M̂f·M̂t_rot))                     (clamped ≥ 0)
//! corrF     = IFFT(F̂·M̂t_rot),  corrM = IFFT(M̂f·T̂_rot)
//! numerator = IFFT(F̂·T̂_rot) − corrF·corrM/overlap
//! fixedDen  = max(IFFT(F²̂·M̂t_rot) − corrF²/overlap, 0)
//! movingDen = max(IFFT(M̂f·T²̂_rot) − corrM²/overlap, 0)
//! NCC       = numerator / √(fixedDen·movingDen)
//! ```
//!
//! post-processed: 0 where `denominator < precisionTolerance`, where `overlap`
//! is below the required overlap (`max(fraction·maxOverlap, requiredNumber, 1)`),
//! and clamped to `[−1, 1]`. Hats denote the FFT of the corresponding
//! (masked / squared-masked) field. Output extent is `Nf+Nt−1` per axis.
//!
//! Voxels with very low overlap (e.g. a single overlapping pixel ⇒ zero local
//! variance) are numerically degenerate and rounding-dependent; the
//! `required_fraction`/`required_number` parameters gate them, and with a
//! non-trivial fraction the result is float-exact to SimpleITK.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use eunomia::Complex;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

use crate::fft::convolution::{fft_nd, ForwardFft, InverseFft};

/// Masked FFT normalized cross-correlation (`itk::MaskedFFTNormalizedCorrelation`).
#[derive(Debug, Clone, Default)]
pub struct MaskedFftNormalizedCorrelationFilter {
    /// Minimum overlapping voxels required for a valid correlation (ITK default 0).
    pub required_number_of_overlapping_pixels: u64,
    /// Minimum overlap as a fraction of the maximum overlap (ITK default 0.0).
    pub required_fraction_of_overlapping_pixels: f32,
}

/// Zero-pad a real field into a complex buffer of shape `dims`, optionally
/// 180°-rotating it (placing the rotated content at the origin).
fn pad(src: &[f32], sdims: [usize; 3], dims: [usize; 3], rotate: bool) -> Vec<Complex<f32>> {
    let [sz, sy, sx] = sdims;
    let [_nz, ny, nx] = dims;
    let mut buf = vec![Complex::new(0.0f32, 0.0); dims[0] * ny * nx];
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let v = src[(z * sy + y) * sx + x];
                let (dz, dy, dx) = if rotate {
                    (sz - 1 - z, sy - 1 - y, sx - 1 - x)
                } else {
                    (z, y, x)
                };
                buf[(dz * ny + dy) * nx + dx] = Complex::new(v, 0.0);
            }
        }
    }
    buf
}

fn fwd(mut buf: Vec<Complex<f32>>, dims: [usize; 3]) -> Vec<Complex<f32>> {
    fft_nd::<3, ForwardFft>(&mut buf, &dims);
    buf
}

/// Element-wise product of two spectra, inverse-FFT'd and normalized to the real
/// circular correlation.
fn corr(a: &[Complex<f32>], b: &[Complex<f32>], dims: [usize; 3]) -> Vec<f32> {
    let n = (dims[0] * dims[1] * dims[2]) as f32;
    let mut prod: Vec<Complex<f32>> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    fft_nd::<3, InverseFft>(&mut prod, &dims);
    prod.iter().map(|c| c.re / n).collect()
}

impl MaskedFftNormalizedCorrelationFilter {
    /// Compute the masked FFT NCC. `fixed`/`fixed_mask` share a shape, as do
    /// `moving`/`moving_mask`. Output extent is `fixed + moving − 1` per axis.
    pub fn apply<B: Backend>(
        &self,
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        fixed_mask: &Image<B, 3>,
        moving_mask: &Image<B, 3>,
    ) -> Result<Image<B, 3>> {
        let (f, fd) = extract_vec_infallible(fixed);
        let (mf, fmd) = extract_vec_infallible(fixed_mask);
        let (t, td) = extract_vec_infallible(moving);
        let (mt, tmd) = extract_vec_infallible(moving_mask);
        if fmd != fd {
            bail!("masked_fft_ncc: fixed mask shape {fmd:?} != fixed {fd:?}");
        }
        if tmd != td {
            bail!("masked_fft_ncc: moving mask shape {tmd:?} != moving {td:?}");
        }
        let dims = [fd[0] + td[0] - 1, fd[1] + td[1] - 1, fd[2] + td[2] - 1];
        let n = dims[0] * dims[1] * dims[2];

        // Masked / squared-masked fields.
        let fm: Vec<f32> = f.iter().zip(&mf).map(|(a, b)| a * b).collect();
        let f2m: Vec<f32> = f.iter().zip(&mf).map(|(a, b)| a * a * b).collect();
        let tm: Vec<f32> = t.iter().zip(&mt).map(|(a, b)| a * b).collect();
        let t2m: Vec<f32> = t.iter().zip(&mt).map(|(a, b)| a * a * b).collect();

        let f_fft = fwd(pad(&fm, fd, dims, false), dims);
        let mf_fft = fwd(pad(&mf, fd, dims, false), dims);
        let f2_fft = fwd(pad(&f2m, fd, dims, false), dims);
        let t_fft = fwd(pad(&tm, td, dims, true), dims);
        let mt_fft = fwd(pad(&mt, td, dims, true), dims);
        let t2_fft = fwd(pad(&t2m, td, dims, true), dims);

        let overlap: Vec<f32> = corr(&mt_fft, &mf_fft, dims)
            .iter()
            .map(|v| v.round().max(0.0))
            .collect();
        let mask_corr_f = corr(&f_fft, &mt_fft, dims);
        let mask_corr_m = corr(&t_fft, &mf_fft, dims);
        let corr_ft = corr(&f_fft, &t_fft, dims);
        let fixed_dc = corr(&f2_fft, &mt_fft, dims);
        let moving_dc = corr(&mf_fft, &t2_fft, dims);

        let overlap_ref = &overlap;
        let corr_ft_ref = &corr_ft;
        let mask_corr_f_ref = &mask_corr_f;
        let mask_corr_m_ref = &mask_corr_m;
        let fixed_dc_ref = &fixed_dc;
        let moving_dc_ref = &moving_dc;

        let (max_denom, max_overlap) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            n,
            || (0.0_f32, 0.0_f32),
            |(mx_d, mx_ov), i| {
                let ov = overlap_ref[i];
                let ov_div = if ov > 0.0 { ov } else { 1.0 };
                let fixed_d =
                    (fixed_dc_ref[i] - mask_corr_f_ref[i] * mask_corr_f_ref[i] / ov_div).max(0.0);
                let moving_d =
                    (moving_dc_ref[i] - mask_corr_m_ref[i] * mask_corr_m_ref[i] / ov_div).max(0.0);
                let denom = (fixed_d * moving_d).sqrt();
                (mx_d.max(denom), mx_ov.max(ov))
            },
            |(a_d, a_ov), (b_d, b_ov)| (a_d.max(b_d), a_ov.max(b_ov)),
        );

        let tol = 1000.0 * f32::EPSILON * max_denom;
        let req = (self.required_fraction_of_overlapping_pixels * max_overlap)
            .max(self.required_number_of_overlapping_pixels as f32)
            .max(1.0);

        let out = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let ov = overlap_ref[i];
            if ov >= req {
                let ov_div = if ov > 0.0 { ov } else { 1.0 };
                let fixed_d =
                    (fixed_dc_ref[i] - mask_corr_f_ref[i] * mask_corr_f_ref[i] / ov_div).max(0.0);
                let moving_d =
                    (moving_dc_ref[i] - mask_corr_m_ref[i] * mask_corr_m_ref[i] / ov_div).max(0.0);
                let denom = (fixed_d * moving_d).sqrt();
                if denom >= tol {
                    let num = corr_ft_ref[i] - mask_corr_f_ref[i] * mask_corr_m_ref[i] / ov_div;
                    return (num / denom).clamp(-1.0, 1.0);
                }
            }
            0.0
        });

        Ok(build_output(out, dims, fixed))
    }
}

/// Build the (possibly larger) output image with the fixed image's spacing.
fn build_output<B: Backend>(
    values: Vec<f32>,
    dims: [usize; 3],
    fixed: &Image<B, 3>,
) -> Image<B, 3> {
    use burn::tensor::{Shape, Tensor, TensorData};
    let device = fixed.data().device();
    let td = TensorData::new(values, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    let sp = fixed.spacing();
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([sp[0], sp[1], sp[2]]),
        Direction::identity(),
    )
}

#[cfg(test)]
#[path = "tests_masked_fft_correlation.rs"]
mod tests_masked_fft_correlation;

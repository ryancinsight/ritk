//! Normalized correlation of an image with a template
//! (`itk::NormalizedCorrelationImageFilter` / `sitk.NormalizedCorrelation`).
//!
//! # Mathematical Specification
//!
//! The template `T` (a small image used as a neighbourhood operator) is first
//! normalized to mean zero and unit norm:
//!
//! ```text
//! mean = Σ T / N,  var = (Σ T² − (Σ T)²/N)/(N−1),  k = √var · √(N−1)
//! nt[i] = (T[i] − mean) / k          (⇒ Σ nt = 0, ‖nt‖ = 1)
//! ```
//!
//! Then for each image voxel `x` whose mask value is non-zero, over the template
//! window (ZeroFluxNeumann boundary):
//!
//! ```text
//! out(x) = Σ_i I(x+i)·nt[i] / √( Σ_i I(x+i)² − (Σ_i I(x+i))² / N )
//! ```
//!
//! which is the correlation of the locally-centered image neighbourhood with the
//! unit template. Masked-out voxels are 0. Internal arithmetic is `f64`
//! (ITK's `RealType` for a float image), so the result is float-exact to
//! SimpleITK.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Compute the normalized correlation of `image` with `template`, gated by `mask`.
///
/// All three are `[z, y, x]` scalar images; `mask` and `image` share a shape,
/// `template` is the (odd-sized) neighbourhood operator. Returns `Err` on a
/// shape mismatch or an even template extent.
pub fn normalized_correlation<B: Backend>(
    image: &Image<B, 3>,
    mask: &Image<B, 3>,
    template: &Image<B, 3>,
) -> Result<Image<B, 3>> {
    let (img, dims) = extract_vec_infallible(image);
    let (msk, mdims) = extract_vec_infallible(mask);
    let (tpl, tdims) = extract_vec_infallible(template);
    if mdims != dims {
        bail!("normalized_correlation: mask shape {mdims:?} != image shape {dims:?}");
    }
    let [nz, ny, nx] = dims;
    let [tz, ty, tx] = tdims;
    if tz % 2 == 0 || ty % 2 == 0 || tx % 2 == 0 {
        bail!("normalized_correlation: template extents must be odd, got {tdims:?}");
    }
    let (rz, ry, rx) = (tz / 2, ty / 2, tx / 2);
    let num = (tz * ty * tx) as f64;

    // Normalize the template to mean zero, unit norm.
    let sum: f64 = tpl.iter().map(|&v| v as f64).sum();
    let sum_sq: f64 = tpl.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum / num;
    let var = (sum_sq - sum * sum / num) / (num - 1.0);
    let k = var.sqrt() * (num - 1.0).sqrt();
    let nt: Vec<f64> = tpl.iter().map(|&v| (v as f64 - mean) / k).collect();

    let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
    let clamp = |v: i64, hi: usize| -> usize { v.max(0).min(hi as i64 - 1) as usize };

    let out: Vec<f32> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny * nx, |i| {
        if msk[i] == 0.0 {
            return 0.0f32;
        }
        let z = i / (ny * nx);
        let y = (i % (ny * nx)) / nx;
        let x = i % nx;
        let mut numerator = 0.0f64;
        let mut s = 0.0f64;
        let mut sq = 0.0f64;
        let mut ti = 0usize;
        for dz in 0..tz {
            let zz = clamp(z as i64 + dz as i64 - rz as i64, nz);
            for dy in 0..ty {
                let yy = clamp(y as i64 + dy as i64 - ry as i64, ny);
                for dx in 0..tx {
                    let xx = clamp(x as i64 + dx as i64 - rx as i64, nx);
                    let v = img[idx(zz, yy, xx)] as f64;
                    numerator += v * nt[ti];
                    s += v;
                    sq += v * v;
                    ti += 1;
                }
            }
        }
        let denom = (sq - s * s / num).sqrt();
        if denom != 0.0 {
            (numerator / denom) as f32
        } else {
            0.0
        }
    });

    Ok(rebuild(out, dims, image))
}

#[cfg(test)]
#[path = "tests_normalized_correlation.rs"]
mod tests_normalized_correlation;

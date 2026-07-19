//! Box sigma (sliding standard-deviation) filter (`itk::BoxSigmaImageFilter`).
//!
//! # Mathematical Specification
//!
//! For per-axis radii `[rz, ry, rx]`, each voxel is replaced by the **sample**
//! standard deviation (Bessel-corrected, divisor `n âˆ’ 1`) over the axis-aligned
//! `(2r+1)` window clipped to the image bounds:
//!
//! ```text
//! W   = ([zâˆ’rz, z+rz] Ã— [yâˆ’ry, y+ry] Ã— [xâˆ’rx, x+rx]) âˆ© image,  n = |W|
//! out = sqrt( (Î£_{kâˆˆW} I(k)Â² âˆ’ (Î£_{kâˆˆW} I(k))Â² / n) / (n âˆ’ 1) )
//! ```
//!
//! Windows with `n â‰¤ 1` yield `0`. Matches `itk::BoxSigmaImageFilter` /
//! `sitk.BoxSigma`, which uses the sample (not population) divisor â€” pinned by a
//! probe: `[10,20,30,40,50]` r=1 â†’ interior `[20,30,40]` gives `10` (sample),
//! not `8.165` (population); the clipped boundary window `[10,20]` gives
//! `7.071`. Shares the clipped-window/shrink-boundary convention with
//! [`super::box_mean::BoxMeanImageFilter`].

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Box sigma filter â€” clipped-window sample standard deviation
/// (ITK `BoxSigmaImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct BoxSigmaImageFilter {
    /// Per-axis radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
}

impl BoxSigmaImageFilter {
    /// Construct with the given per-axis radii.
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Apply the box sigma to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;

        // Per-voxel independent (each output reads only its clipped window), so
        // the grid fans out across threads; the result is bitwise identical to a
        // serial run.
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |flat| {
                let z = flat / (ny * nx);
                let rem = flat % (ny * nx);
                let y = rem / nx;
                let x = rem % nx;
                let z0 = z.saturating_sub(rz);
                let z1 = (z + rz).min(nz - 1);
                let y0 = y.saturating_sub(ry);
                let y1 = (y + ry).min(ny - 1);
                let x0 = x.saturating_sub(rx);
                let x1 = (x + rx).min(nx - 1);
                let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
                for kz in z0..=z1 {
                    for ky in y0..=y1 {
                        let base = (kz * ny + ky) * nx;
                        for kx in x0..=x1 {
                            let v = vals[base + kx] as f64;
                            sum += v;
                            sumsq += v * v;
                        }
                    }
                }
                let n = ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
                if n > 1.0 {
                    let var = (sumsq - sum * sum / n) / (n - 1.0);
                    var.max(0.0).sqrt() as f32
                } else {
                    0.0
                }
            });
        rebuild(out, dims, image)
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;

        // Per-voxel independent (each output reads only its clipped window), so
        // the grid fans out across threads; the result is bitwise identical to a
        // serial run.
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |flat| {
                let z = flat / (ny * nx);
                let rem = flat % (ny * nx);
                let y = rem / nx;
                let x = rem % nx;
                let z0 = z.saturating_sub(rz);
                let z1 = (z + rz).min(nz - 1);
                let y0 = y.saturating_sub(ry);
                let y1 = (y + ry).min(ny - 1);
                let x0 = x.saturating_sub(rx);
                let x1 = (x + rx).min(nx - 1);
                let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
                for kz in z0..=z1 {
                    for ky in y0..=y1 {
                        let base = (kz * ny + ky) * nx;
                        for kx in x0..=x1 {
                            let v = vals[base + kx] as f64;
                            sum += v;
                            sumsq += v * v;
                        }
                    }
                }
                let n = ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
                if n > 1.0 {
                    let var = (sumsq - sum * sum / n) / (n - 1.0);
                    var.max(0.0).sqrt() as f32
                } else {
                    0.0
                }
            });
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_box_sigma.rs"]
mod tests_box_sigma;

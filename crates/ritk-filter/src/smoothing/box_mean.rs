//! Box mean filter (`itk::BoxMeanImageFilter`).
//!
//! # Mathematical Specification
//!
//! For per-axis radii `[rz, ry, rx]`, each voxel is replaced by the arithmetic
//! mean over the axis-aligned `(2r+1)` window **clipped to the image bounds**,
//! dividing by the number of in-bounds voxels actually summed:
//!
//! ```text
//! out(z,y,x) = (1/|W|) Â· Î£_{(k)âˆˆW} I(k),
//! W = [zâˆ’rz, z+rz] Ã— [yâˆ’ry, y+ry] Ã— [xâˆ’rx, x+rx]  âˆ©  image
//! ```
//!
//! # Distinction from [`super::mean::MeanImageFilter`]
//!
//! ITK `MeanImageFilter` / `sitk.Mean` uses ZeroFluxNeumann boundaries â€” it
//! clamps out-of-bounds neighbours to the edge voxel and divides by the **full**
//! `(2r+1)^D` window. `BoxMeanImageFilter` instead **shrinks** the window at the
//! border and divides by the actual in-bounds count. The two agree on the
//! interior but differ on the boundary, e.g. for `[10,20,30,40,50]` with `r=1`:
//! `BoxMean[0] = (10+20)/2 = 15` vs `Mean[0] = (10+10+20)/3 = 13.33`. Pinned by a
//! `sitk.BoxMean` probe.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Box mean filter â€” clipped-window average (ITK `BoxMeanImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct BoxMeanImageFilter {
    /// Per-axis radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
}

impl BoxMeanImageFilter {
    /// Construct with the given per-axis radii.
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Apply the box mean to a 3-D image.
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
                let mut sum = 0.0f64;
                for kz in z0..=z1 {
                    for ky in y0..=y1 {
                        let base = (kz * ny + ky) * nx;
                        for kx in x0..=x1 {
                            sum += vals[base + kx] as f64;
                        }
                    }
                }
                let count = ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
                (sum / count) as f32
            });
        rebuild(out, dims, image)
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
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
                let mut sum = 0.0f64;
                for kz in z0..=z1 {
                    for ky in y0..=y1 {
                        let base = (kz * ny + ky) * nx;
                        for kx in x0..=x1 {
                            sum += vals[base + kx] as f64;
                        }
                    }
                }
                let count = ((z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
                (sum / count) as f32
            });
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_box_mean.rs"]
mod tests_box_mean;

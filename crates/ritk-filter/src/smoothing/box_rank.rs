//! Box rank filter (`itk::RankImageFilter` / `sitk.Rank`).
//!
//! # Mathematical Specification
//!
//! For per-axis radii `[rz, ry, rx]` and a rank fraction `rank ∈ [0, 1]`, each
//! voxel is replaced by the order statistic at index `floor(rank·(n−1))` of the
//! **sorted** `(2r+1)` window clipped to the image bounds (`n` = in-bounds
//! count):
//!
//! ```text
//! W = ([z−rz, z+rz] × [y−ry, y+ry] × [x−rx, x+rx]) ∩ image,  n = |W|
//! out(z,y,x) = sort(W)[ floor(rank·(n−1)) ]
//! ```
//!
//! `rank = 0.5` is the median, `0.0` the minimum, `1.0` the maximum. Verified
//! against `sitk.Rank`: `[10,20,30,40,50]` r=1, rank 0.5 → `[10,20,30,40,40]`
//! (boundary windows shrink to the in-bounds voxels), and the floor index is
//! pinned by `[10,20,30,40]` (n=4) rank 0.5 → `20` (`floor(0.5·3)=1`), not `30`.
//!
//! Distinct from the structuring-element [`crate::rank::PercentileFilter`],
//! which uses replicate (clamp) padding and indexes by the full SE size; this
//! box variant shrinks the window at the border like
//! [`super::box_mean::BoxMeanImageFilter`].

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Box rank filter — order statistic over a clipped window (ITK `RankImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct RankImageFilter {
    /// Per-axis radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
    /// Rank fraction in `[0, 1]` (`0.5` = median). ITK default `0.5`.
    pub rank: f64,
}

impl RankImageFilter {
    /// Construct with the given per-axis radii and rank fraction.
    pub fn new(radius: [usize; 3], rank: f64) -> Self {
        Self { radius, rank }
    }

    /// Apply the box rank filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let rank = self.rank.clamp(0.0, 1.0);
        let cap = (2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1);
        let mut window: Vec<f32> = Vec::with_capacity(cap);
        let mut out = vec![0.0f32; vals.len()];

        for z in 0..nz {
            let z0 = z.saturating_sub(rz);
            let z1 = (z + rz).min(nz - 1);
            for y in 0..ny {
                let y0 = y.saturating_sub(ry);
                let y1 = (y + ry).min(ny - 1);
                for x in 0..nx {
                    let x0 = x.saturating_sub(rx);
                    let x1 = (x + rx).min(nx - 1);
                    window.clear();
                    for kz in z0..=z1 {
                        for ky in y0..=y1 {
                            let base = (kz * ny + ky) * nx;
                            for kx in x0..=x1 {
                                window.push(vals[base + kx]);
                            }
                        }
                    }
                    window.sort_unstable_by(|a, b| a.total_cmp(b));
                    let n = window.len();
                    let idx = ((rank * (n - 1) as f64).floor() as usize).min(n - 1);
                    out[(z * ny + y) * nx + x] = window[idx];
                }
            }
        }
        rebuild(out, dims, image)
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(&self, image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,{
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let rank = self.rank.clamp(0.0, 1.0);
        let cap = (2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1);
        let mut window: Vec<f32> = Vec::with_capacity(cap);
        let mut out = vec![0.0f32; vals.len()];

        for z in 0..nz {
            let z0 = z.saturating_sub(rz);
            let z1 = (z + rz).min(nz - 1);
            for y in 0..ny {
                let y0 = y.saturating_sub(ry);
                let y1 = (y + ry).min(ny - 1);
                for x in 0..nx {
                    let x0 = x.saturating_sub(rx);
                    let x1 = (x + rx).min(nx - 1);
                    window.clear();
                    for kz in z0..=z1 {
                        for ky in y0..=y1 {
                            let base = (kz * ny + ky) * nx;
                            for kx in x0..=x1 {
                                window.push(vals[base + kx]);
                            }
                        }
                    }
                    window.sort_unstable_by(|a, b| a.total_cmp(b));
                    let n = window.len();
                    let idx = ((rank * (n - 1) as f64).floor() as usize).min(n - 1);
                    out[(z * ny + y) * nx + x] = window[idx];
                }
            }
        }
        crate::native_support::rebuild_image(out, dims, image, backend)
    
    }

}

#[cfg(test)]
#[path = "tests_box_rank.rs"]
mod tests_box_rank;

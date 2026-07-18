//! Mean (box) smoothing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I` with voxel spacing `(sz, sy, sx)`, the mean filter at
//! voxel `(iz, iy, ix)` computes the arithmetic mean over the cubic
//! neighbourhood of half-width `radius`:
//!
//! ```text
//! M(iz, iy, ix) = (1 / |N|) Â· Î£_{(kz,ky,kx)âˆˆN(iz,iy,ix)} I(kz, ky, kx)
//! ```
//!
//! where `N(p)` is the set of voxels within `radius` steps in each axis
//! direction, clamped to the image bounds (replicate padding).
//!
//! # Neighbourhood cardinality
//!
//! ```text
//! |N| = (min(iz, r)âˆ’max(izâˆ’r, 0) + ... = (wz)(wy)(wx)
//! ```
//! where `wz = min(iz+r, Nz-1) âˆ’ max(izâˆ’r, 0) + 1`, etc. This accounts for
//! boundary voxels that have fewer neighbours.
//!
//! # ITK parity
//!
//! Corresponds to `itk::MeanImageFilter<InputImageType, OutputImageType>`.
//! ITK default radius = 1 (3Ã—3Ã—3 kernel). `radius = 0` is the identity.
//!
//! # Complexity
//!
//! O(N Â· (2r+1)Â³) â€” a separable integral-image approach would be O(N) per
//! radius, but (2r+1)Â³ â‰¤ 125 for default `r=1`, so the direct approach
//! matches expected workload. Fanned out over the flat voxel index (moirai).
//!
//! # Reference
//!
//! - Gonzalez, R.C. & Woods, R.E. (2008). *Digital Image Processing*, 3rd ed.
//!   Â§3.5.1 Smoothing Linear Filters.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Mean (box) smoothing filter.
///
/// Replaces each voxel with the arithmetic mean of its
/// `(2Â·radius+1)Â³` cubic neighbourhood.
/// `radius = 0` is the identity transform.
#[derive(Debug, Clone)]
pub struct MeanImageFilter {
    /// Half-width of the cubic neighbourhood in voxels. Default 1.
    pub radius: usize,
}

impl MeanImageFilter {
    /// Construct with the given neighbourhood radius (in voxels).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl Default for MeanImageFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

impl MeanImageFilter {
    /// Apply the mean filter to a 3-D image.
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let out = self.mean_values(&vals_vec, dims);
        Ok(rebuild(out, dims, image))
    }

    /// Apply the mean filter to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = self.mean_values(image.data_slice()?, image.shape());
        ritk_image::native::Image::from_flat_on(
            values,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn mean_values(&self, values: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;

        let r = self.radius;

        // identity shortcut
        if r == 0 || nz == 0 || ny == 0 || nx == 0 {
            return values.to_vec();
        }

        // Boundary: ITK MeanImageFilter uses a ZeroFluxNeumann (edge-replicate)
        // neighbourhood â€” the window is always the full (2r+1)Â³ samples with
        // out-of-bounds positions clamped to the nearest edge, and the average
        // divides by the full count. (A shrinking window with a smaller divisor
        // gives different boundary values; the interior is unaffected.)
        let ri = r as isize;
        let (nzi, nyi, nxi) = (nz as isize, ny as isize, nx as isize);
        let count = ((2 * r + 1) * (2 * r + 1) * (2 * r + 1)) as f64;
        // Fan out over the flat voxel index directly: one output allocation, no
        // per-slice intermediate `Vec`s. Each output voxel reads only its clamped
        // window, so the result is bitwise identical to a serial run.
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |flat| {
                let iz = flat / (ny * nx);
                let rem = flat % (ny * nx);
                let iy = rem / nx;
                let ix = rem % nx;
                let mut sum = 0.0f64;
                for kz in -ri..=ri {
                    let zc = (iz as isize + kz).clamp(0, nzi - 1) as usize;
                    for ky in -ri..=ri {
                        let yc = (iy as isize + ky).clamp(0, nyi - 1) as usize;
                        for kx in -ri..=ri {
                            let xc = (ix as isize + kx).clamp(0, nxi - 1) as usize;
                            sum += values[zc * ny * nx + yc * nx + xc] as f64;
                        }
                    }
                }
                (sum / count) as f32
            });

        out
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_mean.rs"]
mod tests_mean;

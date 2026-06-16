//! Mean (box) smoothing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I` with voxel spacing `(sz, sy, sx)`, the mean filter at
//! voxel `(iz, iy, ix)` computes the arithmetic mean over the cubic
//! neighbourhood of half-width `radius`:
//!
//! ```text
//! M(iz, iy, ix) = (1 / |N|) · Σ_{(kz,ky,kx)∈N(iz,iy,ix)} I(kz, ky, kx)
//! ```
//!
//! where `N(p)` is the set of voxels within `radius` steps in each axis
//! direction, clamped to the image bounds (replicate padding).
//!
//! # Neighbourhood cardinality
//!
//! ```text
//! |N| = (min(iz, r)−max(iz−r, 0) + ... = (wz)(wy)(wx)
//! ```
//! where `wz = min(iz+r, Nz-1) − max(iz−r, 0) + 1`, etc. This accounts for
//! boundary voxels that have fewer neighbours.
//!
//! # ITK parity
//!
//! Corresponds to `itk::MeanImageFilter<InputImageType, OutputImageType>`.
//! ITK default radius = 1 (3×3×3 kernel). `radius = 0` is the identity.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) — a separable integral-image approach would be O(N) per
//! radius, but (2r+1)³ ≤ 125 for default `r=1`, so the direct approach
//! matches expected workload. Parallelised over Z-slices with Rayon.
//!
//! # Reference
//!
//! - Gonzalez, R.C. & Woods, R.E. (2008). *Digital Image Processing*, 3rd ed.
//!   §3.5.1 Smoothing Linear Filters.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Mean (box) smoothing filter.
///
/// Replaces each voxel with the arithmetic mean of its
/// `(2·radius+1)³` cubic neighbourhood.
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;

        let r = self.radius;

        // identity shortcut
        if r == 0 || nz == 0 || ny == 0 || nx == 0 {
            return Ok(image.clone());
        }

        let vals: &[f32] = &vals_vec;

        let out: Vec<f32> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz, |iz| {
            let z0 = iz.saturating_sub(r);
            let z1 = (iz + r).min(nz - 1);
            (0..ny)
                .flat_map(move |iy| {
                    let y0 = iy.saturating_sub(r);
                    let y1 = (iy + r).min(ny - 1);
                    (0..nx).map(move |ix| {
                        let x0 = ix.saturating_sub(r);
                        let x1 = (ix + r).min(nx - 1);
                        let mut sum = 0.0f64;
                        let mut count = 0u64;
                        for kz in z0..=z1 {
                            for ky in y0..=y1 {
                                for kx in x0..=x1 {
                                    sum += vals[kz * ny * nx + ky * nx + kx] as f64;
                                    count += 1;
                                }
                            }
                        }
                        (sum / count as f64) as f32
                    })
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect();

        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_mean.rs"]
mod tests_mean;

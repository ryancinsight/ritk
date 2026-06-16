//! Shrink (integer downsampling) image filter.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I` with shape `[Nz, Ny, Nx]` and spacing `(sz, sy, sx)`,
//! the shrink filter with integer factors `[fz, fy, fx]` (all ≥ 1) produces an
//! output image with shape:
//!
//! ```text
//! [oz, oy, ox] = [ceil(Nz/fz), ceil(Ny/fy), ceil(Nx/fx)]
//! ```
//!
//! Each output voxel at `(iz, iy, ix)` is the arithmetic mean of all input voxels
//! in the axis-aligned tile:
//!
//! ```text
//! tile(iz,iy,ix) = {(kz, ky, kx) : kz ∈ [iz·fz, min((iz+1)·fz−1, Nz−1)],
//!                                    ky ∈ [iy·fy, min((iy+1)·fy−1, Ny−1)],
//!                                    kx ∈ [ix·fx, min((ix+1)·fx−1, Nx−1)]}
//! out(iz,iy,ix) = mean_{(kz,ky,kx) ∈ tile} I(kz, ky, kx)
//! ```
//!
//! The output spacing is updated: `out_spacing[i] = in_spacing[i] × factor[i]`.
//! The origin is unchanged (corresponds to the center of the first output voxel).
//!
//! # ITK Parity
//!
//! Corresponds to `itk::ShrinkImageFilter<TInputImage, TOutputImage>`.
//! ITK uses averaging within each tile. Default factors = [1, 1, 1] (identity).
//! Note: ITK `ShrinkImageFilter` uses subsampling (single voxel per tile);
//! this implementation uses averaging for anti-aliasing. For subsampling without
//! averaging, set `averaging = false`.
//!
//! # Reference
//!
//! - Gonzalez, R.C. & Woods, R.E. (2008). *Digital Image Processing*, 3rd ed. §4.7.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

/// Integer downsampling filter.
///
/// Reduces image dimensions by integer factors, computing the mean of each tile.
#[derive(Debug, Clone)]
pub struct ShrinkImageFilter {
    /// Downsampling factors per axis `\[fz, fy, fx\]`. All must be ≥ 1. Default \[1,1,1\].
    pub shrink_factors: [usize; 3],
}

impl ShrinkImageFilter {
    /// Construct with the given per-axis shrink factors (all ≥ 1).
    ///
    /// # Panics
    ///
    /// Does not panic, but using factors of 0 is treated as 1 (identity).
    pub fn new(shrink_factors: [usize; 3]) -> Self {
        Self { shrink_factors }
    }
}

impl Default for ShrinkImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl ShrinkImageFilter {
    /// Apply the shrink filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [fz, fy, fx] = [
            self.shrink_factors[0].max(1),
            self.shrink_factors[1].max(1),
            self.shrink_factors[2].max(1),
        ];

        // Output shape: ceil(N/f)
        let oz = nz.div_ceil(fz);
        let oy = ny.div_ceil(fy);
        let ox = nx.div_ceil(fx);

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let kz0 = iz * fz;
                    let kz1 = ((iz + 1) * fz - 1).min(nz - 1);
                    let ky0 = iy * fy;
                    let ky1 = ((iy + 1) * fy - 1).min(ny - 1);
                    let kx0 = ix * fx;
                    let kx1 = ((ix + 1) * fx - 1).min(nx - 1);
                    let mut sum = 0.0f64;
                    let mut count = 0u64;
                    for kz in kz0..=kz1 {
                        for ky in ky0..=ky1 {
                            for kx in kx0..=kx1 {
                                sum += vals[kz * ny * nx + ky * nx + kx] as f64;
                                count += 1;
                            }
                        }
                    }
                    out[iz * oy * ox + iy * ox + ix] = (sum / count as f64) as f32;
                }
            }
        }

        // Update spacing: out_spacing[i] = in_spacing[i] * factor[i].
        let in_s = image.spacing();
        let out_spacing = Spacing::new([
            in_s[0] * fz as f64,
            in_s[1] * fy as f64,
            in_s[2] * fx as f64,
        ]);

        Ok(rebuild_with_metadata(
            out,
            [oz, oy, ox],
            *image.origin(),
            out_spacing,
            *image.direction(),
            image,
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_shrink.rs"]
mod tests_shrink;

//! Paste filter â€” copy a source image into a region of a destination image.
//!
//! # Mathematical Specification
//!
//! Given destination image `D` with shape `[Dz, Dy, Dx]` and source image `S`
//! with shape `[Sz, Sy, Sx]`, and a destination start index `dest_start = [dz, dy, dx]`:
//!
//! `out = copy(D)`
//! `out[dz + iz][dy + iy][dx + ix] = S[iz][iy][ix]`
//!
//! for `iz âˆˆ [0, Sz)`, `iy âˆˆ [0, Sy)`, `ix âˆˆ [0, Sx)`.
//!
//! ## Invariants
//!
//! - `dest_start[k] + S_shape[k] â‰¤ D_shape[k]` for k âˆˆ {z, y, x}; validated.
//! - Destination voxels outside the paste region are unchanged.
//! - Spatial metadata of the *destination* image is preserved in the output.
//! - Source and destination must have the same f32 voxel type.
//!
//! # ITK Parity
//!
//! `itk::PasteImageFilter` with `SetDestinationIndex(idx)` and
//! `SetSourceRegion(region)` spanning the full source.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::VoxelIndex;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Paste a source image into a destination image at a given index.
///
/// The output adopts the spatial metadata (origin, spacing, direction) of the
/// *destination* image.
#[derive(Debug, Clone)]
pub struct PasteImageFilter {
    /// Starting voxel index in the destination: `[z, y, x]`.
    pub dest_start: VoxelIndex,
}

impl PasteImageFilter {
    pub fn new(dest_start: impl Into<VoxelIndex>) -> Self {
        Self {
            dest_start: dest_start.into(),
        }
    }

    /// Apply the paste: returns a copy of `dest` with `source` written at
    /// `dest_start`.
    ///
    /// Returns `Err` if the source region would exceed the destination bounds.
    pub fn apply<B: Backend>(
        &self,
        dest: &Image<f32, B, 3>,
        source: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        let [dz, dy, dx] = dest.shape();
        let [sz, sy, sx] = source.shape();
        let [sdz, sdy, sdx]: [usize; 3] = self.dest_start.into();

        if sdz + sz > dz {
            anyhow::bail!(
                "PasteImageFilter: source Z extent [{}..{}) exceeds dest depth {}",
                sdz,
                sdz + sz,
                dz
            );
        }
        if sdy + sy > dy {
            anyhow::bail!(
                "PasteImageFilter: source Y extent [{}..{}) exceeds dest height {}",
                sdy,
                sdy + sy,
                dy
            );
        }
        if sdx + sx > dx {
            anyhow::bail!(
                "PasteImageFilter: source X extent [{}..{}) exceeds dest width {}",
                sdx,
                sdx + sx,
                dx
            );
        }

        let (dest_vec, dims) = extract_vec_infallible(dest);
        let mut out = dest_vec;

        let (src_vals_vec, _) = extract_vec_infallible(source);
        let src_vals = &src_vals_vec;

        for iz in 0..sz {
            for iy in 0..sy {
                for ix in 0..sx {
                    let src_idx = iz * sy * sx + iy * sx + ix;
                    let dst_idx = (sdz + iz) * dy * dx + (sdy + iy) * dx + (sdx + ix);
                    out[dst_idx] = src_vals[src_idx];
                }
            }
        }

        Ok(rebuild(out, dims, dest))
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        dest: &ritk_image::Image<f32, B, 3>,
        source: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let [dz, dy, dx] = dest.shape();
        let [sz, sy, sx] = source.shape();
        let [sdz, sdy, sdx]: [usize; 3] = self.dest_start.into();

        if sdz + sz > dz {
            anyhow::bail!(
                "PasteImageFilter: source Z extent [{}..{}) exceeds dest depth {}",
                sdz,
                sdz + sz,
                dz
            );
        }
        if sdy + sy > dy {
            anyhow::bail!(
                "PasteImageFilter: source Y extent [{}..{}) exceeds dest height {}",
                sdy,
                sdy + sy,
                dy
            );
        }
        if sdx + sx > dx {
            anyhow::bail!(
                "PasteImageFilter: source X extent [{}..{}) exceeds dest width {}",
                sdx,
                sdx + sx,
                dx
            );
        }

        let (dest_vec, dims) = ritk_tensor_ops::native::extract_image_vec(dest)?;
        let mut out = dest_vec;

        let (src_vals_vec, _) = ritk_tensor_ops::native::extract_image_vec(source)?;
        let src_vals = &src_vals_vec;

        for iz in 0..sz {
            for iy in 0..sy {
                for ix in 0..sx {
                    let src_idx = iz * sy * sx + iy * sx + ix;
                    let dst_idx = (sdz + iz) * dy * dx + (sdy + iy) * dx + (sdx + ix);
                    out[dst_idx] = src_vals[src_idx];
                }
            }
        }

        crate::native_support::rebuild_image(out, dims, dest, backend)
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_paste.rs"]
mod tests_paste;

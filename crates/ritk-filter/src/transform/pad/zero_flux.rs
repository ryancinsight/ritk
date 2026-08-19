//! Zero-flux Neumann padding over the nearest edge voxel.

use super::{updated_origin, Padding};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_origin};

/// Zero-flux Neumann (edge-replicate / clamp) padding filter.
///
/// Extends the image by repeating the nearest edge voxel — out-of-bounds index
/// `i` along an axis of length `n` reads `clamp(i, 0, n-1)`. Matches ITK
/// `ZeroFluxNeumannPadImageFilter` (`sitk.ZeroFluxNeumannPad`).
#[derive(Debug, Clone)]
pub struct ZeroFluxNeumannPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: Padding,
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: Padding,
}

impl ZeroFluxNeumannPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: Padding, pad_upper: Padding) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for ZeroFluxNeumannPadImageFilter {
    fn default() -> Self {
        Self::new(Padding::new([1, 1, 1]), Padding::new([1, 1, 1]))
    }
}

#[inline]
fn clamp_index(i: i64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    i.clamp(0, n as i64 - 1) as usize
}

impl ZeroFluxNeumannPadImageFilter {
    /// Apply the zero-flux Neumann (edge-replicate) pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = clamp_index(iz as i64 - lz as i64, nz);
                    let sy = clamp_index(iy as i64 - ly as i64, ny);
                    let sx = clamp_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        Ok(rebuild_with_origin(out, [oz, oy, ox], new_origin, image))
    }

    /// Coeus-native sister of [`ZeroFluxNeumannPadImageFilter::apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let vals = image.data_slice()?;
        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = clamp_index(iz as i64 - lz as i64, nz);
                    let sy = clamp_index(iy as i64 - ly as i64, ny);
                    let sx = clamp_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        ritk_image::Image::from_flat_on(
            out,
            [oz, oy, ox],
            new_origin,
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

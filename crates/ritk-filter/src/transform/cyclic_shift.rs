//! Cyclic (periodic) shift of a 3-D image.
//!
//! Rolls the image by an integer per-axis offset with wrap-around: the voxel at
//! output index `i` reads input index `(i − shift) mod n` along each axis, so
//! every voxel is preserved and merely repositioned (no interpolation, no data
//! loss). Matches ITK `CyclicShiftImageFilter` / `sitk.CyclicShift`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Cyclic-shift filter: rolls the image by `shift = [sz, sy, sx]` voxels with
/// periodic wrap-around.
#[derive(Debug, Clone, Copy)]
pub struct CyclicShiftImageFilter {
    /// Per-axis shift in voxels (`[z, y, x]`); may be negative.
    pub shift: [i64; 3],
}

impl CyclicShiftImageFilter {
    /// Create a cyclic-shift filter with the given per-axis (`[z, y, x]`) shift.
    pub fn new(shift: [i64; 3]) -> Self {
        Self { shift }
    }

    /// Apply the cyclic shift. Output has identical shape and spatial metadata.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        // Reduce each shift modulo the (positive) axis length.
        let rem = |s: i64, n: usize| -> usize {
            if n == 0 {
                return 0;
            }
            let n = n as i64;
            (((s % n) + n) % n) as usize
        };
        let (sz, sy, sx) = (
            rem(self.shift[0], nz),
            rem(self.shift[1], ny),
            rem(self.shift[2], nx),
        );

        let mut out = vec![0.0_f32; nz * ny * nx];
        for oz in 0..nz {
            let iz = if oz >= sz { oz - sz } else { oz + nz - sz };
            for oy in 0..ny {
                let iy = if oy >= sy { oy - sy } else { oy + ny - sy };
                for ox in 0..nx {
                    let ix = if ox >= sx { ox - sx } else { ox + nx - sx };
                    out[oz * ny * nx + oy * nx + ox] = vals[iz * ny * nx + iy * nx + ix];
                }
            }
        }
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
        // Reduce each shift modulo the (positive) axis length.
        let rem = |s: i64, n: usize| -> usize {
            if n == 0 {
                return 0;
            }
            let n = n as i64;
            (((s % n) + n) % n) as usize
        };
        let (sz, sy, sx) = (
            rem(self.shift[0], nz),
            rem(self.shift[1], ny),
            rem(self.shift[2], nx),
        );

        let mut out = vec![0.0_f32; nz * ny * nx];
        for oz in 0..nz {
            let iz = if oz >= sz { oz - sz } else { oz + nz - sz };
            for oy in 0..ny {
                let iy = if oy >= sy { oy - sy } else { oy + ny - sy };
                for ox in 0..nx {
                    let ix = if ox >= sx { ox - sx } else { ox + nx - sx };
                    out[oz * ny * nx + oy * nx + ox] = vals[iz * ny * nx + iy * nx + ix];
                }
            }
        }
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_cyclic_shift.rs"]
mod tests_cyclic_shift;

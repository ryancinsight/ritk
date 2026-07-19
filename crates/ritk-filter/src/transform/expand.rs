//! Integer-factor image expansion (upsampling) with linear interpolation.
//!
//! Matches ITK `ExpandImageFilter` / `sitk.Expand` (default linear interpolator):
//! the output grid has `size·factor` voxels and `spacing/factor` per axis, with
//! its origin shifted so the sample positions interleave the input grid:
//!
//! - `spacing_out = spacing / factor`
//! - `origin_out  = origin − ½·spacing + ½·spacing_out`
//!
//! Output voxel `j` along an axis reads the **continuous input index**
//! `ci(j) = (j + ½)/factor − ½` and is linearly interpolated with **edge-clamp**
//! boundary handling (positions outside `[0, n−1]` clamp to the nearest sample),
//! exactly as ITK's `LinearInterpolateImageFunction` behaves at the Expand grid.

use ritk_core::spatial::{Point, Spacing};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

/// Integer-factor expansion filter.
#[derive(Debug, Clone, Copy)]
pub struct ExpandImageFilter {
    /// Per-axis expansion factors `[fz, fy, fx]` (each ≥ 1).
    pub factors: [usize; 3],
}

impl ExpandImageFilter {
    /// Create an expansion filter with the given per-axis (`[z, y, x]`) factors.
    pub fn new(factors: [usize; 3]) -> Self {
        Self { factors }
    }

    /// Apply the expansion. Output is `[nz·fz, ny·fy, nx·fx]`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, [nz, ny, nx]) = extract_vec_infallible(image);
        let [fz, fy, fx] = self.factors;
        let (oz_n, oy_n, ox_n) = (nz * fz, ny * fy, nx * fx);

        // Continuous input index for output index `o` under factor `f`.
        let ci = |o: usize, f: usize| (o as f64 + 0.5) / f as f64 - 0.5;
        // Edge-clamped 1-D linear weights: (i0, i1, w0, w1).
        let lerp = |c: f64, n: usize| -> (usize, usize, f64, f64) {
            let i0f = c.floor();
            let frac = c - i0f;
            let i0 = (i0f as isize).clamp(0, n as isize - 1) as usize;
            let i1 = ((i0f as isize) + 1).clamp(0, n as isize - 1) as usize;
            (i0, i1, 1.0 - frac, frac)
        };

        let mut out = vec![0.0_f32; oz_n * oy_n * ox_n];
        for oz in 0..oz_n {
            let (z0, z1, wz0, wz1) = lerp(ci(oz, fz), nz);
            for oy in 0..oy_n {
                let (y0, y1, wy0, wy1) = lerp(ci(oy, fy), ny);
                for ox in 0..ox_n {
                    let (x0, x1, wx0, wx1) = lerp(ci(ox, fx), nx);
                    let at = |z: usize, y: usize, x: usize| vals[z * ny * nx + y * nx + x] as f64;
                    // Trilinear (separable) interpolation.
                    let v = wz0
                        * (wy0 * (wx0 * at(z0, y0, x0) + wx1 * at(z0, y0, x1))
                            + wy1 * (wx0 * at(z0, y1, x0) + wx1 * at(z0, y1, x1)))
                        + wz1
                            * (wy0 * (wx0 * at(z1, y0, x0) + wx1 * at(z1, y0, x1))
                                + wy1 * (wx0 * at(z1, y1, x0) + wx1 * at(z1, y1, x1)));
                    out[oz * oy_n * ox_n + oy * ox_n + ox] = v as f32;
                }
            }
        }

        // ITK Expand metadata: spacing/factor, origin shifted by half a voxel.
        let sp = image.spacing().to_array();
        let orig = image.origin().to_array();
        let mut sp_out = [0.0_f64; 3];
        let mut orig_out = [0.0_f64; 3];
        for d in 0..3 {
            sp_out[d] = sp[d] / self.factors[d] as f64;
            orig_out[d] = orig[d] - 0.5 * sp[d] + 0.5 * sp_out[d];
        }

        rebuild_with_metadata(
            out,
            [oz_n, oy_n, ox_n],
            Point::new(orig_out),
            Spacing::new(sp_out),
            *image.direction(),
            image,
        )
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
        let (vals, [nz, ny, nx]) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [fz, fy, fx] = self.factors;
        let (oz_n, oy_n, ox_n) = (nz * fz, ny * fy, nx * fx);

        // Continuous input index for output index `o` under factor `f`.
        let ci = |o: usize, f: usize| (o as f64 + 0.5) / f as f64 - 0.5;
        // Edge-clamped 1-D linear weights: (i0, i1, w0, w1).
        let lerp = |c: f64, n: usize| -> (usize, usize, f64, f64) {
            let i0f = c.floor();
            let frac = c - i0f;
            let i0 = (i0f as isize).clamp(0, n as isize - 1) as usize;
            let i1 = ((i0f as isize) + 1).clamp(0, n as isize - 1) as usize;
            (i0, i1, 1.0 - frac, frac)
        };

        let mut out = vec![0.0_f32; oz_n * oy_n * ox_n];
        for oz in 0..oz_n {
            let (z0, z1, wz0, wz1) = lerp(ci(oz, fz), nz);
            for oy in 0..oy_n {
                let (y0, y1, wy0, wy1) = lerp(ci(oy, fy), ny);
                for ox in 0..ox_n {
                    let (x0, x1, wx0, wx1) = lerp(ci(ox, fx), nx);
                    let at = |z: usize, y: usize, x: usize| vals[z * ny * nx + y * nx + x] as f64;
                    // Trilinear (separable) interpolation.
                    let v = wz0
                        * (wy0 * (wx0 * at(z0, y0, x0) + wx1 * at(z0, y0, x1))
                            + wy1 * (wx0 * at(z0, y1, x0) + wx1 * at(z0, y1, x1)))
                        + wz1
                            * (wy0 * (wx0 * at(z1, y0, x0) + wx1 * at(z1, y0, x1))
                                + wy1 * (wx0 * at(z1, y1, x0) + wx1 * at(z1, y1, x1)));
                    out[oz * oy_n * ox_n + oy * ox_n + ox] = v as f32;
                }
            }
        }

        // ITK Expand metadata: spacing/factor, origin shifted by half a voxel.
        let sp = image.spacing().to_array();
        let orig = image.origin().to_array();
        let mut sp_out = [0.0_f64; 3];
        let mut orig_out = [0.0_f64; 3];
        for d in 0..3 {
            sp_out[d] = sp[d] / self.factors[d] as f64;
            orig_out[d] = orig[d] - 0.5 * sp[d] + 0.5 * sp_out[d];
        }

        crate::native_support::rebuild_with_metadata(
            out,
            [oz_n, oy_n, ox_n],
            Point::new(orig_out),
            Spacing::new(sp_out),
            *image.direction(),
            image,
            backend,
        )
    }
}

#[cfg(test)]
#[path = "tests_expand.rs"]
mod tests_expand;

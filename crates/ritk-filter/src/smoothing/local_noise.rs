//! Local noise estimation filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Ports `itk::NoiseImageFilter`. Each voxel is replaced by the **sample**
//! standard deviation over the full `(2r+1)^3` neighborhood, evaluated under a
//! **ZeroFluxNeumann** boundary (out-of-bounds neighbours clamp to the edge
//! voxel, so the neighbourhood count `n` is constant everywhere):
//!
//! ```text
//! n   = ∏_a (2·r_a + 1)
//! out = sqrt( (Σ_k I(k)² − (Σ_k I(k))² / n) / (n − 1) )
//! ```
//!
//! Accumulation is in `f64` (ITK `InputRealType` for a floating-point input).
//!
//! # Boundary contrast with `BoxSigma`
//!
//! This differs from [`super::box_sigma::BoxSigmaImageFilter`] only at the
//! border: `BoxSigma` clips the window to the image (a smaller `n`), whereas
//! `NoiseImageFilter` keeps the full window and repeats the edge value
//! (ZeroFluxNeumann). Interior voxels (≥ `r` from every face) are identical.
//!
//! # ITK parity
//!
//! Corresponds to `itk::NoiseImageFilter` (`sitk.Noise`), default radius
//! `[1, 1, 1]`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Local noise (sample standard deviation) filter (`itk::NoiseImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct NoiseImageFilter {
    /// Per-axis radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
}

impl Default for NoiseImageFilter {
    fn default() -> Self {
        Self { radius: [1, 1, 1] }
    }
}

impl NoiseImageFilter {
    /// Construct with the given per-axis radii.
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Estimate the per-voxel local noise of a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let num = ((2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1)) as f64;
        let mut out = vec![0.0f32; vals.len()];
        if num <= 1.0 {
            // A degenerate single-voxel window has no sample variance.
            return rebuild(out, dims, image);
        }

        let clamp = |i: isize, hi: usize| -> usize { i.clamp(0, hi as isize - 1) as usize };

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
                    for dz in -(rz as isize)..=(rz as isize) {
                        let kz = clamp(z as isize + dz, nz);
                        for dy in -(ry as isize)..=(ry as isize) {
                            let ky = clamp(y as isize + dy, ny);
                            let base = (kz * ny + ky) * nx;
                            for dx in -(rx as isize)..=(rx as isize) {
                                let kx = clamp(x as isize + dx, nx);
                                let v = vals[base + kx] as f64;
                                sum += v;
                                sumsq += v * v;
                            }
                        }
                    }
                    let var = (sumsq - sum * sum / num) / (num - 1.0);
                    out[(z * ny + y) * nx + x] = var.max(0.0).sqrt() as f32;
                }
            }
        }
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_local_noise.rs"]
mod tests_local_noise;

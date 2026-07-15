//! Binary pruning (skeleton spur removal) filter.
//!
//! # Mathematical Specification
//!
//! Ports `itk::BinaryPruningImageFilter` — removes short spurs from a binary
//! skeleton. Operating on the `z`-plane of a `z = 1` image, the input is
//! binarized (`≠ 0 → 1`) and the image is swept `iteration` times in raster
//! order; **in place**, every foreground pixel whose 8-neighbour on-count is
//! below 2 (an endpoint or isolated pixel) is set to 0, so a deletion is visible
//! to later pixels within the same sweep (ZeroFluxNeumann boundary).
//!
//! ```text
//! genus = Σ_{8-neighbours} p
//! if center == 1 and genus < 2:  center ← 0
//! ```
//!
//! Pure binary topology (no floating point), so it is bit-exact to
//! `sitk.BinaryPruning`. Output is binary (`1.0`/`0.0`). ITK default
//! `iteration = 3`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Binary pruning filter (`itk::BinaryPruningImageFilter`, 2-D).
#[derive(Debug, Clone, Copy)]
pub struct BinaryPruningFilter {
    /// Number of pruning sweeps. ITK default `3`.
    pub iteration: usize,
}

impl Default for BinaryPruningFilter {
    fn default() -> Self {
        Self { iteration: 3 }
    }
}

impl BinaryPruningFilter {
    /// Construct with the given number of pruning iterations.
    pub fn new(iteration: usize) -> Self {
        Self { iteration }
    }

    /// Prune skeleton spurs from the `z`-plane(s) of a binary image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let mut img: Vec<u8> = vals.iter().map(|&v| u8::from(v != 0.0)).collect();
        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        for z in 0..nz {
            let plane = z * ny * nx;
            for _ in 0..self.iteration {
                // Raster-order, in-place sweep: a pruned pixel is immediately
                // visible to later neighbours, matching ITK's NeighborhoodIterator.
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = plane + y * nx + x;
                        if img[idx] == 0 {
                            continue;
                        }
                        let yi = y as isize;
                        let xi = x as isize;
                        let g = |dy: isize, dx: isize| -> u8 {
                            img[plane + cl(yi + dy, ny) * nx + cl(xi + dx, nx)]
                        };
                        let genus = g(-1, -1)
                            + g(-1, 0)
                            + g(-1, 1)
                            + g(0, 1)
                            + g(1, 1)
                            + g(1, 0)
                            + g(1, -1)
                            + g(0, -1);
                        if genus < 2 {
                            img[idx] = 0;
                        }
                    }
                }
            }
        }

        let out: Vec<f32> = img.iter().map(|&v| v as f32).collect();
        rebuild(out, dims, image)
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(&self, image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,{
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let mut img: Vec<u8> = vals.iter().map(|&v| u8::from(v != 0.0)).collect();
        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        for z in 0..nz {
            let plane = z * ny * nx;
            for _ in 0..self.iteration {
                // Raster-order, in-place sweep: a pruned pixel is immediately
                // visible to later neighbours, matching ITK's NeighborhoodIterator.
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = plane + y * nx + x;
                        if img[idx] == 0 {
                            continue;
                        }
                        let yi = y as isize;
                        let xi = x as isize;
                        let g = |dy: isize, dx: isize| -> u8 {
                            img[plane + cl(yi + dy, ny) * nx + cl(xi + dx, nx)]
                        };
                        let genus = g(-1, -1)
                            + g(-1, 0)
                            + g(-1, 1)
                            + g(0, 1)
                            + g(1, 1)
                            + g(1, 0)
                            + g(1, -1)
                            + g(0, -1);
                        if genus < 2 {
                            img[idx] = 0;
                        }
                    }
                }
            }
        }

        let out: Vec<f32> = img.iter().map(|&v| v as f32).collect();
        crate::native_support::rebuild_image(out, dims, image, backend)
    
    }

}

#[cfg(test)]
#[path = "tests_binary_pruning.rs"]
mod tests_binary_pruning;

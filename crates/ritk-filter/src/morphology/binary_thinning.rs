//! Binary thinning (2-D skeletonization) filter.
//!
//! # Mathematical Specification
//!
//! Ports `itk::BinaryThinningImageFilter` â€” the Gonzalez & Woods iterative
//! thinning that reduces a binary object to a 1-pixel-wide 8-connected skeleton.
//! Operating on the single `z`-plane of a `z = 1` image, the input is binarized
//! (`â‰  0 â†’ 1`) and then, until no pixel changes, four sub-iterations sweep the
//! image; in each sweep every foreground pixel `p1` with 8-neighbours
//! `p2..p9` (clockwise from north, ZeroFluxNeumann boundary) is marked for
//! deletion when **all** hold, and marked pixels are removed only after the full
//! sweep:
//!
//! ```text
//! A: 1 < Î£ p_i < 7                          (not an endpoint / not interior)
//! B: (Î£ |p_{i+1} âˆ’ p_i|)/2 == 1             (exactly one 0â†’1 transition)
//! step 1: p4 == 0 âˆ¨ p6 == 0   step 2: p2 == 0 âˆ§ p8 == 0
//! step 3: p2 == 0 âˆ¨ p8 == 0   step 4: p4 == 0 âˆ§ p6 == 0
//! ```
//!
//! The output is binary (`1.0` skeleton, `0.0` background). The process is pure
//! binary topology (no floating point), so it is bit-exact to `sitk.BinaryThinning`.
//!
//! Neighbour indices follow Gonzalez & Woods (ITK `(x, y)` offsets mapped to the
//! `(y, x)` plane): `p2..p9` = N, NE, E, SE, S, SW, W, NW.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Binary thinning filter (`itk::BinaryThinningImageFilter`, 2-D).
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryThinningFilter;

impl BinaryThinningFilter {
    /// Construct the filter.
    pub fn new() -> Self {
        Self
    }

    /// Thin the single `z`-plane of a `z = 1` binary image to its skeleton.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        // ITK BinaryThinning is a 2-D filter; ritk represents 2-D as z = 1.
        // Each z-plane is thinned independently so higher z still produces a
        // defined (per-slice) result.
        let mut img: Vec<u8> = vals.iter().map(|&v| u8::from(v != 0.0)).collect();
        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        for z in 0..nz {
            let plane = z * ny * nx;
            let mut changed = true;
            while changed {
                changed = false;
                for step in 1..=4 {
                    let mut to_delete: Vec<usize> = Vec::new();
                    for y in 0..ny {
                        for x in 0..nx {
                            let idx = plane + y * nx + x;
                            if img[idx] != 1 {
                                continue;
                            }
                            let yi = y as isize;
                            let xi = x as isize;
                            let g = |dy: isize, dx: isize| -> u8 {
                                img[plane + cl(yi + dy, ny) * nx + cl(xi + dx, nx)]
                            };
                            // p2..p9: N, NE, E, SE, S, SW, W, NW.
                            let p2 = g(-1, 0);
                            let p3 = g(-1, 1);
                            let p4 = g(0, 1);
                            let p5 = g(1, 1);
                            let p6 = g(1, 0);
                            let p7 = g(1, -1);
                            let p8 = g(0, -1);
                            let p9 = g(-1, -1);

                            let num = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                            let test_a = num > 1 && num < 7;
                            let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
                            let transitions = (d(p3, p2)
                                + d(p4, p3)
                                + d(p5, p4)
                                + d(p6, p5)
                                + d(p7, p6)
                                + d(p8, p7)
                                + d(p9, p8)
                                + d(p2, p9))
                                / 2;
                            let test_b = transitions == 1;
                            let test_cd = match step {
                                1 => p4 == 0 || p6 == 0,
                                2 => p2 == 0 && p8 == 0,
                                3 => p2 == 0 || p8 == 0,
                                _ => p4 == 0 && p6 == 0,
                            };
                            if test_a && test_b && test_cd {
                                to_delete.push(idx);
                                changed = true;
                            }
                        }
                    }
                    for &idx in &to_delete {
                        img[idx] = 0;
                    }
                }
            }
        }

        let out: Vec<f32> = img.iter().map(|&v| v as f32).collect();
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
        // ITK BinaryThinning is a 2-D filter; ritk represents 2-D as z = 1.
        // Each z-plane is thinned independently so higher z still produces a
        // defined (per-slice) result.
        let mut img: Vec<u8> = vals.iter().map(|&v| u8::from(v != 0.0)).collect();
        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };

        for z in 0..nz {
            let plane = z * ny * nx;
            let mut changed = true;
            while changed {
                changed = false;
                for step in 1..=4 {
                    let mut to_delete: Vec<usize> = Vec::new();
                    for y in 0..ny {
                        for x in 0..nx {
                            let idx = plane + y * nx + x;
                            if img[idx] != 1 {
                                continue;
                            }
                            let yi = y as isize;
                            let xi = x as isize;
                            let g = |dy: isize, dx: isize| -> u8 {
                                img[plane + cl(yi + dy, ny) * nx + cl(xi + dx, nx)]
                            };
                            // p2..p9: N, NE, E, SE, S, SW, W, NW.
                            let p2 = g(-1, 0);
                            let p3 = g(-1, 1);
                            let p4 = g(0, 1);
                            let p5 = g(1, 1);
                            let p6 = g(1, 0);
                            let p7 = g(1, -1);
                            let p8 = g(0, -1);
                            let p9 = g(-1, -1);

                            let num = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                            let test_a = num > 1 && num < 7;
                            let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
                            let transitions = (d(p3, p2)
                                + d(p4, p3)
                                + d(p5, p4)
                                + d(p6, p5)
                                + d(p7, p6)
                                + d(p8, p7)
                                + d(p9, p8)
                                + d(p2, p9))
                                / 2;
                            let test_b = transitions == 1;
                            let test_cd = match step {
                                1 => p4 == 0 || p6 == 0,
                                2 => p2 == 0 && p8 == 0,
                                3 => p2 == 0 || p8 == 0,
                                _ => p4 == 0 && p6 == 0,
                            };
                            if test_a && test_b && test_cd {
                                to_delete.push(idx);
                                changed = true;
                            }
                        }
                    }
                    for &idx in &to_delete {
                        img[idx] = 0;
                    }
                }
            }
        }

        let out: Vec<f32> = img.iter().map(|&v| v as f32).collect();
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_binary_thinning.rs"]
mod tests_binary_thinning;

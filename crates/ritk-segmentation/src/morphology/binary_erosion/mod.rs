//! Binary erosion morphological operation.
//!
//! # Mathematical Specification
//!
//! Binary erosion with a box structuring element of half-width `radius` r:
//!
//! (M âŠ– B)(p) = 1  iff  âˆ€q âˆˆ N_r(p): M(q) = 1
//!
//! where N_r(p) is the set of voxels within Chebyshev distance r of p
//! (the axis-aligned hypercube of side 2r+1 centred at p).
//!
//! Out-of-bounds neighbours are treated as **foreground** (the structuring
//! element is clipped at the image boundary), matching ITK's
//! `BinaryErodeImageFilter` default (`BoundaryToForeground = true`): a foreground
//! voxel is not eroded merely for lying against the image edge, and a degenerate
//! `z = 1` (2-D promoted) volume erodes purely in-plane rather than vanishing.
//! This is the dual of dilation, whose out-of-bounds neighbours are background.
//!
//! # Complexity
//! O(n Â· (2r+1)^D) where n is the total voxel count.
//!
//! # Supported dimensionalities
//! D = 1, 2, 3.  For D outside this set the function panics with a clear message.

use ritk_core::image::Image;
use ritk_image::tensor::{Backend, Tensor};
use ritk_tensor_ops::extract_vec_infallible;

/// Binary erosion with a box structuring element of half-width `radius` voxels.
///
/// For each voxel p, output\[p\] = 1.0 iff every voxel within the axis-aligned
/// hypercube of half-width `radius` centred at p is foreground (value > 0.5).
///
/// Out-of-bounds neighbours are treated as foreground (structuring element
/// clipped to the image), matching ITK's `BinaryErodeImageFilter`.
pub struct BinaryErosion {
    /// Half-width of the box structuring element in voxels.
    /// Radius 0 â†’ structuring element = {p} â†’ erosion is the identity.
    /// Radius 1 â†’ 3^D neighbourhood.
    pub radius: usize,
}

impl BinaryErosion {
    /// Create a `BinaryErosion` with the given structuring-element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply erosion to a binary mask image.
    ///
    /// Supports D = 1, 2, 3. Panics for other dimensionalities.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<f32, B, D>) -> Image<f32, B, D> {
        let shape: [usize; D] = mask.shape();
        let device = B::default();
        let (flat_vals, _shape) = extract_vec_infallible(mask);
        let flat: &[f32] = &flat_vals;
        let output = erode(flat, &shape, self.radius);
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
    }

    /// Apply erosion to a Coeus-native binary mask image.
    ///
    /// # Errors
    ///
    /// Returns an error when the mask tensor is not host-addressable/contiguous
    /// or the native output image cannot be constructed.
    pub fn apply_native<B, const D: usize>(
        &self,
        mask: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            erode(mask.data_slice()?, &mask.shape(), self.radius),
            mask.shape(),
            *mask.origin(),
            *mask.spacing(),
            *mask.direction(),
            backend,
        )
    }
}

impl Default for BinaryErosion {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for BinaryErosion {
    fn apply(&self, mask: &Image<f32, B, D>) -> Image<f32, B, D> {
        self.apply(mask)
    }
}

// â”€â”€ Core CPU-side erosion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Apply binary erosion on a flat row-major array for shapes of rank 1, 2, or 3.
///
/// # Panics
///
/// Panics when `shape` is not rank 1, 2, or 3, or when its product does not
/// match `flat.len()`.
pub fn erode(flat: &[f32], shape: &[usize], radius: usize) -> Vec<f32> {
    assert_eq!(
        shape.iter().product::<usize>(),
        flat.len(),
        "binary erosion shape product must match input length"
    );
    match shape.len() {
        1 => erode_line(flat, shape[0], radius),
        2 => erode_plane(flat, shape[0], shape[1], radius),
        3 => erode_volume(flat, shape[0], shape[1], shape[2], radius),
        d => panic!("BinaryErosion: unsupported dimensionality D={d}; only D=1,2,3 are supported"),
    }
}

// â”€â”€ D = 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn erode_line(flat: &[f32], nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nx];
    for ix in 0..nx {
        if flat[ix] <= super::FOREGROUND_THRESHOLD {
            continue;
        }
        let all_fg = ((-r)..=r).all(|dx| {
            let nb = ix as isize + dx;
            if nb < 0 || nb >= nx as isize {
                return true; // out-of-bounds â†’ foreground (no boundary erosion)
            }
            flat[nb as usize] > super::FOREGROUND_THRESHOLD
        });
        if all_fg {
            output[ix] = 1.0;
        }
    }
    output
}

// â”€â”€ D = 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn erode_plane(flat: &[f32], ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; ny * nx];
    for iy in 0..ny {
        for ix in 0..nx {
            let center = iy * nx + ix;
            if flat[center] <= super::FOREGROUND_THRESHOLD {
                continue;
            }
            let all_fg = 'outer: {
                for dy in (-r)..=r {
                    for dx in (-r)..=r {
                        let ny_i = iy as isize + dy;
                        let nx_i = ix as isize + dx;
                        if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                            continue; // out-of-bounds â†’ foreground (clip SE to image)
                        }
                        if flat[ny_i as usize * nx + nx_i as usize] <= super::FOREGROUND_THRESHOLD {
                            break 'outer false;
                        }
                    }
                }
                true
            };
            if all_fg {
                output[center] = 1.0;
            }
        }
    }
    output
}

// â”€â”€ D = 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn erode_volume(flat: &[f32], nz: usize, ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let center = iz * ny * nx + iy * nx + ix;
                if flat[center] <= super::FOREGROUND_THRESHOLD {
                    continue;
                }
                let all_fg = 'outer: {
                    for dz in (-r)..=r {
                        for dy in (-r)..=r {
                            for dx in (-r)..=r {
                                let nz_i = iz as isize + dz;
                                let ny_i = iy as isize + dy;
                                let nx_i = ix as isize + dx;
                                if nz_i < 0
                                    || nz_i >= nz as isize
                                    || ny_i < 0
                                    || ny_i >= ny as isize
                                    || nx_i < 0
                                    || nx_i >= nx as isize
                                {
                                    continue; // out-of-bounds â†’ foreground (clip SE to image)
                                }
                                let nb =
                                    nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                                if flat[nb] <= super::FOREGROUND_THRESHOLD {
                                    break 'outer false;
                                }
                            }
                        }
                    }
                    true
                };
                if all_fg {
                    output[center] = 1.0;
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests;

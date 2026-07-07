//! Binary dilation morphological operation.
//!
//! # Mathematical Specification
//!
//! Binary dilation with a box structuring element of half-width `radius` r:
//!
//! (M ⊕ B)(p) = 1 iff ∃q ∈ N_r(p): M(q) = 1
//!
//! where N_r(p) is the set of voxels within Chebyshev distance r of p
//! (the axis-aligned hypercube of side 2r+1 centred at p).
//!
//! Out-of-bounds neighbours are ignored (treated as non-contributors), so the
//! structuring element is clipped at image boundaries.
//!
//! # Complexity
//!
//! O(n · (2r+1)^D) where n is the total voxel count.
//!
//! # Supported dimensionalities
//!
//! D = 1, 2, 3. For D outside this set the function panics with a clear message.

use ritk_core::image::Image;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_tensor_ops::extract_vec_infallible;

/// Binary dilation with a box structuring element of half-width `radius` voxels.
///
/// For each voxel p, output\[p\] = 1.0 iff at least one voxel within the
/// axis-aligned hypercube of half-width `radius` centred at p is foreground
/// (value > 0.5).
///
/// Out-of-bounds positions in the structuring element are skipped (they do not
/// contribute to dilation), so boundary voxels may still be set if an in-bounds
/// neighbour is foreground.
pub struct BinaryDilation {
    /// Half-width of the box structuring element in voxels.
    /// Radius 0 → structuring element = {p} → dilation is the identity.
    /// Radius 1 → 3^D neighbourhood.
    pub radius: usize,
}

impl BinaryDilation {
    /// Create a `BinaryDilation` with the given structuring-element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply dilation to a binary mask image.
    ///
    /// Supports D = 1, 2, 3. Panics for other dimensionalities.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<B, D>) -> Image<B, D> {
        let shape: [usize; D] = mask.shape();
        let device = mask.data().device();
        let (flat_vals, _shape) = extract_vec_infallible(mask);
        let flat: &[f32] = &flat_vals;
        let output = dilate_nd(flat, &shape, self.radius);
        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
    }
}

impl Default for BinaryDilation {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for BinaryDilation {
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        self.apply(mask)
    }
}

// ── Core CPU-side dilation ────────────────────────────────────────────────────

/// Apply binary dilation on a flat row-major array for shapes of rank 1, 2, or 3.
///
/// Panics for ranks other than 1, 2, 3.
pub(super) fn dilate_nd(flat: &[f32], shape: &[usize], radius: usize) -> Vec<f32> {
    match shape.len() {
        1 => dilate_line(flat, shape[0], radius),
        2 => dilate_plane(flat, shape[0], shape[1], radius),
        3 => dilate_volume(flat, shape[0], shape[1], shape[2], radius),
        d => panic!("BinaryDilation: unsupported dimensionality D={d}; only D=1,2,3 are supported"),
    }
}

// ── D = 1 ─────────────────────────────────────────────────────────────────────

fn dilate_line(flat: &[f32], nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nx];
    for (ix, out) in output.iter_mut().enumerate().take(nx) {
        let any_fg = ((-r)..=r).any(|dx| {
            let nb = ix as isize + dx;
            if nb < 0 || nb >= nx as isize {
                return false; // out-of-bounds → skip
            }
            flat[nb as usize] > super::FOREGROUND_THRESHOLD
        });
        if any_fg {
            *out = 1.0;
        }
    }
    output
}

// ── D = 2 ─────────────────────────────────────────────────────────────────────

fn dilate_plane(flat: &[f32], ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; ny * nx];
    for iy in 0..ny {
        for ix in 0..nx {
            let any_fg = 'outer: {
                for dy in (-r)..=r {
                    for dx in (-r)..=r {
                        let ny_i = iy as isize + dy;
                        let nx_i = ix as isize + dx;
                        if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                            continue; // out-of-bounds → skip
                        }
                        if flat[ny_i as usize * nx + nx_i as usize] > super::FOREGROUND_THRESHOLD {
                            break 'outer true;
                        }
                    }
                }
                false
            };
            if any_fg {
                output[iy * nx + ix] = 1.0;
            }
        }
    }
    output
}

// ── D = 3 ─────────────────────────────────────────────────────────────────────

fn dilate_volume(flat: &[f32], nz: usize, ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let any_fg = 'outer: {
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
                                    continue; // out-of-bounds → skip
                                }
                                let nb =
                                    nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                                if flat[nb] > super::FOREGROUND_THRESHOLD {
                                    break 'outer true;
                                }
                            }
                        }
                    }
                    false
                };
                if any_fg {
                    output[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests;

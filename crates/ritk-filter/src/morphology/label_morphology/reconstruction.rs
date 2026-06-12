//! Geodesic morphological reconstruction for grayscale f32 images.
//!
//! # Mathematical Specification
//!
//! **Dilation reconstruction** (Vincent 1993):
//!   Given marker M and mask I with M <= I:
//!   M* = lim_{k->inf} min(dilate_1(M_k), I)
//!   where dilate_1 is one-step dilation with the unit-radius cubic B_1.
//!
//! **Erosion reconstruction**:
//!   Given marker M and mask I with M >= I:
//!   M* = lim_{k->inf} max(erode_1(M_k), I)
//!
//! Convergence criterion: max_x |M_{k+1}(x) - M_k(x)| < 1e-5, or max_iter reached.
//!
//! # References
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image analysis.
//!   *IEEE Trans. Image Process.* 2(2):176-201.

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ════════════════════════════════════════════════════════════════════════════
// ReconstructionMode
// ════════════════════════════════════════════════════════════════════════════

/// Reconstruction mode for geodesic morphological reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Geodesic dilation: marker expands upward constrained by mask (M <= I).
    Dilation,
    /// Geodesic erosion: marker contracts downward constrained by mask (M >= I).
    Erosion,
}

// ════════════════════════════════════════════════════════════════════════════
// MorphologicalReconstruction
// ════════════════════════════════════════════════════════════════════════════

/// Geodesic morphological reconstruction for grayscale f32 images.
#[derive(Debug, Clone)]
pub struct MorphologicalReconstruction {
    pub mode: ReconstructionMode,
    pub max_iter: usize,
}

impl MorphologicalReconstruction {
    pub fn new(mode: ReconstructionMode) -> Self {
        Self {
            mode,
            max_iter: 200,
        }
    }

    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Apply geodesic reconstruction.
    ///
    /// # Arguments
    /// - `marker`: initial marker image; must have same shape as `mask`
    /// - `mask`: constraint mask image
    ///
    /// # Errors
    /// Returns `Err` when marker and mask shapes differ.
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let (marker_vals, dims) = extract_vec(marker)?;
        let (mask_vals, mask_dims) = extract_vec(mask)?;
        if dims != mask_dims {
            anyhow::bail!(
                "MorphologicalReconstruction: marker shape {:?} != mask shape {:?}",
                dims,
                mask_dims
            );
        }

        // Clamp marker to enforce M <= I (dilation) or M >= I (erosion)
        let mut current: Vec<f32> = match self.mode {
            ReconstructionMode::Dilation => marker_vals
                .iter()
                .zip(mask_vals.iter())
                .map(|(&m, &i)| m.min(i))
                .collect(),
            ReconstructionMode::Erosion => marker_vals
                .iter()
                .zip(mask_vals.iter())
                .map(|(&m, &i)| m.max(i))
                .collect(),
        };
        let mask_vec: Vec<f32> = mask_vals.to_vec();

        for _ in 0..self.max_iter {
            let next = match self.mode {
                ReconstructionMode::Dilation => {
                    let dilated = dilate1_scalar(&current, dims);
                    dilated
                        .iter()
                        .zip(mask_vec.iter())
                        .map(|(&d, &m)| d.min(m))
                        .collect::<Vec<f32>>()
                }
                ReconstructionMode::Erosion => {
                    let eroded = erode1_scalar(&current, dims);
                    eroded
                        .iter()
                        .zip(mask_vec.iter())
                        .map(|(&e, &m)| e.max(m))
                        .collect::<Vec<f32>>()
                }
            };

            // Convergence check
            let max_delta = current
                .iter()
                .zip(next.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            current = next;
            if max_delta < 1e-5 {
                break;
            }
        }

        let device = marker.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(current, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *marker.origin(),
            *marker.spacing(),
            *marker.direction(),
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Private neighbourhood helpers
// ════════════════════════════════════════════════════════════════════════════

/// One-step grayscale dilation (max in 3x3x3 neighbourhood, clamp padding).
fn dilate1_scalar(data: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(data.len());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut mx = f32::NEG_INFINITY;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let zz = (iz as i32 + dz).clamp(0, nz as i32 - 1) as usize;
                            let yy = (iy as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                            let xx = (ix as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                            let v = data[zz * ny * nx + yy * nx + xx];
                            if v > mx {
                                mx = v;
                            }
                        }
                    }
                }
                out.push(mx);
            }
        }
    }
    out
}

/// One-step grayscale erosion (min in 3x3x3 neighbourhood).
///
/// Out-of-bounds positions contribute f32::NEG_INFINITY so that boundary
/// voxels erode toward the mask value during geodesic erosion reconstruction.
/// This is mathematically required: on a finite-support domain the exterior
/// acts as a strict lower bound, enabling convergence from marker to mask.
fn erode1_scalar(data: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(data.len());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut mn = f32::INFINITY;
                'outer_e: for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let zz = iz as i32 + dz;
                            let yy = iy as i32 + dy;
                            let xx = ix as i32 + dx;
                            if zz < 0
                                || zz >= nz as i32
                                || yy < 0
                                || yy >= ny as i32
                                || xx < 0
                                || xx >= nx as i32
                            {
                                mn = f32::NEG_INFINITY;
                                break 'outer_e;
                            }
                            let v = data[zz as usize * ny * nx + yy as usize * nx + xx as usize];
                            if v < mn {
                                mn = v;
                            }
                        }
                    }
                }
                out.push(mn);
            }
        }
    }
    out
}

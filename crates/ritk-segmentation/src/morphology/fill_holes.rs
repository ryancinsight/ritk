//! Binary fill holes morphological operation.
//!
//! Fills enclosed background regions in a binary mask by flood-filling
//! from border voxels and inverting reachability.
//!
//! # Mathematical definition
//! Given binary mask M, let E = set of background voxels reachable from
//! any border voxel via 6-connected background paths.
//! Output(x) = M(x) if M(x)=1 OR x in E, else 1 (hole -> foreground).
//!
//! Only the boundary of a non-degenerate axis seeds the exterior set, so a z=1
//! (2-D promoted) volume fills its in-plane holes exactly like a 2-D image
//! rather than treating every voxel as a z-face border.

use super::MorphologicalOperation;
use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::VecDeque;

/// Fills enclosed background holes in a 3-D binary mask.
///
/// Uses 6-connected border flood-fill to identify exterior background voxels.
/// All background voxels not reachable from any border face are set to foreground.
pub struct BinaryFillHoles;

impl BinaryFillHoles {
    /// Apply hole filling to a Coeus-native binary mask.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite samples, inaccessible backend storage,
    /// or output construction failure.
    pub fn apply_native<B>(
        &self,
        mask: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = mask.data_slice()?;
        super::ensure_finite_mask(values)?;
        crate::native_output::from_values(mask, fill_holes_values(values, mask.shape()), backend)
    }
}

impl<B: Backend> MorphologicalOperation<B, 3> for BinaryFillHoles {
    fn apply(&self, mask: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let shape = mask.shape();
        let device = B::default();
        let (vals_vec, _shape) = extract_vec_infallible(mask);
        let out = fill_holes_values(&vals_vec, shape);
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &out, &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
            .expect("invariant: segmentation output tensor preserves the image rank")
    }
}

pub(crate) fn fill_holes_values(vals: &[f32], [nz, ny, nx]: [usize; 3]) -> Vec<f32> {
    let n = nz * ny * nx;
    let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
    let is_background = |v: f32| v < super::FOREGROUND_THRESHOLD;
    let mut reachable = vec![false; n];
    let mut queue = VecDeque::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let border = (nz > 1 && (iz == 0 || iz == nz - 1))
                    || (ny > 1 && (iy == 0 || iy == ny - 1))
                    || (nx > 1 && (ix == 0 || ix == nx - 1));
                let i = idx(iz, iy, ix);
                if border && is_background(vals[i]) && !reachable[i] {
                    reachable[i] = true;
                    queue.push_back((iz, iy, ix));
                }
            }
        }
    }
    const NEIGHBORS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    while let Some((iz, iy, ix)) = queue.pop_front() {
        for (dz, dy, dx) in NEIGHBORS {
            let (zz, yy, xx) = (iz as isize + dz, iy as isize + dy, ix as isize + dx);
            if zz < 0 || yy < 0 || xx < 0 {
                continue;
            }
            let voxel = [zz as usize, yy as usize, xx as usize];
            if voxel[0] >= nz || voxel[1] >= ny || voxel[2] >= nx {
                continue;
            }
            let i = idx(voxel[0], voxel[1], voxel[2]);
            if !reachable[i] && is_background(vals[i]) {
                reachable[i] = true;
                queue.push_back((voxel[0], voxel[1], voxel[2]));
            }
        }
    }
    vals.iter()
        .enumerate()
        .map(|(i, &value)| {
            if is_background(value) && !reachable[i] {
                1.0
            } else {
                value
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "tests_fill_holes.rs"]
mod tests;

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
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::VecDeque;

/// Fills enclosed background holes in a 3-D binary mask.
///
/// Uses 6-connected border flood-fill to identify exterior background voxels.
/// All background voxels not reachable from any border face are set to foreground.
pub struct BinaryFillHoles;

impl<B: Backend> MorphologicalOperation<B, 3> for BinaryFillHoles {
    fn apply(&self, mask: &Image<B, 3>) -> Image<B, 3> {
        let shape = mask.shape();
        let [nz, ny, nx] = shape;
        let n = nz * ny * nx;
        let device = mask.data().device();

        let (vals_vec, _shape) = extract_vec_infallible(mask);
        let vals: &[f32] = &vals_vec;

        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
        let is_background = |v: f32| v < super::FOREGROUND_THRESHOLD;

        let mut reachable = vec![false; n];
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    // A voxel seeds the exterior flood-fill only if it lies on the
                    // boundary of a NON-degenerate axis. A size-1 axis (e.g. a z=1
                    // 2-D promoted volume) must not mark every voxel as a z-face
                    // border — that would treat interior holes as exterior and
                    // leave them unfilled.
                    let border = (nz > 1 && (iz == 0 || iz == nz - 1))
                        || (ny > 1 && (iy == 0 || iy == ny - 1))
                        || (nx > 1 && (ix == 0 || ix == nx - 1));
                    if border && is_background(vals[idx(iz, iy, ix)]) {
                        let i = idx(iz, iy, ix);
                        if !reachable[i] {
                            reachable[i] = true;
                            queue.push_back((iz, iy, ix));
                        }
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
                let nz_ = iz as isize + dz;
                let ny_ = iy as isize + dy;
                let nx_ = ix as isize + dx;
                if nz_ < 0 || ny_ < 0 || nx_ < 0 {
                    continue;
                }
                let (nz_u, ny_u, nx_u) = (nz_ as usize, ny_ as usize, nx_ as usize);
                if nz_u >= nz || ny_u >= ny || nx_u >= nx {
                    continue;
                }
                let ni = idx(nz_u, ny_u, nx_u);
                if !reachable[ni] && is_background(vals[ni]) {
                    reachable[ni] = true;
                    queue.push_back((nz_u, ny_u, nx_u));
                }
            }
        }

        let mut out: Vec<f32> = vals.to_vec();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let i = idx(iz, iy, ix);
                    if is_background(out[i]) && !reachable[i] {
                        out[i] = 1.0;
                    }
                }
            }
        }

        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(out, Shape::new([nz, ny, nx])), &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
    }
}

#[cfg(test)]
#[path = "tests_fill_holes.rs"]
mod tests;

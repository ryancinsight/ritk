//! Binary fill holes morphological operation.
//!
//! Fills enclosed background regions in a binary mask by flood-filling
//! from border voxels and inverting reachability.
//!
//! # Mathematical definition
//! Given binary mask M, let E = set of background voxels reachable from
//! any border voxel via 6-connected background paths.
//! Output(x) = M(x) if M(x)=1 OR x in E, else 1 (hole -> foreground).

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use crate::image::Image;
use super::MorphologicalOperation;
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

        let mask_data = mask.data().clone().into_data();
        let vals: &[f32] = mask_data.as_slice::<f32>().expect("f32 mask tensor data");

        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
        let is_background = |v: f32| v < 0.5_f32;

        let mut reachable = vec![false; n];
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let border = iz == 0 || iz == nz - 1 || iy == 0 || iy == ny - 1 || ix == 0 || ix == nx - 1;
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
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
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

        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(out, Shape::new([nz, ny, nx])),
            &device,
        );
        Image::new(
            tensor,
            mask.origin().clone(),
            mask.spacing().clone(),
            mask.direction().clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    type Backend = burn_ndarray::NdArray<f32>;

    fn make_mask(vals: Vec<f32>, shape: [usize; 3]) -> Image<Backend, 3> {
        let device = Default::default();
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(vals, Shape::new(shape)),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    #[test]
    fn test_fill_holes_solid_sphere_unchanged() {
        let shape = [5usize, 5, 5];
        let n = 125;
        let mut vals = vec![0.0f32; n];
        for iz in 0..5usize {
            for iy in 0..5usize {
                for ix in 0..5usize {
                    let d2 = ((iz as i32 - 2).pow(2)
                        + (iy as i32 - 2).pow(2)
                        + (ix as i32 - 2).pow(2)) as f32;
                    if d2 <= 1.0 {
                        vals[iz * 25 + iy * 5 + ix] = 1.0;
                    }
                }
            }
        }
        let mask = make_mask(vals.clone(), shape);
        let result = BinaryFillHoles.apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        assert_eq!(out_vals, vals.as_slice(), "solid sphere must be unchanged");
    }

    #[test]
    fn test_fill_holes_hollow_sphere_fills_interior() {
        let shape = [7usize, 7, 7];
        let n = 343;
        let mut vals = vec![0.0f32; n];
        for iz in 0..7usize {
            for iy in 0..7usize {
                for ix in 0..7usize {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 >= 4.0 && d2 <= 9.0 {
                        vals[iz * 49 + iy * 7 + ix] = 1.0;
                    }
                }
            }
        }
        let mask = make_mask(vals.clone(), shape);
        let result = BinaryFillHoles.apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        for iz in 0..7usize {
            for iy in 0..7usize {
                for ix in 0..7usize {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 < 4.0 {
                        assert_eq!(
                            out_vals[iz * 49 + iy * 7 + ix],
                            1.0,
                            "interior hole at ({},{},{}) must be filled",
                            iz, iy, ix
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fill_holes_all_zero_unchanged() {
        let shape = [3usize, 3, 3];
        let vals = vec![0.0f32; 27];
        let mask = make_mask(vals.clone(), shape);
        let result = BinaryFillHoles.apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        assert_eq!(out_vals, vals.as_slice(), "all-zero mask must be unchanged");
    }

    #[test]
    fn test_fill_holes_all_one_unchanged() {
        let shape = [3usize, 3, 3];
        let vals = vec![1.0f32; 27];
        let mask = make_mask(vals.clone(), shape);
        let result = BinaryFillHoles.apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        assert_eq!(out_vals, vals.as_slice(), "all-one mask must be unchanged");
    }

    #[test]
    fn test_fill_holes_output_shape_preserved() {
        let shape = [4usize, 5, 6];
        let mask = make_mask(vec![0.0f32; 120], shape);
        let result = BinaryFillHoles.apply(&mask);
        assert_eq!(result.shape(), shape);
    }
}

//! Binary morphological gradient.
//!
//! Extracts the boundary of a binary foreground region as the set-theoretic
//! difference between dilation and erosion:
//!
//!   MorphGradient(M) = Dilation(M) AND NOT Erosion(M)
//!
//! The result is 1.0 at boundary voxels (in dilation but not erosion) and
//! 0.0 at interior foreground, exterior background, and all other voxels.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use crate::image::Image;
use super::{BinaryDilation, BinaryErosion, MorphologicalOperation};

/// Extracts the morphological gradient (boundary) of a binary mask.
///
/// Output voxel x is 1.0 iff  and .
pub struct MorphologicalGradient {
    /// Ball radius for structuring element.
    pub radius: usize,
}

impl Default for MorphologicalGradient {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl MorphologicalGradient {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl<B: Backend> MorphologicalOperation<B, 3> for MorphologicalGradient {
    fn apply(&self, mask: &Image<B, 3>) -> Image<B, 3> {
        let dilated = BinaryDilation { radius: self.radius }.apply(mask);
        let eroded = BinaryErosion { radius: self.radius }.apply(mask);

        let shape = mask.shape();
        let [nz, ny, nx] = shape;
        let n = nz * ny * nx;
        let device = mask.data().device();

        let dil_data = dilated.data().clone().into_data();
        let dil_vals: &[f32] = dil_data.as_slice::<f32>().expect("f32 dilated data");

        let ero_data = eroded.data().clone().into_data();
        let ero_vals: &[f32] = ero_data.as_slice::<f32>().expect("f32 eroded data");

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            if dil_vals[i] >= 0.5 && ero_vals[i] < 0.5 {
                out[i] = 1.0;
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
        Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity())
    }

    #[test]
    fn test_gradient_all_zero_gives_all_zero() {
        let mask = make_mask(vec![0.0f32; 27], [3, 3, 3]);
        let result = MorphologicalGradient::new(1).apply(&mask);
        let result_data = result.data().clone().into_data();
        let vals = result_data.as_slice::<f32>().unwrap();
        assert!(vals.iter().all(|&v| v < 0.5), "all-zero input must produce all-zero gradient");
    }

    /// BinaryErosion treats out-of-bounds neighbors as background (0.0).
    /// Therefore, border voxels of an all-1 mask are eroded to 0, and the gradient
    /// is 1 there (dilation=1, erosion=0).  Only interior voxels -- those whose
    /// full r=1 neighborhood is within image bounds -- are invariant to this
    /// boundary effect and must have gradient 0.
    #[test]
    fn test_gradient_all_one_interior_voxels_zero() {
        // 5x5x5 all-one mask; interior indices 1..=3 in each axis have a complete r=1 cube.
        let mask = make_mask(vec![1.0f32; 125], [5, 5, 5]);
        let result = MorphologicalGradient::new(1).apply(&mask);
        let result_data = result.data().clone().into_data();
        let vals = result_data.as_slice::<f32>().unwrap();
        for iz in 1..=3usize {
            for iy in 1..=3usize {
                for ix in 1..=3usize {
                    let i = iz * 25 + iy * 5 + ix;
                    assert_eq!(
                        vals[i], 0.0,
                        "interior voxel ({},{},{}) must have zero gradient for all-one mask",
                        iz, iy, ix
                    );
                }
            }
        }
    }

    #[test]
    fn test_gradient_solid_ball_only_boundary_voxels_nonzero() {
        let shape = [7usize, 7, 7];
        let mut vals = vec![0.0f32; 343];
        for iz in 0..7usize {
            for iy in 0..7usize {
                for ix in 0..7usize {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 <= 4.0 {
                        vals[iz * 49 + iy * 7 + ix] = 1.0;
                    }
                }
            }
        }
        let mask = make_mask(vals.clone(), shape);
        let result = MorphologicalGradient::new(1).apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        let center_idx = 3 * 49 + 3 * 7 + 3;
        assert_eq!(out_vals[center_idx], 0.0, "center voxel must not be a boundary");
        let boundary_count = out_vals.iter().filter(|&&v| v >= 0.5).count();
        assert!(boundary_count > 0, "gradient must detect at least one boundary voxel");
    }

    #[test]
    fn test_gradient_output_shape_preserved() {
        let mask = make_mask(vec![0.0f32; 60], [3, 4, 5]);
        let result = MorphologicalGradient::new(1).apply(&mask);
        assert_eq!(result.shape(), [3, 4, 5]);
    }

    #[test]
    fn test_gradient_values_binary() {
        let mut vals = vec![0.0f32; 125];
        for i in 50..75 {
            vals[i] = 1.0;
        }
        let mask = make_mask(vals, [5, 5, 5]);
        let result = MorphologicalGradient::new(1).apply(&mask);
        let result_data = result.data().clone().into_data();
        let out_vals = result_data.as_slice::<f32>().unwrap();
        assert!(
            out_vals.iter().all(|&v| v == 0.0 || v == 1.0),
            "gradient output must be binary"
        );
    }
}

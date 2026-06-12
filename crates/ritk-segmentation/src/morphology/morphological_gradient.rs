//! Binary morphological gradient.
//!
//! Extracts the boundary of a binary foreground region as the set-theoretic
//! difference between dilation and erosion:
//!
//!   MorphGradient(M) = Dilation(M) AND NOT Erosion(M)
//!
//! The result is 1.0 at boundary voxels (in dilation but not erosion) and
//! 0.0 at interior foreground, exterior background, and all other voxels.

use super::{BinaryDilation, BinaryErosion, MorphologicalOperation};
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

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
        let dilated = BinaryDilation {
            radius: self.radius,
        }
        .apply(mask);
        let eroded = BinaryErosion {
            radius: self.radius,
        }
        .apply(mask);

        let shape = mask.shape();
        let [nz, ny, nx] = shape;
        let n = nz * ny * nx;
        let device = mask.data().device();

        let (dil_vals, _) = extract_vec_infallible(&dilated);
        let (ero_vals, _) = extract_vec_infallible(&eroded);

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            if dil_vals[i] >= super::FOREGROUND_THRESHOLD
                && ero_vals[i] < super::FOREGROUND_THRESHOLD
            {
                out[i] = 1.0;
            }
        }

        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(out, Shape::new([nz, ny, nx])), &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
    }
}

#[cfg(test)]
#[path = "tests_morphological_gradient.rs"]
mod tests;

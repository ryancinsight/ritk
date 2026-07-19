//! Binary morphological gradient.
//!
//! Extracts the boundary of a binary foreground region as the set-theoretic
//! difference between dilation and erosion:
//!
//!   MorphGradient(M) = Dilation(M) AND NOT Erosion(M)
//!
//! The result is 1.0 at boundary voxels (in dilation but not erosion) and
//! 0.0 at interior foreground, exterior background, and all other voxels.

use super::MorphologicalOperation;
use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Extracts the morphological gradient (boundary) of a binary mask.
///
/// Output voxel x is 1.0 iff  and .
pub struct MorphologicalGradient {
    /// Ball radius for structuring element.
    radius: usize,
}

impl Default for MorphologicalGradient {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl MorphologicalGradient {
    /// Create a gradient with the specified box half-width.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Return the box half-width.
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Apply the binary morphological gradient to a Coeus-native mask.
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
        crate::native_output::from_values(
            mask,
            gradient_values(values, mask.shape(), self.radius),
            backend,
        )
    }
}

impl<B: Backend> MorphologicalOperation<B, 3> for MorphologicalGradient {
    fn apply(&self, mask: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let shape = mask.shape();
        let device = B::default();
        let (values, _) = extract_vec_infallible(mask);
        let out = gradient_values(&values, shape, self.radius);
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &out, &device);
        Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
            .expect("invariant: segmentation output tensor preserves the image rank")
    }
}

fn gradient_values(values: &[f32], shape: [usize; 3], radius: usize) -> Vec<f32> {
    let dilated = super::binary_dilation::dilate_nd(values, &shape, radius);
    let eroded = super::binary_erosion::erode(values, &shape, radius);
    dilated
        .into_iter()
        .zip(eroded)
        .map(|(dilated, eroded)| {
            if dilated >= super::FOREGROUND_THRESHOLD && eroded < super::FOREGROUND_THRESHOLD {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "tests_morphological_gradient.rs"]
mod tests;

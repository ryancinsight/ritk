//! Image type with physical metadata and coordinate transformations.
//!
//! This module provides the Image struct which represents medical images
//! with tensor data and physical space metadata (origin, spacing, direction).

use crate::spatial::{Direction, Point, Spacing};
use anyhow::anyhow;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Medical image with physical metadata.
///
/// The Image type combines tensor data (potentially on GPU) with physical
/// space metadata that describes how image indices map to physical coordinates.
#[derive(Debug, Clone)]
pub struct Image<B: Backend, const D: usize> {
    /// The pixel data, potentially on GPU.
    data: Tensor<B, D>,
    /// Physical coordinate of the first pixel (index 0,0,0).
    origin: Point<D>,
    /// Physical distance between pixels along each axis.
    spacing: Spacing<D>,
    /// Orientation of the image axes.
    direction: Direction<D>,
}

impl<B: Backend, const D: usize> Image<B, D> {
    /// Create a new image with the given data and metadata.
    pub fn new(
        data: Tensor<B, D>,
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        Self {
            data,
            origin,
            spacing,
            direction,
        }
    }

    /// Get the image data tensor.
    pub fn data(&self) -> &Tensor<B, D> {
        &self.data
    }

    /// Get the origin (physical coordinate of first pixel).
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// Get the spacing (physical distance between pixels).
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// Get the direction (orientation matrix).
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }

    /// Get the image shape as an array.
    pub fn shape(&self) -> [usize; D] {
        self.data
            .shape()
            .dims
            .try_into()
            .expect("Tensor rank mismatch")
    }

    /// Consume the image and return the underlying tensor.
    ///
    /// Useful when the caller holds exclusive ownership and needs to avoid
    /// the arc-clone path in tensor data extraction.
    pub fn into_tensor(self) -> Tensor<B, D> {
        self.data
    }

    /// Consume the image and return all components.
    ///
    /// Returns `(tensor, origin, spacing, direction)`.
    pub fn into_parts(self) -> (Tensor<B, D>, Point<D>, Spacing<D>, Direction<D>) {
        (self.data, self.origin, self.spacing, self.direction)
    }

    /// Extract the underlying f32 tensor data as a `Vec<f32>`.
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[inline]
    pub fn data_vec(&self) -> Vec<f32> {
        self.data
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("Image::data_vec requires f32 backend tensor")
    }

    /// Extract the underlying f32 tensor data, propagating dtype errors.
    pub fn try_data_vec(&self) -> anyhow::Result<Vec<f32>> {
        self.data
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| anyhow!("image data is not f32: {:?}", e))
    }

    /// Provide a `&[f32]` view of the image data to a closure without
    /// allocating a `Vec`.
    ///
    /// Use this when only a read-only slice is needed and the caller does
    /// not require error propagation (returns `Ok`-wrapped values).
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[inline]
    pub fn with_data_slice<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let data = self.data.clone().into_data();
        let slice = data
            .as_slice::<f32>()
            .expect("image data must be contiguous f32");
        f(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;
    type Point3 = Point<3>;
    type Spacing3 = Spacing<3>;
    type Direction3 = Direction<3>;

    #[test]
    fn test_image_creation() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::new(data, origin, spacing, direction);

        assert_eq!(image.shape(), [10, 10, 10]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
    }
}

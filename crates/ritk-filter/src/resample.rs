//! Resample image filter.
//!
//! This module provides ResampleImageFilter which resamples an image
//! into a new coordinate system using a transform and an interpolator.

use ritk_core::image::Image;
use ritk_image::coeus::coeus_ops::BackendOps;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_interpolation::Interpolator;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::Transform;
use std::marker::PhantomData;

/// Coeus-native 3-D resampling operations.
pub mod native;

/// Resample image filter.
///
/// Resamples an image by applying a transform to map points from the
/// output image space to the input image space, and then interpolating values.
///
/// The transform maps from Output Physical Space -> Input Physical Space.
/// This is often the inverse of the registration transform (Fixed -> Moving).
///
/// # Type Parameters
/// * `B` - The Coeus backend
/// * `T` - The transform type
/// * `I` - The interpolator type
/// * `D` - The dimensionality (2 or 3)
pub struct ResampleImageFilter<B, T, I, const D: usize>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    T: Transform<B, D>,
    I: Interpolator<B>,
{
    size: [usize; D],
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
    transform: T,
    interpolator: I,
    default_pixel_value: f64,
    _phantom: PhantomData<fn() -> B>,
}

impl<B, T, I, const D: usize> ResampleImageFilter<B, T, I, D>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    T: Transform<B, D>,
    I: Interpolator<B>,
{
    /// Create a new resample filter.
    ///
    /// # Arguments
    /// * `size` - Output image size (pixels)
    /// * `origin` - Output image origin (physical)
    /// * `spacing` - Output image spacing (physical)
    /// * `direction` - Output image direction (matrix)
    /// * `transform` - Transform from output space to input space
    /// * `interpolator` - Interpolator for input image sampling
    pub fn new(
        size: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        transform: T,
        interpolator: I,
    ) -> Self {
        Self {
            size,
            origin,
            spacing,
            direction,
            transform,
            interpolator,
            default_pixel_value: 0.0,
            _phantom: PhantomData::<fn() -> B>,
        }
    }

    /// Set default pixel value for outside the field of view.
    pub fn with_default_pixel_value(mut self, value: f64) -> Self {
        self.default_pixel_value = value;
        self
    }

    /// Create from a reference image.
    ///
    /// Uses metadata (size, origin, spacing, direction) from the reference image.
    pub fn new_from_reference(reference: &Image<f32, B, D>, transform: T, interpolator: I) -> Self {
        Self::new(
            reference.shape(),
            *reference.origin(),
            *reference.spacing(),
            *reference.direction(),
            transform,
            interpolator,
        )
    }

    /// Apply filter to an input image.
    pub fn apply(&self, input: &Image<f32, B, D>) -> Image<f32, B, D> {
        let output_indices = self.generate_grid_indices();
        let output_data = self
            .resample_indices(input, output_indices)
            .reshape(self.size);

        Image::new(output_data, self.origin, self.spacing, self.direction)
            .expect("resampling constructss a tensor with the configured image rank")
    }

    /// Resample the values for one block of output grid indices: map output
    /// indices → output physical points → input physical points → input
    /// continuous indices, interpolate, then substitute the default pixel value
    /// for samples that fall outside the input buffer (matching ITK).
    fn resample_indices(
        &self,
        input: &Image<f32, B, D>,
        output_indices: Tensor<f32, B>,
    ) -> Tensor<f32, B> {
        let output_points = self.indices_to_physical(output_indices);
        let input_points = self.transform.transform_points(output_points);
        let input_indices = input.world_to_index_native(&input_points);
        let values = self
            .interpolator
            .interpolate(input.data(), input_indices.clone());
        self.apply_default_outside_buffer(values, input_indices, input.shape())
    }

    /// Replace interpolated values with `default_pixel_value` wherever the
    /// continuous input index falls outside the input buffer.
    ///
    /// This reproduces ITK's `InterpolateImageFunction::IsInsideBuffer`: a
    /// continuous index is inside iff, for every axis, it lies in the half-open
    /// interval `[-0.5, N - 0.5)` (`N` = axis size). Inside that interval the
    /// interpolator's own edge-clamping handles taps that reach one voxel past
    /// the border (so a sample at index `N - 0.75` still interpolates against the
    /// clamped edge value); outside it, ITK emits the default pixel value rather
    /// than the clamped edge — without this check ritk edge-clamped the entire
    /// out-of-FOV halo instead.
    ///
    /// `input_indices` columns are innermost-first (`column c` ↔ spatial axis
    /// `D - 1 - c`), matching `world_to_index_tensor` and the interpolators.
    fn apply_default_outside_buffer(
        &self,
        values: Tensor<f32, B>,
        input_indices: Tensor<f32, B>,
        input_shape: [usize; D],
    ) -> Tensor<f32, B> {
        let indices = input_indices.to_vec();
        let mut sampled = values.to_vec();
        for (point, value) in sampled.iter_mut().enumerate() {
            let inside = (0..D).all(|column| {
                let axis = D - 1 - column;
                let index = indices[point * D + column];
                index >= -0.5 && index < input_shape[axis] as f32 - 0.5
            });
            if !inside {
                *value = self.default_pixel_value as f32;
            }
        }
        Tensor::<f32, B>::from_slice([sampled.len()], &sampled)
    }

    fn generate_grid_indices(&self) -> Tensor<f32, B> {
        let count = self.size.iter().product();
        let mut strides = [1usize; D];
        for axis in (0..D.saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * self.size[axis + 1];
        }
        let mut indices = Vec::with_capacity(count * D);
        for linear in 0..count {
            for column in 0..D {
                let axis = D - 1 - column;
                indices.push(((linear / strides[axis]) % self.size[axis]) as f32);
            }
        }
        Tensor::<f32, B>::from_slice([count, D], &indices)
    }

    fn indices_to_physical(&self, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let indices = indices.to_vec();
        let mut points = Vec::with_capacity(indices.len());
        for index in indices.chunks_exact(D) {
            for world_axis in 0..D {
                let mut coordinate = self.origin[world_axis];
                for (column, &index_coordinate) in index.iter().enumerate() {
                    let image_axis = D - 1 - column;
                    coordinate += self.direction[(world_axis, image_axis)]
                        * self.spacing[image_axis]
                        * index_coordinate as f64;
                }
                points.push(coordinate as f32);
            }
        }
        Tensor::<f32, B>::from_slice([points.len() / D, D], &points)
    }
}

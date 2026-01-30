//! Resample image filter.
//!
//! This module provides ResampleImageFilter which resamples an image
//! into a new coordinate system using a transform and an interpolator.

use std::marker::PhantomData;
use burn::tensor::{Tensor, Shape, TensorData};
use burn::tensor::backend::Backend;
use crate::image::Image;
use crate::spatial::{Point, Spacing, Direction};
use crate::transform::trait_::Transform;
use crate::interpolation::trait_::Interpolator;

/// Resample image filter.
///
/// Resamples an image by applying a transform to map points from the
/// output image space to the input image space, and then interpolating values.
///
/// The transform maps from Output Physical Space -> Input Physical Space.
/// This is often the inverse of the registration transform (Fixed -> Moving).
///
/// # Type Parameters
/// * `B` - The Burn backend
/// * `T` - The transform type
/// * `I` - The interpolator type
/// * `D` - The dimensionality (2 or 3)
pub struct ResampleImageFilter<B, T, I, const D: usize>
where
    B: Backend,
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
    _phantom: PhantomData<B>,
}

impl<B, T, I, const D: usize> ResampleImageFilter<B, T, I, D>
where
    B: Backend,
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
            _phantom: PhantomData,
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
    pub fn new_from_reference(
        reference: &Image<B, D>,
        transform: T,
        interpolator: I,
    ) -> Self {
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
    pub fn apply(&self, input: &Image<B, D>) -> Image<B, D> {
        let device = input.data().device();
        
        // 1. Generate grid of indices for output image
        let output_indices = self.generate_grid_indices(&device);
        
        // 2. Convert output indices to output physical points
        let output_points = self.indices_to_physical(output_indices.clone(), &device);
        
        // 3. Apply transform to get input physical points
        // Transform maps Output Space -> Input Space
        let input_points = self.transform.transform_points(output_points);
        
        // 4. Convert input physical points to input continuous indices
        let input_indices = input.world_to_index_tensor(input_points);
        
        // 5. Interpolate values
        let output_flat = self.interpolator.interpolate(input.data(), input_indices);
        
        // 6. Reshape to output size
        let output_data = output_flat.reshape(Shape::new(self.size));
        
        Image::new(
            output_data,
            self.origin,
            self.spacing,
            self.direction,
        )
    }
    
    fn generate_grid_indices(&self, device: &<B as Backend>::Device) -> Tensor<B, 2> {
        // Generate indices for D=2 or D=3
        if D == 2 {
            let h = self.size[0];
            let w = self.size[1];
            
            let y_range = Tensor::<B, 1, burn::tensor::Int>::arange(0..h as i64, device);
            let x_range = Tensor::<B, 1, burn::tensor::Int>::arange(0..w as i64, device);
            
            // Create meshgrid using repeat with slice
            let y_grid = y_range.reshape([h, 1]).repeat(&[1, w]).reshape([h * w]);
            let x_grid = x_range.reshape([1, w]).repeat(&[h, 1]).reshape([h * w]);
            
            let y_grid = y_grid.float();
            let x_grid = x_grid.float();
            
            Tensor::cat(vec![x_grid.unsqueeze_dim(1), y_grid.unsqueeze_dim(1)], 1)
        } else if D == 3 {
            let d = self.size[0];
            let h = self.size[1];
            let w = self.size[2];
            
            let z_range = Tensor::<B, 1, burn::tensor::Int>::arange(0..d as i64, device);
            let y_range = Tensor::<B, 1, burn::tensor::Int>::arange(0..h as i64, device);
            let x_range = Tensor::<B, 1, burn::tensor::Int>::arange(0..w as i64, device);
            
            // Create 3D meshgrid using repeat with slice
            let z_grid = z_range.reshape([d, 1, 1]).repeat(&[1, h, w]).reshape([d * h * w]);
            let y_grid = y_range.reshape([1, h, 1]).repeat(&[d, 1, w]).reshape([d * h * w]);
            let x_grid = x_range.reshape([1, 1, w]).repeat(&[d, h, 1]).reshape([d * h * w]);
            
            let z_grid = z_grid.float();
            let y_grid = y_grid.float();
            let x_grid = x_grid.float();
            
            Tensor::cat(vec![
                x_grid.unsqueeze_dim(1),
                y_grid.unsqueeze_dim(1),
                z_grid.unsqueeze_dim(1)
            ], 1)
        } else {
            panic!("Unsupported dimensionality");
        }
    }
    
    fn indices_to_physical(&self, indices: Tensor<B, 2>, device: &<B as Backend>::Device) -> Tensor<B, 2> {
        // point = origin + Direction * (index * spacing)
        
        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
            device,
        ).reshape([1, D]);
            
        // 2. Prepare Transform Matrix
        // We want: Point = Origin + Index.matmul(T)
        // where T corresponds to applying spacing then direction
        
        let spacing_vec: Vec<f32> = (0..D).map(|i| self.spacing[i] as f32).collect();
        let spacing_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(spacing_vec, burn::tensor::Shape::new([D])),
            device,
        ).reshape([1, D]);
        
        let scaled_indices = indices * spacing_tensor;
        
        // Direction: [D, D]
        // We need D^T for matmul with row vectors
        let mut dir_data = Vec::with_capacity(D * D);
        for c in 0..D {
            for r in 0..D {
                dir_data.push(self.direction[(r, c)] as f32);
            }
        }
        
        let dir_t_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(dir_data, burn::tensor::Shape::new([D, D])),
            device,
        );
        
        let rotated = scaled_indices.matmul(dir_t_tensor);
        
        origin_tensor + rotated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use crate::transform::translation::TranslationTransform;
    use crate::interpolation::linear::LinearInterpolator;
    use crate::spatial::{Point2, Spacing2, Direction2};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_resample_translation_2d() {
        let device = Default::default();
        
        // 1. Create a 10x10 image with a 2x2 square in the center (4,4) to (5,5)
        let mut data = vec![0.0; 100];
        data[4 * 10 + 4] = 1.0;
        data[4 * 10 + 5] = 1.0;
        data[5 * 10 + 4] = 1.0;
        data[5 * 10 + 5] = 1.0;
        
        let tensor = Tensor::<TestBackend, 2>::from_floats(
            burn::tensor::TensorData::new(data, Shape::new([10, 10])),
            &device
        );
        
        let origin = Point2::new([0.0, 0.0]);
        let spacing = Spacing2::new([1.0, 1.0]);
        let direction = Direction2::identity();
        
        let image = Image::new(tensor, origin, spacing, direction);
        
        // 2. Define Transform: Shift by +2 in X, +1 in Y
        let offset = Tensor::<TestBackend, 1>::from_floats([-2.0, -1.0], &device);
        let transform = TranslationTransform::<TestBackend, 2>::new(offset);
        
        // 3. Define Interpolator
        let interpolator = LinearInterpolator::new();
        
        // 4. Create Resample Filter
        let filter = ResampleImageFilter::new_from_reference(
            &image,
            transform,
            interpolator,
        );
        
        // 5. Apply
        let result = filter.apply(&image);
        
        // 6. Verify
        let result_data = result.data().clone().into_data();
        let slice = result_data.as_slice::<f32>().unwrap();
        
        // Check (6,5) -> index 5*10 + 6 = 56
        assert!(slice[56] > 0.9);
        // Check (7,5) -> index 5*10 + 7 = 57
        assert!(slice[57] > 0.9);
        // Check (6,6) -> index 6*10 + 6 = 66
        assert!(slice[66] > 0.9);
        // Check (7,6) -> index 6*10 + 7 = 67
        assert!(slice[67] > 0.9);
        
        // Check original location (4,4) -> 44 should be 0
        assert!(slice[44] < 0.1);
    }
}

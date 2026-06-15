//! Resample image filter.
//!
//! This module provides ResampleImageFilter which resamples an image
//! into a new coordinate system using a transform and an interpolator.

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_interpolation::Interpolator;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::Transform;
use std::marker::PhantomData;

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
    _phantom: PhantomData<fn() -> B>,
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
    pub fn new_from_reference(reference: &Image<B, D>, transform: T, interpolator: I) -> Self {
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
        let [n_pixels, _] = output_indices.dims();

        let output_flat = if n_pixels <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            // Process all at once

            // 2. Convert output indices to output physical points
            let output_points = self.indices_to_physical(output_indices, &device);

            // 3. Apply transform to get input physical points
            let input_points = self.transform.transform_points(output_points);

            // 4. Convert input physical points to input continuous indices
            let input_indices = input.world_to_index_tensor(input_points);

            // 5. Interpolate values
            self.interpolator.interpolate(input.data(), input_indices)
        } else {
            // Process in chunks
            let num_chunks = n_pixels.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_pixels);

                let chunk_range = start..end;
                let chunk_indices = output_indices.clone().slice([chunk_range]);

                // Pipeline for this chunk
                let chunk_out_points = self.indices_to_physical(chunk_indices, &device);
                let chunk_in_points = self.transform.transform_points(chunk_out_points);
                let chunk_in_indices = input.world_to_index_tensor(chunk_in_points);
                let chunk_values = self
                    .interpolator
                    .interpolate(input.data(), chunk_in_indices);

                chunks.push(chunk_values);
            }
            Tensor::cat(chunks, 0)
        };

        // 6. Reshape to output size
        let output_data = output_flat.reshape(Shape::new(self.size));

        Image::new(output_data, self.origin, self.spacing, self.direction)
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
            let z_grid = z_range
                .reshape([d, 1, 1])
                .repeat(&[1, h, w])
                .reshape([d * h * w]);
            let y_grid = y_range
                .reshape([1, h, 1])
                .repeat(&[d, 1, w])
                .reshape([d * h * w]);
            let x_grid = x_range
                .reshape([1, 1, w])
                .repeat(&[d, h, 1])
                .reshape([d * h * w]);

            let z_grid = z_grid.float();
            let y_grid = y_grid.float();
            let x_grid = x_grid.float();

            Tensor::cat(
                vec![
                    x_grid.unsqueeze_dim(1),
                    y_grid.unsqueeze_dim(1),
                    z_grid.unsqueeze_dim(1),
                ],
                1,
            )
        } else {
            unreachable!(
                "D is const-generic and callers are only instantiated for D âˆˆ {{2, 3}}; D = {D}"
            );
        }
    }

    fn indices_to_physical(
        &self,
        indices: Tensor<B, 2>,
        device: &<B as Backend>::Device,
    ) -> Tensor<B, 2> {
        // point = origin + Direction * (index * spacing)

        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
            device,
        )
        .reshape([1, D]);

        // 2. Prepare Transform Matrix M so that Point = Index.matmul(M) + Origin.
        //
        // The grid index columns are INNERMOST-FIRST (column 0 = x = axis D-1, see
        // `generate_grid_indices` and the interpolation kernels), while spacing and
        // direction are stored AXIS-MAJOR (index 0 = depth/z). Index column `r`
        // therefore corresponds to spatial axis `D-1-r`; pairing them by `r`
        // directly scrambles world coordinates whenever spacing is anisotropic or a
        // grid axis is degenerate (e.g. a z = 1 promoted 2-D image), while leaving
        // isotropic/cubic cases unaffected. This mirrors `Image::index_to_world_tensor`.
        let mut m_data = Vec::with_capacity(D * D);
        for r in 0..D {
            let axis = D - 1 - r;
            for c in 0..D {
                m_data.push((self.spacing[axis] * self.direction[(c, axis)]) as f32);
            }
        }

        let m_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(m_data, burn::tensor::Shape::new([D, D])),
            device,
        );

        indices.matmul(m_tensor) + origin_tensor
    }
}

#[cfg(test)]
#[path = "tests_resample.rs"]
mod tests;

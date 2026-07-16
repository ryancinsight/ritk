//! Resample image filter.
//!
//! This module provides ResampleImageFilter which resamples an image
//! into a new coordinate system using a transform and an interpolator.

use coeus_core::{Backend, ComputeBackend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_image::native::Image;
use ritk_interpolation::Interpolator;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::Transform;

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
    _phantom: std::marker::PhantomData<fn() -> B>,
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
            _phantom: std::marker::PhantomData,
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
    pub fn apply(
        &self,
        input: &Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, D>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let n_pixels: usize = self.size.iter().product();

        let output_flat = if n_pixels <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            let output_indices = self.generate_grid_indices(backend);
            self.resample_indices(input, output_indices, backend)?
        } else {
            let num_chunks = n_pixels.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n_pixels);

                let chunk_indices = self.generate_grid_indices_slice(start, end, backend);
                chunks.push(self.resample_indices(input, chunk_indices, backend)?);
            }
            cat_tensors(&chunks, backend)?
        };

        let output_data = output_flat.reshape(self.size);

        Image::new(
            output_data,
            self.origin,
            self.spacing,
            self.direction,
        )
    }

    /// Resample the values for one block of output grid indices: map output
    /// indices → output physical points → input physical points → input
    /// continuous indices, interpolate, then substitute the default pixel value
    /// for samples that fall outside the input buffer (matching ITK).
    fn resample_indices(
        &self,
        input: &Image<f32, B, D>,
        output_indices: Tensor<f32, B>,
        backend: &B,
    ) -> anyhow::Result<Tensor<f32, B>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let output_points = self.indices_to_physical(&output_indices, backend);
        let input_points = self.transform.transform_points(output_points);
        let input_indices = input.world_to_index_native_on(&input_points, backend);
        let values = self
            .interpolator
            .interpolate(input.data(), input_indices.clone());
        self.apply_default_outside_buffer(values, &input_indices, input.shape(), backend)
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
    /// `D - 1 - c`), matching `world_to_index_native_on` and the interpolators.
    fn apply_default_outside_buffer(
        &self,
        values: Tensor<f32, B>,
        input_indices: &Tensor<f32, B>,
        input_shape: [usize; D],
        backend: &B,
    ) -> anyhow::Result<Tensor<f32, B>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let n = input_indices.shape()[0];
        let idx_slice = input_indices.as_slice();
        let val_slice = values.as_slice();
        let mut out = Vec::with_capacity(n);

        for point in 0..n {
            let mut inside = true;
            for c in 0..D {
                let axis = D - 1 - c;
                let idx = idx_slice[point * D + c];
                let upper = input_shape[axis] as f32 - 0.5;
                if idx < -0.5 || idx >= upper {
                    inside = false;
                    break;
                }
            }
            if inside {
                out.push(val_slice[point]);
            } else {
                out.push(self.default_pixel_value as f32);
            }
        }

        Ok(Tensor::from_slice_on([n], &out, backend))
    }

    fn generate_grid_indices(&self, backend: &B) -> Tensor<f32, B>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let n: usize = self.size.iter().product();
        let mut data = Vec::with_capacity(n * D);

        if D == 2 {
            let h = self.size[0];
            let w = self.size[1];
            for y in 0..h {
                for x in 0..w {
                    // innermost-first: x, then y
                    data.push(x as f32);
                    data.push(y as f32);
                }
            }
        } else if D == 3 {
            let d = self.size[0];
            let h = self.size[1];
            let w = self.size[2];
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        // innermost-first: x, y, z
                        data.push(x as f32);
                        data.push(y as f32);
                        data.push(z as f32);
                    }
                }
            }
        } else {
            unreachable!("D is const-generic and callers are only instantiated for D ∈ {{2, 3}}; D = {D}");
        }

        Tensor::from_slice_on([n, D], &data, backend)
    }

    fn generate_grid_indices_slice(
        &self,
        start: usize,
        end: usize,
        backend: &B,
    ) -> Tensor<f32, B>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let n = end - start;
        let mut data = Vec::with_capacity(n * D);

        if D == 2 {
            let w = self.size[1];
            for linear in start..end {
                let y = linear / w;
                let x = linear % w;
                data.push(x as f32);
                data.push(y as f32);
            }
        } else if D == 3 {
            let h = self.size[1];
            let w = self.size[2];
            for linear in start..end {
                let z = linear / (h * w);
                let rem = linear % (h * w);
                let y = rem / w;
                let x = rem % w;
                data.push(x as f32);
                data.push(y as f32);
                data.push(z as f32);
            }
        } else {
            unreachable!("D is const-generic and callers are only instantiated for D ∈ {{2, 3}}; D = {D}");
        }

        Tensor::from_slice_on([n, D], &data, backend)
    }

    fn indices_to_physical(
        &self,
        indices: &Tensor<f32, B>,
        backend: &B,
    ) -> Tensor<f32, B>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let n = indices.shape()[0];
        let idx_slice = indices.as_slice();
        let mut out = Vec::with_capacity(n * D);

        for point in 0..n {
            // idx columns are innermost-first; map to axis-major index
            let mut axis_idx = [0.0f32; D];
            for c in 0..D {
                axis_idx[D - 1 - c] = idx_slice[point * D + c];
            }

            // point = origin + Direction * (index * spacing)
            let mut scaled = [0.0f32; D];
            for axis in 0..D {
                scaled[axis] = axis_idx[axis] * self.spacing[axis] as f32;
            }

            for c in 0..D {
                let mut coord = self.origin[c] as f32;
                for axis in 0..D {
                    coord += self.direction[(c, axis)] as f32 * scaled[axis];
                }
                out.push(coord);
            }
        }

        Tensor::from_slice_on([n, D], &out, backend)
    }
}

/// Concatenate a sequence of `[n, D]` tensors along axis 0.
///
/// Helper used when chunking large resampling jobs.
fn cat_tensors<B: ComputeBackend>(
    tensors: &[Tensor<f32, B>],
    backend: &B,
) -> anyhow::Result<Tensor<f32, B>>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    if tensors.is_empty() {
        anyhow::bail!("cannot concatenate empty tensor list");
    }
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }
    let total: usize = tensors.iter().map(|t| t.shape()[0]).sum();
    let d = tensors[0].shape()[1];
    let mut data = Vec::with_capacity(total * d);
    for t in tensors {
        data.extend_from_slice(t.as_slice());
    }
    Ok(Tensor::from_slice_on([total, d], &data, backend))
}

#[cfg(test)]
#[path = "tests_resample.rs"]
mod tests;

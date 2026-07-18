//! Image type with physical metadata and coordinate transformations.
//!
//! This module provides the Image struct which represents medical images
//! with tensor data and physical space metadata (origin, spacing, direction).

use std::borrow::Cow;
use std::fmt;

use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

/// Medical image with physical metadata, backed by a Coeus tensor.
///
/// The Image type combines tensor data (potentially on GPU) with physical
/// space metadata that describes how image indices map to physical coordinates.
///
/// `T` is the scalar element type (e.g. `f32`, `f64`) and `B` is the compute
/// backend (e.g. `SequentialBackend`, `MoiraiBackend`).
#[derive(Clone)]
pub struct Image<T, B, const D: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// The pixel data, potentially on GPU.
    data: Tensor<T, B>,
    /// Physical coordinate of the first pixel (index 0,0,0).
    origin: Point<D>,
    /// Physical distance between pixels along each axis.
    spacing: Spacing<D>,
    /// Orientation of the image axes.
    direction: Direction<D>,
}

impl<T, B, const D: usize> fmt::Debug for Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .finish()
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Create a new image with the given data and metadata.
    pub fn new(
        data: Tensor<T, B>,
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

    /// Construct an image from flat voxel data on a specific backend.
    #[inline]
    pub fn from_flat_on(
        values: Vec<T>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        backend: &B,
    ) -> Self {
        let tensor = Tensor::<T, B>::from_slice_on(dims, &values, backend);
        Self::new(tensor, origin, spacing, direction)
    }

    /// Get the image data tensor.
    pub fn data(&self) -> &Tensor<T, B> {
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
        self.data.shape().try_into().expect("Tensor rank mismatch")
    }

    /// Consume the image and return the underlying tensor.
    ///
    /// Useful when the caller holds exclusive ownership and needs to avoid
    /// the arc-clone path in tensor data extraction.
    pub fn into_tensor(self) -> Tensor<T, B> {
        self.data
    }

    /// Consume the image and return all components.
    ///
    /// Returns `(tensor, origin, spacing, direction)`.
    pub fn into_parts(self) -> (Tensor<T, B>, Point<D>, Spacing<D>, Direction<D>) {
        (self.data, self.origin, self.spacing, self.direction)
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend + Default,
{
    /// Extract the underlying tensor data as a `Vec<T>`, propagating errors.
    ///
    /// Materializes logical tensor values through the backend's canonical
    /// device-to-host transfer.
    pub fn try_data_vec(&self) -> anyhow::Result<Vec<T>> {
        Ok(self.data.to_vec())
    }
}

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar + std::fmt::Debug,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    /// Provide a `&[T]` view of the image data to a closure without
    /// allocating a `Vec`.
    ///
    /// Use this when only a read-only slice is needed and the caller does
    /// not require error propagation (returns `Ok`-wrapped values).
    ///
    /// # Panics
    /// Panics if the tensor is not contiguous or not CPU-addressable.
    #[inline]
    pub fn with_data_slice<R>(&self, f: impl FnOnce(&[T]) -> R) -> R {
        f(self.data.as_slice())
    }

    /// Zero-copy slice view of the image data as `Cow<'_, [T]>`.
    ///
    /// Borrows when the tensor is contiguous; materializes a compact copy
    /// otherwise.
    #[inline]
    pub fn data_slice(&self) -> Cow<'_, [T]> {
        if self.data.is_contiguous() {
            Cow::Borrowed(self.data.as_slice())
        } else {
            Cow::Owned(
                self.data
                    .to_contiguous_on(&B::default())
                    .as_slice()
                    .to_vec(),
            )
        }
    }

    /// Fallible variant of [`Self::data_slice`].
    pub fn try_data_slice(&self) -> anyhow::Result<Cow<'_, [T]>> {
        Ok(self.data_slice())
    }
}

/// f32-specialized convenience methods (replaces the former burn f32 path).
impl<B, const D: usize> Image<f32, B, D>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    /// Construct an image from flat f32 voxel data on the default backend.
    #[inline]
    pub fn from_flat(
        values: Vec<f32>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        Self::from_flat_on(values, dims, origin, spacing, direction, &B::default())
    }
}

/// Batch coordinate transforms (ported from the former burn path).
impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar + coeus_core::Float,
    B: ComputeBackend + Default + coeus_ops::BackendOps<T>,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    /// Batch transform physical points to continuous indices.
    ///
    /// Coeus counterpart of the former Burn `world_to_index_tensor`.
    /// Delegates to the native implementation in [`crate::native::Image`].
    #[must_use]
    pub fn world_to_index_tensor(&self, points: Tensor<T, B>) -> Tensor<T, B> {
        let native = crate::native::Image::<T, B, D>::new(
            self.data.clone(),
            self.origin,
            self.spacing,
            self.direction,
        )
        .expect("rank invariant preserved by construction");
        native.world_to_index_native(&points)
    }

    /// Batch transform continuous indices to physical points.
    ///
    /// Coeus counterpart of the former Burn `index_to_world_tensor`.
    /// Delegates to the native implementation in [`crate::native::Image`].
    #[must_use]
    pub fn index_to_world_tensor(&self, indices: Tensor<T, B>) -> Tensor<T, B> {
        let native = crate::native::Image::<T, B, D>::new(
            self.data.clone(),
            self.origin,
            self.spacing,
            self.direction,
        )
        .expect("rank invariant preserved by construction");
        native.index_to_world_native(&indices)
    }
}

/// Single-point coordinate transforms (ported from the former burn path).
impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Convert a continuous index to a physical point.
    ///
    /// `point = origin + Direction * (index * spacing)`
    pub fn transform_continuous_index_to_physical_point(
        &self,
        index: &ritk_spatial::Point<D>,
    ) -> ritk_spatial::Point<D> {
        let mut scaled_index = ritk_spatial::Vector::<D>::zeros();
        for i in 0..D {
            scaled_index[i] = index[i] * self.spacing()[i];
        }
        let rotated = *self.direction() * scaled_index;
        *self.origin() + rotated
    }

    /// Convert a continuous physical point to a continuous index.
    ///
    /// `index = (Direction^-1 * (point - origin)) / spacing`
    pub fn transform_physical_point_to_continuous_index(
        &self,
        point: &ritk_spatial::Point<D>,
    ) -> ritk_spatial::Point<D> {
        let diff = *point - *self.origin();
        let inv_dir = self
            .direction()
            .try_inverse()
            .expect("direction matrix must be invertible");
        let rotated = inv_dir * diff;
        let mut index = ritk_spatial::Point::<D>::origin();
        for i in 0..D {
            index[i] = rotated[i] / self.spacing()[i];
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;
    type Point3 = Point<3>;
    type Spacing3 = Spacing<3>;
    type Direction3 = Direction<3>;

    #[test]
    fn test_image_creation() {
        let backend = B::default();
        let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let image = Image::<f32, B, 3>::new(data, origin, spacing, direction);

        assert_eq!(image.shape(), [10, 10, 10]);
        assert_eq!(image.origin(), &origin);
        assert_eq!(image.spacing(), &spacing);
        assert_eq!(image.direction(), &direction);
    }
}

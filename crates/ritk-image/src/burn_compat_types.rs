//! Image type with physical metadata and coordinate transformations.
//!
//! This module provides the Image struct which represents medical images
//! with tensor data and physical space metadata (origin, spacing, direction).

use std::borrow::Cow;

use crate::tensor::backend::Backend;
use crate::tensor::{Shape, Tensor, TensorData};
use anyhow::anyhow;
use ritk_spatial::{Direction, Point, Spacing};

/// **Soft-deprecated (Atlas Batch #3 sub-batch #2, 2026-07-06; docstring-only)** —
/// prefer [`crate::native::Image<T, B, D>`], where `T` implements
/// `coeus_core::Scalar` and `B` implements `coeus_core::ComputeBackend`.
/// No `#[deprecated]` attribute on this item to avoid cascading
/// `#[warn(deprecated)]` warnings across 671 burner consumer files in
/// `xtask/burn_surface.allowlist`. Migration plan:
/// `atlas/docs/adr/0012-ritk-burn-trait-rebind.md` §Sub-batch #2.
///
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

    /// Construct an image from flat f32 voxel data on a specific backend device.
    #[inline]
    pub fn from_flat_on(
        values: Vec<f32>,
        dims: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
        device: &B::Device,
    ) -> Self {
        let tensor = Tensor::<B, D>::from_data(TensorData::new(values, Shape::new(dims)), device);
        Self::new(tensor, origin, spacing, direction)
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

    /// Extract the underlying f32 tensor data, propagating dtype errors.
    ///
    /// Materializes the tensor to host via `into_data()`.  For large volumes on
    /// a host backend this `into_data()` step dominates; callers on the
    /// concrete `NdArray` backend that need throughput should use the zero-copy
    /// `into_primitive()` + `as_slice_memory_order()` path instead (see
    /// `ritk-python`'s `image_to_vec`).
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

    /// Read access to the image data as `Cow<'_, [f32]>`.
    ///
    /// Always returns `Cow::Owned` with the current backend API.
    /// The API contract is established so callers handle both variants;
    /// a future backend that exposes zero-copy host slices will switch
    /// to returning `Cow::Borrowed` without breaking call sites.
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[deprecated(since = "0.1.0", note = "use data_slice() instead")]
    #[inline]
    pub fn data_as_cow(&self) -> Cow<'_, [f32]> {
        self.data_slice()
    }

    /// Zero-copy slice view of the image data.
    ///
    /// Currently always returns `Cow::Owned` because Burn's public `Tensor` API
    /// does not expose a stable `&[f32]` host slice accessor. The `Cow` contract
    /// is established so call sites that do `let v: &[f32] = &*image.data_slice()`
    /// will work without changes once a future Burn release adds `Tensor::as_slice()`.
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[inline]
    pub fn data_slice(&self) -> Cow<'_, [f32]> {
        Cow::Owned(
            self.data
                .clone()
                .into_data()
                .into_vec::<f32>()
                .expect("Image::data_slice requires f32 backend tensor"),
        )
    }

    /// Fallible variant of [`Self::data_slice`].
    ///
    /// Propagates dtype mismatch errors (non-`f32` backend) instead of
    /// panicking. Returns `Cow::Owned` in all cases for the same reason
    /// documented on `data_slice`.
    pub fn try_data_slice(&self) -> anyhow::Result<Cow<'_, [f32]>> {
        // Both AD and non-AD paths require Owned because `try_data_vec` consumes
        // the tensor's data into a new Vec. A Borrowed path would require a
        // backend-specific buffer reference that the current Burn API does not
        // expose for autodiff inner tensors.
        Ok(Cow::Owned(self.try_data_vec()?))
    }

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

    /// Batch transform axis-major physical points to innermost-first indices.
    ///
    /// The input columns follow the metadata order (`z, y, x` for a 3-D
    /// image); the returned columns follow the interpolation-grid order
    /// (`x, y, z`). This is the convention consumed by the Burn-compatible
    /// interpolation kernels.
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        use crate::tensor::{Shape, TensorData};
        let device = points.device();
        let origin_vals: Vec<f32> = (0..D).map(|d| self.origin[d] as f32).collect();
        let origin_t =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vals, Shape::new([D])), &device);

        let inverse_direction = self
            .direction
            .try_inverse()
            .expect("invariant: image direction must be invertible");
        let mut transform = Vec::with_capacity(D * D);
        for source_axis in 0..D {
            for column in 0..D {
                let output_axis = D - 1 - column;
                transform.push(
                    (inverse_direction[(output_axis, source_axis)] / self.spacing[output_axis])
                        as f32,
                );
            }
        }
        let transform_t =
            Tensor::<B, 2>::from_data(TensorData::new(transform, Shape::new([D, D])), &device);
        let shifted = points - origin_t.unsqueeze_dim(0);
        shifted.matmul(transform_t)
    }

    /// Batch transform innermost-first indices to axis-major physical points.
    ///
    /// The input columns follow the interpolation-grid order (`x, y, z` for a
    /// 3-D image); the returned physical-point columns follow metadata order
    /// (`z, y, x`).
    pub fn index_to_world_tensor(&self, indices: Tensor<B, 2>) -> Tensor<B, 2> {
        use crate::tensor::{Shape, TensorData};
        let device = indices.device();
        let origin_vals: Vec<f32> = (0..D).map(|d| self.origin[d] as f32).collect();
        let origin_t =
            Tensor::<B, 1>::from_data(TensorData::new(origin_vals, Shape::new([D])), &device);

        let mut transform = Vec::with_capacity(D * D);
        for column in 0..D {
            let input_axis = D - 1 - column;
            for output_axis in 0..D {
                transform.push(
                    (self.spacing[input_axis] * self.direction[(output_axis, input_axis)]) as f32,
                );
            }
        }
        let transform_t =
            Tensor::<B, 2>::from_data(TensorData::new(transform, Shape::new([D, D])), &device);
        indices.matmul(transform_t) + origin_t.unsqueeze_dim(0)
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

    #[test]
    fn batch_coordinate_transforms_preserve_axis_order_and_direction() {
        let device = Default::default();
        let image = Image::new(
            Tensor::<Backend, 3>::zeros([2, 3, 5], &device),
            Point3::new([10.0, 20.0, 30.0]),
            Spacing3::new([2.0, 3.0, 5.0]),
            Direction3::from_row_major([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        );
        let indices = Tensor::<Backend, 2>::from_floats([[4.0, 2.0, 1.0]], &device);

        let world = image.index_to_world_tensor(indices.clone());
        let recovered = image.world_to_index_tensor(world.clone());
        assert_eq!(
            world.into_data().as_slice::<f32>().unwrap(),
            &[4.0, 22.0, 50.0]
        );

        assert_eq!(
            recovered.into_data().as_slice::<f32>().unwrap(),
            &[4.0, 2.0, 1.0]
        );
    }
}

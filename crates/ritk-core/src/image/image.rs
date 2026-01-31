//! Image type with physical metadata and coordinate transformations.
//!
//! This module provides the Image struct which represents medical images
//! with tensor data and physical space metadata (origin, spacing, direction).

use burn::tensor::{Tensor, TensorData};
use burn::tensor::backend::Backend;
use crate::spatial::{Point, Spacing, Direction};

/// Medical image with physical metadata.
///
/// The Image type combines tensor data (potentially on GPU) with physical
/// space metadata that describes how image indices map to physical coordinates.
///
/// # Type Parameters
/// * `B` - The backend (CPU or GPU) for tensor operations
/// * `D` - The dimensionality of the image (2 or 3)
///
/// # Coordinate Systems
/// * **Index Space**: Discrete pixel/voxel indices (integer coordinates)
/// * **Physical Space**: Continuous coordinates in mm or other units
///
/// # Examples
/// ```rust
/// use ritk_core::Image;
/// use ritk_core::spatial::{Point3, Spacing3, Direction3};
/// use burn::tensor::Tensor;
/// use burn_ndarray::NdArray;
///
/// type Backend = NdArray<f32>;
///
/// let device = Default::default();
/// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
/// let origin = Point3::new([0.0, 0.0, 0.0]);
/// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
/// let direction = Direction3::identity();
/// let image = Image::new(data, origin, spacing, direction);
/// ```
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
    ///
    /// # Arguments
    /// * `data` - The image data as a tensor
    /// * `origin` - Physical coordinate of the first pixel
    /// * `spacing` - Physical distance between pixels along each axis
    /// * `direction` - Orientation matrix of the image axes
    ///
    /// # Examples
    /// ```rust
    /// use ritk_core::Image;
    /// use ritk_core::spatial::{Point3, Spacing3, Direction3};
    /// use burn::tensor::Tensor;
    /// use burn_ndarray::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    ///
    /// let device = Default::default();
    /// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    /// let origin = Point3::new([0.0, 0.0, 0.0]);
    /// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    /// let direction = Direction3::identity();
    /// let image = Image::new(data, origin, spacing, direction);
    /// ```
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
        self.data.shape().dims.try_into().expect("Tensor rank mismatch")
    }

    /// Convert a continuous physical point to a continuous index.
    ///
    /// This transformation maps from physical space to index space using:
    /// `index = (Direction^-1 * (point - origin)) / spacing`
    ///
    /// # Arguments
    /// * `point` - A point in physical space
    ///
    /// # Returns
    /// The corresponding continuous index
    ///
    /// # Examples
    /// ```rust
    /// use ritk_core::Image;
    /// use ritk_core::spatial::{Point3, Spacing3, Direction3};
    /// use burn::tensor::Tensor;
    /// use burn_ndarray::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    /// let origin = Point3::new([0.0, 0.0, 0.0]);
    /// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    /// let direction = Direction3::identity();
    /// let image = Image::new(data, origin, spacing, direction);
    ///
    /// let point = Point3::new([5.0, 5.0, 5.0]);
    /// let index = image.transform_physical_point_to_continuous_index(&point);
    /// ```
    pub fn transform_physical_point_to_continuous_index(&self, point: &Point<D>) -> Point<D> {
        // index = (Direction^-1 * (point - origin)) / spacing
        // Note: This implementation assumes direction is orthogonal/rotation matrix usually, but we do full inverse.
        // Nalgebra operations are on CPU.
        
        let diff = *point - self.origin;
        // Inverse direction
        let inv_dir = self.direction.try_inverse().expect("Direction matrix must be invertible");
        let rotated = inv_dir * diff;
        
        // Element-wise division by spacing
        let mut index = Point::<D>::origin();
        for i in 0..D {
            index[i] = rotated[i] / self.spacing[i];
        }
        index
    }
    
    /// Convert a continuous index to a physical point.
    ///
    /// This transformation maps from index space to physical space using:
    /// `point = origin + Direction * (index * spacing)`
    ///
    /// # Arguments
    /// * `index` - A continuous index
    ///
    /// # Returns
    /// The corresponding physical point
    ///
    /// # Examples
    /// ```rust
    /// use ritk_core::Image;
    /// use ritk_core::spatial::{Point3, Spacing3, Direction3};
    /// use burn::tensor::Tensor;
    /// use burn_ndarray::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    /// let origin = Point3::new([0.0, 0.0, 0.0]);
    /// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    /// let direction = Direction3::identity();
    /// let image = Image::new(data, origin, spacing, direction);
    ///
    /// let index = Point3::new([5.0, 5.0, 5.0]);
    /// let point = image.transform_continuous_index_to_physical_point(&index);
    /// ```
    pub fn transform_continuous_index_to_physical_point(&self, index: &Point<D>) -> Point<D> {
        // point = origin + Direction * (index * spacing)
        let mut scaled_index = crate::spatial::Vector::<D>::zeros();
        for i in 0..D {
            scaled_index[i] = index[i] * self.spacing[i];
        }
        
        let rotated = self.direction * scaled_index;
        self.origin + rotated
    }

    /// Batch transform physical points to continuous indices using tensors.
    ///
    /// Maps from physical space to index space.
    ///
    /// # Arguments
    /// * `points` - A tensor of shape `[Batch, D]` containing physical points
    ///
    /// # Returns
    /// A tensor of shape `[Batch, D]` containing continuous indices
    ///
    /// # Examples
    /// ```rust
    /// use ritk_core::Image;
    /// use ritk_core::spatial::{Point3, Spacing3, Direction3};
    /// use burn::tensor::Tensor;
    /// use burn_ndarray::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    /// let origin = Point3::new([0.0, 0.0, 0.0]);
    /// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    /// let direction = Direction3::identity();
    /// let image = Image::new(data, origin, spacing, direction);
    ///
    /// let points = Tensor::<Backend, 2>::zeros([100, 3], &device);
    /// let indices = image.world_to_index_tensor(points);
    /// ```
    pub fn world_to_index_tensor(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = points.device();
        
        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
            &device,
        ).reshape([1, D]);
            
        // 2. Prepare Transform Matrix T = (S^-1 * D^-1)^T = (D^-1)^T * S^-1
        // I = (P - O) @ T
        // T_rc = (D^-1)_cr / S_c
        
        let inv_dir = self.direction.try_inverse().expect("Direction matrix must be invertible");

        let mut t_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                // T[r, c] uses inv_dir[c, r] and spacing[c]
                let val = (inv_dir[(c, r)] / self.spacing[c]) as f32;
                t_data.push(val);
            }
        }
        
        let t_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(t_data, burn::tensor::Shape::new([D, D])),
            &device,
        );
            
        let diff = points - origin_tensor;
        diff.matmul(t_tensor)
    }

    /// Batch transform continuous indices to physical points using tensors.
    ///
    /// Maps from index space to physical space.
    ///
    /// # Arguments
    /// * `indices` - A tensor of shape `[Batch, D]` containing continuous indices
    ///
    /// # Returns
    /// A tensor of shape `[Batch, D]` containing physical points
    ///
    /// # Examples
    /// ```rust
    /// use ritk_core::Image;
    /// use ritk_core::spatial::{Point3, Spacing3, Direction3};
    /// use burn::tensor::Tensor;
    /// use burn_ndarray::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    /// let origin = Point3::new([0.0, 0.0, 0.0]);
    /// let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    /// let direction = Direction3::identity();
    /// let image = Image::new(data, origin, spacing, direction);
    ///
    /// let indices = Tensor::<Backend, 2>::zeros([100, 3], &device);
    /// let points = image.index_to_world_tensor(indices);
    /// ```
    pub fn index_to_world_tensor(&self, indices: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = indices.device();
        
        // 1. Prepare Origin Tensor [1, D]
        let origin_vec: Vec<f32> = (0..D).map(|i| self.origin[i] as f32).collect();
        let origin_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(origin_vec, burn::tensor::Shape::new([D])),
            &device,
        ).reshape([1, D]);
            
        // 2. Prepare Transform Matrix M = S * D^T
        // P = O + I @ M
        // M_rc = S_r * D_cr
        
        let mut m_data = Vec::with_capacity(D * D);
        for r in 0..D {
            for c in 0..D {
                let val = (self.spacing[r] * self.direction[(c, r)]) as f32;
                m_data.push(val);
            }
        }
        
        let m_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(m_data, burn::tensor::Shape::new([D, D])),
            &device,
        );
            
        let rotated = indices.matmul(m_tensor);
        rotated + origin_tensor
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
    fn test_physical_to_index_transform() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let point = Point3::new([5.0, 5.0, 5.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);
        
        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_index_to_physical_transform() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let index = Point3::new([5.0, 5.0, 5.0]);
        let point = image.transform_continuous_index_to_physical_point(&index);
        
        assert!((point[0] - 5.0).abs() < 1e-6);
        assert!((point[1] - 5.0).abs() < 1e-6);
        assert!((point[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_roundtrip() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let original_point = Point3::new([3.5, 4.5, 5.5]);
        let index = image.transform_physical_point_to_continuous_index(&original_point);
        let transformed_point = image.transform_continuous_index_to_physical_point(&index);
        
        assert!((original_point[0] - transformed_point[0]).abs() < 1e-6);
        assert!((original_point[1] - transformed_point[1]).abs() < 1e-6);
        assert!((original_point[2] - transformed_point[2]).abs() < 1e-6);
    }

    #[test]
    fn test_non_unit_spacing() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([2.0, 2.0, 2.0]);
        let direction = Direction3::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let point = Point3::new([10.0, 10.0, 10.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);
        
        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_non_zero_origin() {
        let device = Default::default();
        let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
        let origin = Point3::new([10.0, 20.0, 30.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();
        
        let image = Image::new(data, origin, spacing, direction);
        
        let point = Point3::new([15.0, 25.0, 35.0]);
        let index = image.transform_physical_point_to_continuous_index(&point);
        
        assert!((index[0] - 5.0).abs() < 1e-6);
        assert!((index[1] - 5.0).abs() < 1e-6);
        assert!((index[2] - 5.0).abs() < 1e-6);
    }
}

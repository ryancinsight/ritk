//! Transform trait for spatial coordinate transformations.
//!
//! This module defines the core Transform trait that all spatial transforms must implement.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crate::spatial::{Point, Spacing, Direction};

/// Transform trait for spatial coordinate transformations.
///
/// Maps points from one physical space to another.
/// All transforms must implement this trait to be used in registration.
/// Note: This trait does not enforce `burn::module::Module` inheritance,
/// allowing for both trainable (Module) and non-trainable (pure) transforms.
///
/// # Type Parameters
/// * `B` - The Burn backend
/// * `D` - The spatial dimensionality (2 or 3)
pub trait Transform<B: Backend, const D: usize> {
    /// Apply transform to a batch of points.
    ///
    /// # Arguments
    /// * `points` - Tensor of shape `[Batch, D]` containing the input points
    ///
    /// # Returns
    /// Tensor of shape `[Batch, D]` containing the transformed points
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Get the inverse transform (if available).
    ///
    /// Not all transforms are easily invertible, so this returns an Option.
    fn inverse(&self) -> Option<Box<dyn Transform<B, D>>> {
        None
    }
}

/// Trait for transforms that can be resampled to a new grid/resolution.
///
/// This is used in multi-resolution registration to adapt the transform
/// (e.g., displacement field) when moving from coarse to fine levels.
pub trait Resampleable<B: Backend, const D: usize> {
    /// Resample the transform to match a new image grid.
    ///
    /// # Arguments
    /// * `shape` - The new image shape
    /// * `origin` - The new image origin
    /// * `spacing` - The new image spacing
    /// * `direction` - The new image direction
    ///
    /// # Returns
    /// A new instance of the transform adapted to the new grid.
    fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self;
}

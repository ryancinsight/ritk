//! The hoisted-inverse transform pair for an image's Cartesian grid.
//!
//! The third granularity: [`super::point`] converts one coordinate under a
//! fresh direction inversion, [`super::batch`] converts a whole tensor under one
//! backend dispatch, and this converts many coordinates one at a time — the
//! shape a per-voxel sweep, a coordinate-descent iteration, or a contour walk
//! needs — by hoisting the inversion out of the loop into
//! [`CartesianGridGeometry`].
//!
//! This is the entry point every consumer should reach for rather than writing
//! the affine out again: `image.grid_geometry()?` is the whole construction.

use coeus_core::{ComputeBackend, Scalar};
use ritk_spatial::{CartesianGridGeometry, NonCartesianGrid};

use crate::types::Image;

impl<T, B, const D: usize> Image<T, B, D>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// The image's index/world transform pair, with the direction inverse
    /// computed once.
    ///
    /// # Errors
    ///
    /// Returns an error when the image carries a beam-space acquisition map,
    /// for which the affine pair is not the mapping, or when the direction
    /// cosine matrix is singular. See [`CartesianGridGeometry::new`].
    pub fn grid_geometry(&self) -> Result<CartesianGridGeometry<D>, NonCartesianGrid> {
        CartesianGridGeometry::new(&self.origin, &self.spacing, &self.direction, &self.map)
    }
}

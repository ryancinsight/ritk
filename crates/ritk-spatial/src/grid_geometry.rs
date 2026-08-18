//! The Cartesian index/world transform pair, with the direction inverse hoisted.
//!
//! One family for every index↔physical mapping on a Cartesian raster. The pair
//! is `point = origin + D S index` and `index = S^-1 D^-1 (point - origin)`,
//! where `D` is the direction cosine matrix and `S` the diagonal spacing matrix.
//!
//! `ritk_image::Image` exposes the same two directions per point (ADR 0018) and
//! per tensor (ADR 0020), dispatching on the image's [`CoordinateMap`] so a
//! beam-space acquisition maps through its own geometry. Those dispatching forms
//! recompute a full Gauss-Jordan inversion of the direction matrix on every
//! call, which is fine per point and ruinous inside a per-voxel loop. This type
//! is the Cartesian branch of that same pair with the inversion hoisted out: it
//! is constructed once from an image's geometry and then answers both directions
//! with no allocation and no per-point decomposition.
//!
//! It is deliberately restricted to [`CoordinateMap::Cartesian`], so it cannot
//! silently return a Cartesian answer for a beam-space acquisition the way the
//! dispatching form prevents.
//!
//! # Axis order
//!
//! Index coordinates are in tensor-axis order — for `D = 3`, `[i0, i1, i2] =
//! [z, y, x]`, axis 0 slowest-varying — matching `spacing` and the *columns* of
//! `direction`. Physical coordinates are LPS `[x, y, z]`, matching `origin` and
//! the *rows* of `direction`. The two orders differ; conflating them is the
//! defect this type exists to make hard to write.

use crate::{CoordinateMap, Direction, Point, Spacing, Vector};

/// Index/world transform pair for a Cartesian image grid, with the direction
/// inverse hoisted for repeated use.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianGridGeometry<const D: usize> {
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
    inverse_direction: Direction<D>,
}

impl<const D: usize> CartesianGridGeometry<D> {
    /// Build the transform pair from an image's geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when `map` is not [`CoordinateMap::Cartesian`], because
    /// a beam-space index tuple is a beam and a sample rather than a raster
    /// coordinate and the affine formula would return a point in no physical
    /// space at all; and when `direction` is singular, which is a malformed
    /// file header rather than a programmer error.
    pub fn new(
        origin: &Point<D>,
        spacing: &Spacing<D>,
        direction: &Direction<D>,
        map: &CoordinateMap,
    ) -> Result<Self, NonCartesianGrid> {
        if !map.is_cartesian() {
            return Err(NonCartesianGrid::CoordinateMap(format!("{map:?}")));
        }
        let inverse_direction = direction
            .try_inverse()
            .ok_or(NonCartesianGrid::SingularDirection)?;
        Ok(Self {
            origin: *origin,
            spacing: *spacing,
            direction: *direction,
            inverse_direction,
        })
    }

    /// Physical LPS point of the continuous index, computing `origin + D S index`.
    #[must_use]
    pub fn point(&self, index: [f64; D]) -> [f64; D] {
        let mut scaled = Vector::zeros();
        for axis in 0..D {
            scaled[axis] = index[axis] * self.spacing[axis];
        }
        (self.origin + self.direction * scaled).to_array()
    }

    /// Continuous index of the physical LPS point, computing
    /// `S^-1 D^-1 (point - origin)`.
    #[must_use]
    pub fn index(&self, point: [f64; D]) -> [f64; D] {
        let rotated = self.inverse_direction * (Point::new(point) - self.origin);
        let mut index = [0.0; D];
        for axis in 0..D {
            index[axis] = rotated[axis] / self.spacing[axis];
        }
        index
    }

    /// Physical direction along which index `axis` advances — column `axis` of
    /// the direction matrix, unnormalised.
    ///
    /// # Panics
    ///
    /// Panics when `axis >= D`.
    #[must_use]
    pub fn axis_direction(&self, axis: usize) -> [f64; D] {
        assert!(axis < D, "axis {axis} is out of range for a {D}-D grid");
        let mut column = [0.0; D];
        for (row, entry) in column.iter_mut().enumerate() {
            *entry = self.direction[(row, axis)];
        }
        column
    }

    /// The grid origin — the physical point of index zero.
    #[must_use]
    pub fn origin(&self) -> &Point<D> {
        &self.origin
    }

    /// The physical distance between neighbouring voxels along each index axis.
    #[must_use]
    pub fn spacing(&self) -> &Spacing<D> {
        &self.spacing
    }

    /// The direction cosine matrix.
    #[must_use]
    pub fn direction(&self) -> &Direction<D> {
        &self.direction
    }
}

/// Why a [`CartesianGridGeometry`] could not be built.
#[derive(Debug, thiserror::Error)]
pub enum NonCartesianGrid {
    /// The image carries a beam-space acquisition map, for which an affine
    /// index/world pair is not the mapping.
    #[error("a Cartesian grid geometry requires a Cartesian coordinate map; got {0}")]
    CoordinateMap(String),
    /// The direction cosine matrix has no inverse, so world→index is undefined.
    #[error("image direction matrix is singular")]
    SingularDirection,
}

#[cfg(test)]
#[path = "tests_grid_geometry.rs"]
mod tests;

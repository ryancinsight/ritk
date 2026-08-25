//! Hoisted Cartesian index/world geometry for grid-sweeping filters.
//!
//! [`ritk_image::Image::continuous_index_to_physical_point`] and its inverse
//! (ADR 0018) are the canonical single-point transforms, and callers that touch
//! a handful of points should use them directly. The inverse recomputes a full
//! Gauss-Jordan inversion of the direction matrix on every call, which is fine
//! per point and ruinous inside a per-voxel coordinate-descent loop.
//!
//! This type is that pair with the inversion hoisted out of the loop: it is
//! constructed once from an image's geometry, validates the assumptions the
//! grid-sweeping filters make, and then answers both directions with no
//! allocation and no per-point decomposition. It implements the same Cartesian
//! formula as the seam — `point = origin + D S index` — and is deliberately
//! restricted to [`CoordinateMap::Cartesian`] so it cannot silently return a
//! Cartesian answer for a beam-space acquisition the way the seam's dispatch
//! would prevent.
//!
//! # Axis order
//!
//! Index coordinates are in tensor-axis order `[i0, i1, i2] = [z, y, x]`, axis
//! 0 slowest-varying, matching `spacing` and the *columns* of `direction`.
//! Physical coordinates are LPS `[x, y, z]`, matching `origin` and the *rows*
//! of `direction`. The two orders differ; conflating them is the defect this
//! type exists to make hard to write.

use ritk_spatial::{CoordinateMap, Direction, Point, Spacing, Vector};

/// Index/world transform pair for a Cartesian image grid, with the direction
/// inverse hoisted for repeated use.
#[derive(Debug, Clone)]
pub(crate) struct CartesianGridGeometry {
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
    inverse_direction: Direction<3>,
}

impl CartesianGridGeometry {
    /// Build the transform pair from an image's geometry.
    ///
    /// # Errors
    ///
    /// Returns an error when `map` is not [`CoordinateMap::Cartesian`], because
    /// a beam-space index pair is a beam and a sample rather than a raster
    /// coordinate and the affine formula would return a point in no physical
    /// space at all; and when `direction` is singular, which is a malformed
    /// file header rather than a programmer error.
    pub(crate) fn new(
        origin: &Point<3>,
        spacing: &Spacing<3>,
        direction: &Direction<3>,
        map: &CoordinateMap,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            map.is_cartesian(),
            "displacement-field inversion requires a Cartesian image; got {map:?}"
        );
        let inverse_direction = direction
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("image direction matrix is singular"))?;
        Ok(Self {
            origin: *origin,
            spacing: *spacing,
            direction: *direction,
            inverse_direction,
        })
    }

    /// Physical LPS point `[x, y, z]` of the continuous index `[z, y, x]`.
    ///
    /// Computes `origin + D S index`.
    pub(crate) fn point(&self, index: [f64; 3]) -> [f64; 3] {
        let scaled = Vector::new([
            index[0] * self.spacing[0],
            index[1] * self.spacing[1],
            index[2] * self.spacing[2],
        ]);
        let point = self.origin + self.direction * scaled;
        [point[0], point[1], point[2]]
    }

    /// Continuous index `[z, y, x]` of the physical LPS point `[x, y, z]`.
    ///
    /// Computes `S^-1 D^-1 (point - origin)`.
    pub(crate) fn index(&self, point: [f64; 3]) -> [f64; 3] {
        let rotated = self.inverse_direction * (Point::new(point) - self.origin);
        [
            rotated[0] / self.spacing[0],
            rotated[1] / self.spacing[1],
            rotated[2] / self.spacing[2],
        ]
    }

    /// Physical direction along which index `axis` advances — column `axis` of
    /// the direction matrix, unnormalised.
    pub(crate) fn axis_direction(&self, axis: usize) -> [f64; 3] {
        [
            self.direction[(0, axis)],
            self.direction[(1, axis)],
            self.direction[(2, axis)],
        ]
    }
}

#[cfg(test)]
#[path = "tests_grid_geometry.rs"]
mod tests;

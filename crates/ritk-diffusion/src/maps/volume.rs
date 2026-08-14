//! Spatial lookup over a fitted tensor field.
//!
//! [`DiffusionMaps`] is a flat list of per-voxel results with no notion of where
//! any voxel sits. Tractography asks the opposite question — "what is the fibre
//! orientation *here*" — so it needs the volume's shape. [`DtiVolume`] adds it.
//!
//! # Frame
//!
//! Queries are in **voxel-index space**, ordered `[depth, row, column]` to match
//! `Image::shape()` and the layout the maps were fitted from. The volume
//! therefore carries no origin or spacing: an index needs none, and applying a
//! physical transform here would duplicate one the image already owns.
//! Physical coordinates belong at the IO boundary, through
//! `Image::continuous_index_to_physical_point`.
//!
//! Note that `crate::noddi::NoddiVolume` uses the opposite convention — shape
//! `[nx, ny, nz]` with the query's first component as the *fastest* axis — and
//! converts from physical space internally. That divergence predates this
//! module; it is recorded rather than silently matched, since matching it would
//! put this type at odds with the `Image` layout its data comes from.

use ritk_spatial::{Point, Vector};

use super::{DiffusionMaps, DiffusionMapsError};
#[cfg(test)]
use super::{DiffusionMapsConfig, fit_diffusion_maps};

/// Below this squared norm a stored eigenvector carries no orientation.
///
/// The masked-out voxels store exact zeros, so this only has to separate those
/// from a genuine unit vector; it is not a physical threshold.
const DEGENERATE_NORM_SQUARED: f64 = 1.0e-30;

/// A fitted tensor field placed on a voxel grid.
///
/// Built from [`DiffusionMaps`] plus the shape the maps were fitted over.
#[derive(Debug, Clone)]
pub struct DtiVolume {
    maps: DiffusionMaps,
    shape: [usize; 3],
    anisotropy_floor: f64,
}

impl DtiVolume {
    /// Place `maps` on a `[depth, row, column]` grid.
    ///
    /// `anisotropy_floor` is the fractional anisotropy below which
    /// [`Self::direction_at`] reports no orientation. Tracking through
    /// near-isotropic tissue follows an eigenvector that the data does not
    /// distinguish from any other, so a streamline continuing there is
    /// describing noise; 0.2 or so is the conventional white-matter floor.
    ///
    /// # Errors
    ///
    /// [`DiffusionMapsError::VolumeLengthMismatch`] when `shape` does not
    /// describe exactly the voxels `maps` holds, and
    /// [`DiffusionMapsError::InvalidConfiguration`] when `anisotropy_floor` is
    /// not a fraction in `[0, 1]`.
    pub fn new(
        maps: DiffusionMaps,
        shape: [usize; 3],
        anisotropy_floor: f64,
    ) -> Result<Self, DiffusionMapsError> {
        if !anisotropy_floor.is_finite() || !(0.0..=1.0).contains(&anisotropy_floor) {
            return Err(DiffusionMapsError::InvalidConfiguration {
                parameter: "anisotropy_floor",
                value: anisotropy_floor,
            });
        }

        let expected = shape.iter().product::<usize>();
        if maps.len() != expected {
            return Err(DiffusionMapsError::VolumeLengthMismatch {
                index: 0,
                length: maps.len(),
                expected,
            });
        }

        Ok(Self {
            maps,
            shape,
            anisotropy_floor,
        })
    }

    /// Grid extent, `[depth, rows, columns]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// The underlying per-voxel results.
    #[must_use]
    pub const fn maps(&self) -> &DiffusionMaps {
        &self.maps
    }

    /// Fibre orientation at a voxel index, or `None` where none is supported.
    ///
    /// Nearest-neighbour: the index is rounded to a voxel. `None` is returned
    /// outside the grid, at a voxel that was never fitted, below the anisotropy
    /// floor, or where the stored eigenvector is degenerate — each of which
    /// means a streamline arriving here should stop rather than continue on an
    /// orientation the data does not support.
    #[must_use]
    pub fn direction_at(&self, index: &Point<3>) -> Option<Vector<3>> {
        let voxel = self.voxel_of(index)?;
        if !self.maps.mask()[voxel] {
            return None;
        }
        if self.maps.fractional_anisotropy_at(voxel) < self.anisotropy_floor {
            return None;
        }

        let direction = self.maps.principal_eigenvector()[voxel];
        let norm_squared = direction
            .iter()
            .map(|component| component * component)
            .sum::<f64>();
        if norm_squared < DEGENERATE_NORM_SQUARED {
            return None;
        }

        let scale = norm_squared.sqrt().recip();
        Some(Vector::new(direction.map(|component| component * scale)))
    }

    /// Flat voxel offset for an index, or `None` when it falls outside the grid.
    fn voxel_of(&self, index: &Point<3>) -> Option<usize> {
        let mut offset = 0_usize;
        for (axis, extent) in self.shape.iter().enumerate() {
            let rounded = index[axis].round();
            if !rounded.is_finite() || rounded < 0.0 {
                return None;
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "negative and non-finite values are rejected immediately above"
            )]
            let coordinate = rounded as usize;
            if coordinate >= *extent {
                return None;
            }
            offset = offset * extent + coordinate;
        }
        Some(offset)
    }
}

#[cfg(test)]
#[path = "volume_tests.rs"]
mod tests;

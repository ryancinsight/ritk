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

use crate::dti::symmetric_eigen;

use super::{DiffusionMaps, DiffusionMapsError};
#[cfg(test)]
use super::{DiffusionMapsConfig, fit_diffusion_maps};

/// Relative gap below which the interpolated dyadic has no dominant axis.
///
/// This is a floating-point degeneracy guard, not a tuning parameter: exactly
/// equal leading eigenvalues mean the local orientations cancel and no
/// direction is defined. Coherent white matter produces `λ₀ ≫ λ₁`, so the
/// threshold never fires there.
const DEGENERATE_EIGENVALUE_GAP: f64 = 1.0e-12;

/// Below this squared norm a stored eigenvector carries no orientation.
///
/// The masked-out voxels store exact zeros, so this only has to separate those
/// from a genuine unit vector; it is not a physical threshold.
const DEGENERATE_NORM_SQUARED: f64 = 1.0e-30;

/// How an orientation is sampled between voxel centres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DirectionInterpolation {
    /// The orientation of the nearest voxel centre.
    ///
    /// Constant within a voxel and discontinuous at each boundary, so a
    /// streamline crossing a boundary sees the full inter-voxel angle at once
    /// and can exceed a turn limit that the underlying bundle never does.
    Nearest,

    /// Trilinear over the outer product `v vᵀ` of the surrounding voxels.
    ///
    /// Orientations cannot be averaged directly: an eigenvector has no sign, so
    /// `v` and `−v` describe the same fibre and averaging them cancels to zero.
    /// The outer product is invariant under that sign — `(−v)(−v)ᵀ = v vᵀ` — so
    /// interpolating it and taking the principal eigenvector of the result
    /// combines orientations without ever needing to guess a sign. This is
    /// structural, not a heuristic alignment that can pick wrong.
    #[default]
    Trilinear,
}

/// A fitted tensor field placed on a voxel grid.
///
/// Built from [`DiffusionMaps`] plus the shape the maps were fitted over.
#[derive(Debug, Clone)]
pub struct DtiVolume {
    maps: DiffusionMaps,
    shape: [usize; 3],
    anisotropy_floor: f64,
    interpolation: DirectionInterpolation,
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
            interpolation: DirectionInterpolation::default(),
        })
    }

    /// Sample orientations by `interpolation` instead of the default.
    #[must_use]
    pub const fn with_interpolation(mut self, interpolation: DirectionInterpolation) -> Self {
        self.interpolation = interpolation;
        self
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
        // Whether a streamline may continue is decided by the voxel it is in,
        // not by its neighbours: interpolation smooths the orientation, it does
        // not extend the trackable region. Keeping the two separate means
        // switching interpolation cannot silently change where tracking stops.
        let nearest = self.trackable(self.voxel_of(index)?)?;

        match self.interpolation {
            DirectionInterpolation::Nearest => unit(nearest),
            DirectionInterpolation::Trilinear => unit(self.interpolated(index).unwrap_or(nearest)),
        }
    }

    /// The stored orientation at `voxel`, if a streamline may follow it.
    fn trackable(&self, voxel: usize) -> Option<[f64; 3]> {
        if !self.maps.mask()[voxel] {
            return None;
        }
        if self.maps.fractional_anisotropy_at(voxel) < self.anisotropy_floor {
            return None;
        }
        Some(self.maps.principal_eigenvector()[voxel])
    }

    /// Orientation interpolated from the surrounding voxels, if one is defined.
    ///
    /// Accumulates `Σ wᵢ vᵢvᵢᵀ` over the eight trilinear neighbours that are
    /// trackable, then takes the principal eigenvector of the sum. Neighbours
    /// outside the grid or outside the mask contribute nothing rather than
    /// contributing a zero orientation, which would pull the result toward an
    /// axis that no voxel actually holds.
    ///
    /// `None` when no dominant axis survives — the contributing orientations
    /// disagree, which is what a fibre crossing looks like to a single-tensor
    /// model.
    fn interpolated(&self, index: &Point<3>) -> Option<[f64; 3]> {
        let mut dyadic = [0.0_f64; 6];
        let mut total = 0.0_f64;

        for corner in 0..8_usize {
            let mut weight = 1.0_f64;
            let mut voxel = 0_usize;
            for (axis, extent) in self.shape.iter().enumerate() {
                let base = index[axis].floor();
                let fraction = index[axis] - base;
                let upper = corner & (1 << axis) != 0;
                weight *= if upper { fraction } else { 1.0 - fraction };

                let coordinate = base + f64::from(u8::from(upper));
                if !(0.0..*extent as f64).contains(&coordinate) {
                    weight = 0.0;
                    break;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the range check immediately above admits only valid indices"
                )]
                let coordinate = coordinate as usize;
                voxel = voxel * extent + coordinate;
            }
            if weight <= 0.0 {
                continue;
            }
            let Some([x, y, z]) = self.trackable(voxel) else {
                continue;
            };

            // Voigt order [xx, yy, zz, xy, xz, yz], matching the decomposition.
            for (slot, product) in dyadic
                .iter_mut()
                .zip([x * x, y * y, z * z, x * y, x * z, y * z])
            {
                *slot += weight * product;
            }
            total += weight;
        }

        if total <= 0.0 {
            return None;
        }

        let eigen = symmetric_eigen(dyadic);
        let dominant = eigen.values[0] - eigen.values[1];
        if dominant <= eigen.values[0].abs() * DEGENERATE_EIGENVALUE_GAP {
            return None;
        }
        Some(eigen.vectors[0])
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

/// Normalise a stored orientation, rejecting a degenerate one.
fn unit(direction: [f64; 3]) -> Option<Vector<3>> {
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

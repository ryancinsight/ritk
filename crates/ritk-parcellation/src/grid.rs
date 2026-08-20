//! Where a parcellation's voxels sit in physical space.
//!
//! A label volume is only useful if a physical point can be mapped to the voxel
//! containing it — that is the whole of what a streamline endpoint, a seed, or a
//! registered atlas needs from it. The mapping is the standard medical-imaging
//! affine:
//!
//! ```text
//! p = origin + R · (s ⊙ i)
//! ```
//!
//! where `i` is the continuous voxel index, `s` the per-axis spacing, `R` the
//! direction cosine matrix whose columns are the physical directions of the
//! index axes, and `origin` the physical position of voxel `(0, 0, 0)`'s centre.
//!
//! # Why the direction matrix is not optional
//!
//! Reducing the mapping to `origin + s ⊙ i` — treating the index axes as
//! parallel to the physical ones — is correct only for an axis-aligned volume.
//! Almost no acquired volume is: a scanner stores an oblique slice stack, and
//! `qform`/`sform` in NIfTI, the space directions in NRRD, and the DICOM image
//! orientation all exist to record that obliquity. Dropping it does not fail
//! loudly; it silently returns the label of a *different* region, displaced by
//! up to the volume's extent times the sine of the obliquity. Connectome edges
//! built on such lookups are wrong in a way no downstream check catches, because
//! every label involved is a legitimate label.
//!
//! So the grid carries the full affine, and an axis-aligned volume is the
//! special case where `R` is the identity rather than the only case supported.

use ritk_spatial::{Direction, Point};
use serde::{Deserialize, Serialize};

use crate::ParcellationError;

/// Voxel grid geometry: where each index sits in physical space.
///
/// Constructed through [`Self::new`], which rejects a geometry that cannot
/// describe a volume — a zero extent, a non-finite or zero spacing, or a
/// direction matrix that is not invertible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParcellationGrid {
    shape: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    /// Direction cosines, row-major; column `k` is the physical direction of
    /// index axis `k`.
    direction: [f64; 9],
    /// `(R · diag(s))⁻¹`, row-major.
    ///
    /// Cached because the inverse is what every lookup needs and the forward
    /// form is what every file format supplies; recomputing it per query would
    /// put a 3×3 inversion inside the per-streamline path.
    inverse: [f64; 9],
}

impl ParcellationGrid {
    /// Build a grid from its shape, spacing, origin, and direction cosines.
    ///
    /// `direction` is row-major with the physical direction of index axis `k` in
    /// column `k` — the convention NIfTI's `sform`, NRRD's space directions, and
    /// DICOM's image orientation all reduce to.
    ///
    /// # Errors
    ///
    /// [`ParcellationError::DegenerateGrid`] when any extent is zero, any
    /// spacing is not finite and positive, any origin or direction entry is not
    /// finite, or the direction matrix is singular.
    pub fn new(
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        direction: [f64; 9],
    ) -> Result<Self, ParcellationError> {
        if shape.contains(&0) {
            return Err(ParcellationError::DegenerateGrid {
                reason: "every axis must have a nonzero extent",
            });
        }
        if spacing
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ParcellationError::DegenerateGrid {
                reason: "every spacing must be finite and positive",
            });
        }
        if origin
            .iter()
            .chain(direction.iter())
            .any(|v| !v.is_finite())
        {
            return Err(ParcellationError::DegenerateGrid {
                reason: "origin and direction entries must be finite",
            });
        }

        // The full index-to-physical linear part folds spacing into the
        // direction cosines, so one inversion serves both.
        let mut linear = [0.0_f64; 9];
        for row in 0..3 {
            for column in 0..3 {
                linear[row * 3 + column] = direction[row * 3 + column] * spacing[column];
            }
        }
        let inverse = invert_3x3(linear).ok_or(ParcellationError::DegenerateGrid {
            reason: "the direction matrix is singular, so no physical point maps to an index",
        })?;

        Ok(Self {
            shape,
            spacing,
            origin,
            direction,
            inverse,
        })
    }

    /// An axis-aligned grid — the direction matrix is the identity.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn axis_aligned(
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
    ) -> Result<Self, ParcellationError> {
        Self::new(shape, spacing, origin, IDENTITY)
    }

    /// Grid extent `[nx, ny, nz]` in voxels.
    #[must_use]
    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Voxel size `[sx, sy, sz]` in mm.
    #[must_use]
    pub const fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Physical position of the first voxel's centre.
    #[must_use]
    pub const fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Direction cosines, row-major.
    #[must_use]
    pub const fn direction(&self) -> [f64; 9] {
        self.direction
    }

    /// Total voxel count.
    #[must_use]
    pub const fn voxel_count(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    /// Volume of one voxel in mm³.
    ///
    /// The determinant of the direction matrix is `±1` for an orthonormal frame,
    /// so this is the product of the spacings; the absolute determinant is
    /// carried anyway so that a sheared or non-unit direction matrix — which a
    /// resampled atlas can present — still gives the right volume.
    #[must_use]
    pub fn voxel_volume(&self) -> f64 {
        let scale = self.spacing[0] * self.spacing[1] * self.spacing[2];
        scale * determinant_3x3(self.direction).abs()
    }

    /// Continuous voxel index of a physical point.
    ///
    /// Returns `None` when any coordinate is not finite. The result is *not*
    /// bounded to the grid: a point outside the volume maps to an index outside
    /// it, which is the information the caller needs to decide what to do.
    #[must_use]
    pub fn continuous_index_of(&self, point: &Point<3>) -> Option<[f64; 3]> {
        let [px, py, pz] = point.to_array();
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            return None;
        }
        let offset = [
            px - self.origin[0],
            py - self.origin[1],
            pz - self.origin[2],
        ];
        Some(apply_3x3(self.inverse, offset))
    }

    /// Physical position of a voxel centre.
    #[must_use]
    pub fn physical_point_of(&self, index: [usize; 3]) -> Point<3> {
        self.physical_point_of_continuous(index.map(|value| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid extents are far below f64's exact-integer range"
            )]
            let coordinate = value as f64;
            coordinate
        }))
    }

    /// Physical position of a continuous voxel index.
    #[must_use]
    pub fn physical_point_of_continuous(&self, index: [f64; 3]) -> Point<3> {
        let scaled = [
            index[0] * self.spacing[0],
            index[1] * self.spacing[1],
            index[2] * self.spacing[2],
        ];
        let rotated = apply_3x3(self.direction, scaled);
        Point::new([
            self.origin[0] + rotated[0],
            self.origin[1] + rotated[1],
            self.origin[2] + rotated[2],
        ])
    }

    /// Nearest in-bounds voxel index for a physical point.
    ///
    /// Returns `None` when the point is outside the volume or not finite.
    /// Rounding is applied before the bounds test, so a point within half a
    /// voxel of the edge still lands on the edge voxel — the volume covers half
    /// a voxel beyond the outermost centre, and excluding that band would carve
    /// a hole out of every surface.
    #[must_use]
    pub fn voxel_of(&self, point: &Point<3>) -> Option<[usize; 3]> {
        let continuous = self.continuous_index_of(point)?;
        self.clamp_to_grid(continuous)
    }

    /// Bounds-test a rounded continuous index.
    fn clamp_to_grid(&self, continuous: [f64; 3]) -> Option<[usize; 3]> {
        let mut index = [0_usize; 3];
        for axis in 0..3 {
            let rounded = continuous[axis].round();
            if !rounded.is_finite() || rounded < 0.0 {
                return None;
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the value is finite, nonnegative, and bounds-checked against the extent below"
            )]
            let candidate = rounded as usize;
            if candidate >= self.shape[axis] {
                return None;
            }
            index[axis] = candidate;
        }
        Some(index)
    }

    /// Flat offset of a voxel index in the crate's z-major storage order.
    ///
    /// Returns `None` when the index is outside the grid.
    #[must_use]
    pub const fn offset_of(&self, index: [usize; 3]) -> Option<usize> {
        let [nx, ny, nz] = self.shape;
        let [ix, iy, iz] = index;
        if ix >= nx || iy >= ny || iz >= nz {
            return None;
        }
        Some(iz * ny * nx + iy * nx + ix)
    }

    /// Voxel index of a flat offset — the inverse of [`Self::offset_of`].
    ///
    /// Returns `None` when the offset is past the end of the volume.
    #[must_use]
    pub const fn index_of_offset(&self, offset: usize) -> Option<[usize; 3]> {
        if offset >= self.voxel_count() {
            return None;
        }
        let [nx, ny, _] = self.shape;
        let plane = nx * ny;
        Some([offset % nx, (offset % plane) / nx, offset / plane])
    }

    /// The direction cosines as a [`Direction`], for callers composing this grid
    /// with a spatial transform.
    #[must_use]
    pub fn direction_matrix(&self) -> Direction<3> {
        Direction::from_row_major(self.direction)
    }

    /// Squared physical length of an integer index offset, in mm².
    ///
    /// Returned squared because the callers that need it are comparing
    /// distances, and the square root is the expensive half of the comparison.
    #[must_use]
    pub fn physical_displacement_of(&self, offset: [isize; 3]) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "offsets are bounded by the grid extent, far below f64's exact-integer range"
        )]
        let steps = offset.map(|value| value as f64);
        let scaled = [
            steps[0] * self.spacing[0],
            steps[1] * self.spacing[1],
            steps[2] * self.spacing[2],
        ];
        let displacement = apply_3x3(self.direction, scaled);
        displacement.iter().map(|value| value * value).sum()
    }

    /// Largest index offset per axis that any point within `radius` mm of a
    /// voxel centre can reach.
    ///
    /// A neighbourhood search enumerates integer offsets and must not miss one.
    /// The offset of a physical displacement `p` is `L⁻¹p` with `L = R·diag(s)`,
    /// so component `k` is bounded by the Cauchy-Schwarz inequality:
    ///
    /// ```text
    /// |(L⁻¹p)ₖ| ≤ ‖rowₖ(L⁻¹)‖ · ‖p‖ ≤ ‖rowₖ(L⁻¹)‖ · radius
    /// ```
    ///
    /// The bound is therefore exact for an orthonormal direction matrix — where
    /// the row norm is `1/sₖ` and it reduces to `radius/sₖ` — and stays a valid
    /// over-estimate for a sheared one, which is the direction an enumeration
    /// bound must err in. Callers still test the true physical distance per
    /// offset, so over-inclusion costs work rather than correctness.
    ///
    /// A non-finite or negative radius yields no neighbourhood at all.
    #[must_use]
    pub fn index_radius_bounds(&self, radius: f64) -> [usize; 3] {
        if !radius.is_finite() || radius < 0.0 {
            return [0; 3];
        }
        let mut bounds = [0_usize; 3];
        for (axis, bound) in bounds.iter_mut().enumerate() {
            let row = &self.inverse[axis * 3..axis * 3 + 3];
            let row_norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            let extent = (radius * row_norm).ceil();
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the product of two nonnegative finite values, clamped to the grid extent"
            )]
            let steps = extent.min(self.shape[axis] as f64) as usize;
            *bound = steps;
        }
        bounds
    }
}

/// Row-major 3×3 identity.
const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn apply_3x3(matrix: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}

fn determinant_3x3(m: [f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Invert a row-major 3×3 matrix by its adjugate, or `None` when singular.
///
/// Singularity is judged relative to the matrix's own magnitude rather than
/// against a fixed epsilon, so a grid expressed in metres classifies the same as
/// the identical grid expressed in millimetres.
fn invert_3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let determinant = determinant_3x3(m);
    let magnitude = m.iter().fold(0.0_f64, |peak, value| peak.max(value.abs()));
    if !determinant.is_finite() || determinant.abs() <= magnitude.powi(3) * f64::EPSILON {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inverse_determinant,
        (m[2] * m[7] - m[1] * m[8]) * inverse_determinant,
        (m[1] * m[5] - m[2] * m[4]) * inverse_determinant,
        (m[5] * m[6] - m[3] * m[8]) * inverse_determinant,
        (m[0] * m[8] - m[2] * m[6]) * inverse_determinant,
        (m[2] * m[3] - m[0] * m[5]) * inverse_determinant,
        (m[3] * m[7] - m[4] * m[6]) * inverse_determinant,
        (m[1] * m[6] - m[0] * m[7]) * inverse_determinant,
        (m[0] * m[4] - m[1] * m[3]) * inverse_determinant,
    ])
}

#[cfg(test)]
mod tests;

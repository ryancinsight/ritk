//! Finding the nearest labelled voxel to a point.
//!
//! # The problem this solves
//!
//! Assigning a streamline endpoint to a region by looking up the label directly
//! under it discards most of the tractogram. Streamlines are tracked through
//! white matter and stop where the orientation field stops being coherent, which
//! is at or just short of the grey-matter boundary — while a cortical
//! parcellation labels the grey matter and leaves the white matter background.
//! The endpoint therefore lands in a region-less voxel, and the streamline is
//! dropped despite ending exactly where it should.
//!
//! Searching a small neighbourhood recovers those. The endpoint is assigned to
//! the nearest labelled voxel within a radius, so a streamline stopping a
//! millimetre short of the cortical ribbon is attributed to the parcel it was
//! heading into rather than discarded.
//!
//! # Why the radius is the caller's decision, and what it costs
//!
//! The radius trades recall against specificity, and the trade is not free in
//! either direction. Too small and the tractogram is decimated. Too large and
//! endpoints reach past their own gyrus into a neighbouring parcel across a
//! sulcus — anatomically adjacent, functionally unconnected — and manufacture
//! edges that no fibre supports. The usable range is set by the local anatomy:
//! cortical thickness is 2–4 mm and sulcal walls approach within a millimetre of
//! each other, so a radius of a few millimetres is where the two failure modes
//! balance. Nothing here picks a value, because nothing here knows the
//! parcellation's resolution or how the streamlines were terminated.
//!
//! Note that the search cannot distinguish "the nearest parcel across the sulcus"
//! from "the parcel this fibre actually entered": it measures distance, not
//! connectivity. That is a limitation of the method rather than of this
//! implementation, and it is the reason the assignment is reported with the
//! distance it required.

use ritk_spatial::Point;

use crate::{BACKGROUND, Parcellation, ParcellationError, ParcellationGrid};

/// A labelled voxel found near a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestLabel {
    /// The region label found.
    pub label: u32,
    /// Voxel index it was found at.
    pub index: [usize; 3],
    /// Physical distance in mm from the query point to that voxel's centre.
    ///
    /// Zero when the query point already sat in a labelled voxel. Reporting it
    /// lets a caller weigh or reject an assignment that only just fell inside
    /// the radius.
    pub distance: f64,
}

/// A neighbourhood search prepared for one grid and radius.
///
/// The offsets within the radius are enumerated once and sorted by physical
/// distance, so a query walks them in order and stops at the first labelled
/// voxel. Preparing the search separately from running it is what keeps a
/// per-streamline loop from re-deriving the same neighbourhood for every
/// endpoint; a whole-brain tractogram runs this a million times.
#[derive(Debug, Clone)]
pub struct NearestLabelSearch {
    /// Index offsets and the physical distance between the centres they join,
    /// ascending by that distance.
    offsets: Box<[([isize; 3], f64)]>,
    radius: f64,
}

impl NearestLabelSearch {
    /// Prepare a search over `grid` out to `radius` millimetres.
    ///
    /// A radius of zero prepares the degenerate search that inspects only the
    /// voxel containing the point, which is the exact endpoint lookup.
    ///
    /// # Errors
    ///
    /// [`ParcellationError::InvalidRadius`] when the radius is not finite and
    /// nonnegative.
    pub fn new(grid: &ParcellationGrid, radius: f64) -> Result<Self, ParcellationError> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(ParcellationError::InvalidRadius { value: radius });
        }

        // Offsets are enumerated between voxel *centres*, but the radius is
        // measured from the query point, which sits anywhere inside its voxel.
        // The two differ by up to half a voxel diagonal, so the enumeration
        // reaches that much further than the radius — otherwise a voxel whose
        // centre is genuinely within the radius of the point, but more than
        // `radius` from the containing voxel's centre, is never even considered.
        let reach = radius + half_voxel_diagonal(grid);
        let bounds = grid.index_radius_bounds(reach);
        let mut offsets = Vec::new();

        #[expect(
            clippy::cast_possible_wrap,
            reason = "bounds are clamped to the grid extent, far below isize::MAX"
        )]
        let limits = bounds.map(|bound| bound as isize);
        for dz in -limits[2]..=limits[2] {
            for dy in -limits[1]..=limits[1] {
                for dx in -limits[0]..=limits[0] {
                    // The offset's physical length is measured through the grid's
                    // own affine, so an anisotropic or oblique volume gets true
                    // millimetres rather than a voxel count wearing a mm label.
                    let distance = grid.physical_displacement_of([dx, dy, dz]).sqrt();
                    if distance <= reach {
                        offsets.push(([dx, dy, dz], distance));
                    }
                }
            }
        }
        offsets.sort_by(|(_, left), (_, right)| left.total_cmp(right));

        Ok(Self {
            offsets: offsets.into_boxed_slice(),
            radius,
        })
    }

    /// The radius this search was prepared for, in mm.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Number of voxel offsets the search inspects in the worst case.
    #[must_use]
    pub const fn neighbourhood_size(&self) -> usize {
        self.offsets.len()
    }

    /// The nearest labelled voxel to `point`.
    ///
    /// A voxel qualifies when its centre lies within the radius of `point`. The
    /// voxel *containing* the point always qualifies, whatever the radius —
    /// otherwise a zero radius would admit nothing at all, since a point is
    /// rarely exactly on a voxel centre.
    ///
    /// Returns `None` when the point is not finite, falls outside the grid, or
    /// has no qualifying labelled voxel.
    ///
    /// # Why this is not simply the first hit in offset order
    ///
    /// The offsets are ordered by the distance between voxel *centres*, but what
    /// is reported and compared is the distance from the *point*, which sits
    /// anywhere inside its voxel. The two orders disagree by up to half a voxel
    /// diagonal, so the first labelled voxel encountered can be further from the
    /// point than one reached later — and for an endpoint near a parcel
    /// boundary, that is the difference between the parcel it is in and the one
    /// across the border.
    ///
    /// So every candidate is scored and the best kept. The walk still stops
    /// early: a candidate at centre-distance `c` is at least `c − drift` from
    /// the point, where `drift` is how far the point sits from its own voxel's
    /// centre, so once `c − drift` exceeds the best distance found, nothing
    /// further can improve on it.
    ///
    /// Ties between equidistant voxels resolve to whichever the enumeration
    /// reached first, which is deterministic for a given grid and radius but
    /// carries no anatomical meaning — an endpoint exactly equidistant from two
    /// parcels is genuinely ambiguous.
    #[must_use]
    pub fn find(&self, parcellation: &Parcellation, point: &Point<3>) -> Option<NearestLabel> {
        let grid = parcellation.grid();
        let centre = grid.voxel_of(point)?;
        let drift = euclidean_distance(point, &grid.physical_point_of(centre));
        let shape = grid.shape();

        let mut best: Option<NearestLabel> = None;
        for (offset, centre_distance) in &self.offsets {
            if let Some(found) = &best
                && centre_distance - drift > found.distance
            {
                break;
            }
            let Some(index) = shifted_index(centre, *offset, shape) else {
                continue;
            };
            let Some(label) = parcellation.label_at_index(index) else {
                continue;
            };
            if label == BACKGROUND {
                continue;
            }
            let distance = euclidean_distance(point, &grid.physical_point_of(index));
            let containing = *offset == [0; 3];
            if !containing && distance > self.radius {
                continue;
            }
            if best.as_ref().is_none_or(|found| distance < found.distance) {
                best = Some(NearestLabel {
                    label,
                    index,
                    distance,
                });
            }
        }
        best
    }
}

/// Half the distance across a voxel, corner to corner, in mm.
///
/// The most a query point can sit from the centre of the voxel containing it,
/// and therefore the gap between the centre-to-centre order the offsets are
/// enumerated in and the point-to-centre distances [`NearestLabelSearch::find`]
/// compares.
fn half_voxel_diagonal(grid: &ParcellationGrid) -> f64 {
    0.5 * grid.physical_displacement_of([1, 1, 1]).sqrt()
}

/// Apply an offset to a voxel index, or `None` when it leaves the grid.
fn shifted_index(centre: [usize; 3], offset: [isize; 3], shape: [usize; 3]) -> Option<[usize; 3]> {
    let mut index = [0_usize; 3];
    for axis in 0..3 {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "grid extents are far below isize::MAX"
        )]
        let shifted = centre[axis] as isize + offset[axis];
        #[expect(
            clippy::cast_possible_wrap,
            reason = "grid extents are far below isize::MAX"
        )]
        let extent = shape[axis] as isize;
        if shifted < 0 || shifted >= extent {
            return None;
        }
        #[expect(
            clippy::cast_sign_loss,
            reason = "bounds-checked nonnegative immediately above"
        )]
        let in_range = shifted as usize;
        index[axis] = in_range;
    }
    Some(index)
}

fn euclidean_distance(left: &Point<3>, right: &Point<3>) -> f64 {
    let [lx, ly, lz] = left.to_array();
    let [rx, ry, rz] = right.to_array();
    ((lx - rx).powi(2) + (ly - ry).powi(2) + (lz - rz).powi(2)).sqrt()
}

#[cfg(test)]
mod tests;

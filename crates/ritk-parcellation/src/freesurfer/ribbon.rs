//! Rasterising a surface annotation into a volumetric parcellation.
//!
//! A [`SurfaceAnnotation`] labels vertices of a mesh; a [`Parcellation`] labels
//! voxels. Bridging them means deciding which voxels a labelled vertex owns, and
//! the answer is the *cortical ribbon*: the sheet of grey matter between the
//! white-matter surface and the pial surface, which the two meshes bound.
//!
//! # Method
//!
//! The two surfaces share a vertex numbering — vertex `i` of `lh.white` and
//! vertex `i` of `lh.pial` are the inner and outer end of the same cortical
//! column. Walking that segment and stamping the vertex's label into every voxel
//! it passes through fills the ribbon, one column at a time. This is what
//! FreeSurfer's own `mri_surf2vol --fill-ribbon` does.
//!
//! # What this is and is not
//!
//! It fills the ribbon; it does not *tessellate* it. Voxels are claimed by the
//! columns that pass through them, so a voxel no column crosses stays
//! background even if it lies geometrically inside the ribbon. That happens
//! where the mesh is coarse relative to the voxel grid — a surface sampled more
//! sparsely than the volume leaves gaps between adjacent columns.
//!
//! The remedy is sampling, not interpolation: `steps` controls how finely each
//! column is walked, and the vertex spacing controls the rest. Increasing steps
//! closes gaps *along* a column and does nothing for gaps *between* columns, so
//! a coarse mesh on a fine grid is a limitation of the input rather than of the
//! setting. [`RibbonReport::unfilled_columns`] reports columns that claimed no
//! voxel at all, which is the signal that the surface and the grid disagree in
//! resolution.
//!
//! # Contested voxels
//!
//! Two columns from different parcels can cross the same voxel, at a parcel
//! boundary or where the ribbon folds into a sulcus. The first column to claim
//! it keeps it, and the count of such collisions is reported: a large one means
//! the boundaries in the result are arbitrary at the voxel scale, which is worth
//! knowing before building a connectome whose endpoints land exactly there.
//!
//! # Frames
//!
//! Both surfaces must already be in the grid's physical frame. FreeSurfer stores
//! them in surface RAS, which differs from the scanner frame by the volume's
//! `c_ras` translation — see [`super::surface`], and apply
//! [`Surface::translated`] before calling this.

use crate::{BACKGROUND, Parcellation, ParcellationError, ParcellationGrid};

use super::{FreeSurferSurfaceError, Surface, SurfaceAnnotation};

/// How much of the ribbon the rasterisation managed to fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RibbonReport {
    /// Cortical columns walked — one per labelled vertex.
    pub columns: usize,
    /// Columns that claimed no voxel, because every step landed outside the
    /// grid or on a voxel already taken.
    ///
    /// A large share means the surface and the grid disagree in resolution, and
    /// the result has holes a denser mesh would fill.
    pub unfilled_columns: usize,
    /// Voxels a column wanted but another parcel had already claimed.
    ///
    /// Concentrated at parcel boundaries and inside sulcal folds. A large count
    /// means the boundaries in the result are arbitrary at the voxel scale.
    pub contested_voxels: usize,
    /// Voxels that ended up labelled.
    pub filled_voxels: usize,
}

/// Rasterise a cortical annotation into a parcellation on `grid`.
///
/// `white` and `pial` must share a vertex numbering — they are the inner and
/// outer surfaces of the same reconstruction — and both must be in the grid's
/// physical frame. `steps` is how many samples each cortical column is walked
/// with; a value near twice the ribbon thickness in voxels leaves no gap along
/// a column.
///
/// Vertices whose annotation label is [`BACKGROUND`] are skipped, since they
/// carry no parcel to stamp.
///
/// # Errors
///
/// [`FreeSurferSurfaceError::InvalidVertexCount`] when the two surfaces or the
/// annotation disagree on how many vertices there are — which means they are not
/// from one reconstruction, and pairing them would join unrelated points.
///
/// # Panics
///
/// Never; the parcellation error is returned rather than raised.
pub fn rasterise_ribbon(
    white: &Surface,
    pial: &Surface,
    annotation: &SurfaceAnnotation,
    grid: &ParcellationGrid,
    steps: usize,
) -> Result<(Parcellation, RibbonReport), RibbonError> {
    let vertices = white.vertex_count();
    if pial.vertex_count() != vertices || annotation.vertex_labels.len() != vertices {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "reported for diagnosis; surface vertex counts are bounded well below i32::MAX"
        )]
        let count = vertices as i32;
        return Err(RibbonError::Surface(
            FreeSurferSurfaceError::InvalidVertexCount { count },
        ));
    }

    // At least the two endpoints, so a zero or one step still walks the column
    // rather than silently sampling nothing.
    let samples = steps.max(2);
    let mut labels = vec![BACKGROUND; grid.voxel_count()];
    let mut report = RibbonReport::default();

    for (vertex, label) in annotation.vertex_labels.iter().copied().enumerate() {
        if label == BACKGROUND {
            continue;
        }
        report.columns += 1;
        let inner = white.vertices()[vertex];
        let outer = pial.vertices()[vertex];

        let mut claimed = false;
        for step in 0..samples {
            #[expect(
                clippy::cast_precision_loss,
                reason = "sample counts are small integers"
            )]
            let fraction = step as f64 / (samples - 1) as f64;
            let point = ritk_spatial::Point::new([
                inner[0] + (outer[0] - inner[0]) * fraction,
                inner[1] + (outer[1] - inner[1]) * fraction,
                inner[2] + (outer[2] - inner[2]) * fraction,
            ]);
            let Some(offset) = grid
                .voxel_of(&point)
                .and_then(|index| grid.offset_of(index))
            else {
                continue;
            };
            if labels[offset] == BACKGROUND {
                labels[offset] = label;
                report.filled_voxels += 1;
                claimed = true;
            } else if labels[offset] != label {
                // Another parcel got here first. Keeping the earlier claim is
                // arbitrary, which is why the count is reported rather than
                // resolved silently.
                report.contested_voxels += 1;
            } else {
                claimed = true;
            }
        }
        if !claimed {
            report.unfilled_columns += 1;
        }
    }

    let names = annotation.label_table.clone();
    let parcellation = Parcellation::new(labels.into_boxed_slice(), grid.clone(), names)?;
    Ok((parcellation, report))
}

/// Failure while rasterising a surface annotation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RibbonError {
    /// The surfaces or the annotation do not describe one reconstruction.
    #[error("surface error: {0}")]
    Surface(#[from] FreeSurferSurfaceError),
    /// The rasterised volume is not a usable parcellation.
    ///
    /// Reached when no column landed inside the grid at all, which means the
    /// surfaces are in a different frame from the volume — the commonest cause
    /// being surface RAS coordinates used without the `c_ras` translation.
    #[error("rasterised ribbon is not a parcellation: {0}")]
    Parcellation(#[from] ParcellationError),
}

#[cfg(test)]
mod tests;

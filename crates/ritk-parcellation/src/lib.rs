//! Anatomical parcellation volumes and the measures taken from them.
//!
//! A *parcellation* is a label volume: every voxel carries the identifier of the
//! anatomical region it belongs to, with the reserved label `0` meaning
//! background. It is the shared vocabulary between the two halves of a
//! connectomics pipeline — something produces one (an atlas registered onto a
//! subject, a segmentation, a surface annotation rasterised into a volume), and
//! something consumes one (`ritk-connectome`, region-wise statistics, targeting).
//!
//! This crate owns the type itself, its geometry, and the measures that are
//! properties of a parcellation rather than of whatever produced or consumes it.
//! It deliberately depends on nothing but spatial primitives: a vocabulary crate
//! that dragged in a registration stack would force every consumer to build one.
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`grid`] | Voxel-to-physical affine, including the direction cosines |
//! | [`regions`] | Per-region volume, centroid, and extent |
//! | [`search`] | Nearest labelled voxel to a point, within a radius |
//! | [`freesurfer`] | FreeSurfer colour lookup tables and surface annotations |
//!
//! # Example
//!
//! ```
//! use ritk_parcellation::{Parcellation, ParcellationGrid};
//! use ritk_spatial::Point;
//!
//! // A 2×1×1 volume with one voxel in each of two regions, 2 mm apart.
//! let grid = ParcellationGrid::axis_aligned([2, 1, 1], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0])?;
//! let parcellation = Parcellation::new(
//!     Box::new([1, 7]),
//!     grid,
//!     vec![(1, "Left".into()), (7, "Right".into())],
//! )?;
//!
//! assert_eq!(parcellation.region_labels(), vec![1, 7]);
//! assert_eq!(parcellation.label_at(&Point::new([2.0, 0.0, 0.0])), Some(7));
//! assert_eq!(parcellation.name_of(7), Some("Right"));
//! # Ok::<(), ritk_parcellation::ParcellationError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod freesurfer;
pub mod grid;
pub mod regions;
pub mod search;

pub use grid::ParcellationGrid;
pub use regions::RegionStatistics;
pub use search::NearestLabelSearch;

use ritk_spatial::Point;
use serde::{Deserialize, Serialize};

/// The label reserved for background — outside the brain, or unassigned.
pub const BACKGROUND: u32 = 0;

/// Failure while constructing or querying a parcellation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParcellationError {
    /// The grid geometry cannot describe a volume.
    #[error("degenerate parcellation grid: {reason}")]
    DegenerateGrid {
        /// Which part of the geometry is unusable.
        reason: &'static str,
    },
    /// The label array does not cover the grid.
    #[error("grid of {expected} voxels needs {expected} labels, got {actual}")]
    LabelCountMismatch {
        /// Voxels the grid declares.
        expected: usize,
        /// Labels supplied.
        actual: usize,
    },
    /// Every voxel is background, so there is nothing to parcellate.
    #[error("parcellation has no labelled regions (every voxel is background)")]
    EmptyParcellation,
    /// A search radius is not a usable distance.
    #[error("search radius must be finite and nonnegative, got {value}")]
    InvalidRadius {
        /// The rejected radius, in mm.
        value: f64,
    },
}

/// A labelled anatomical volume.
///
/// Labels are stored z-major — index `[ix, iy, iz]` at offset
/// `iz·ny·nx + iy·nx + ix` — matching the layout every volumetric format writes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parcellation {
    labels: Box<[u32]>,
    grid: ParcellationGrid,
    /// Human-readable names keyed by label, sorted by label for binary search.
    /// A label may have no name; a name may exist for [`BACKGROUND`].
    region_names: Box<[(u32, String)]>,
    /// Sorted, deduplicated non-background labels.
    ///
    /// Held rather than recomputed because every region-wise operation needs it
    /// and deriving it is a full pass over the volume.
    region_labels: Box<[u32]>,
}

impl Parcellation {
    /// Build a parcellation from a flat label array and its grid.
    ///
    /// # Errors
    ///
    /// [`ParcellationError::LabelCountMismatch`] when the array does not cover
    /// the grid, and [`ParcellationError::EmptyParcellation`] when every voxel
    /// is [`BACKGROUND`] — a parcellation with no regions cannot answer any
    /// question asked of it, so it is rejected at construction rather than
    /// returning empty results later.
    pub fn new(
        labels: Box<[u32]>,
        grid: ParcellationGrid,
        region_names: Vec<(u32, String)>,
    ) -> Result<Self, ParcellationError> {
        let expected = grid.voxel_count();
        if labels.len() != expected {
            return Err(ParcellationError::LabelCountMismatch {
                expected,
                actual: labels.len(),
            });
        }

        let mut region_labels: Vec<u32> = labels
            .iter()
            .copied()
            .filter(|label| *label != BACKGROUND)
            .collect();
        region_labels.sort_unstable();
        region_labels.dedup();
        if region_labels.is_empty() {
            return Err(ParcellationError::EmptyParcellation);
        }

        let mut region_names = region_names;
        region_names.sort_by_key(|(label, _)| *label);
        region_names.dedup_by_key(|(label, _)| *label);

        Ok(Self {
            labels,
            grid,
            region_names: region_names.into_boxed_slice(),
            region_labels: region_labels.into_boxed_slice(),
        })
    }

    /// The voxel grid this parcellation lives on.
    #[must_use]
    pub const fn grid(&self) -> &ParcellationGrid {
        &self.grid
    }

    /// Raw label array, z-major.
    #[must_use]
    pub const fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Sorted, deduplicated non-background region labels.
    #[must_use]
    pub fn region_labels(&self) -> Vec<u32> {
        self.region_labels.to_vec()
    }

    /// Sorted, deduplicated non-background region labels, borrowed.
    #[must_use]
    pub const fn region_label_slice(&self) -> &[u32] {
        &self.region_labels
    }

    /// Number of distinct non-background regions.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.region_labels.len()
    }

    /// Whether a label is present in this parcellation.
    #[must_use]
    pub fn contains_region(&self, label: u32) -> bool {
        self.region_labels.binary_search(&label).is_ok()
    }

    /// Every `(label, name)` pair supplied at construction, sorted by label.
    #[must_use]
    pub const fn region_names(&self) -> &[(u32, String)] {
        &self.region_names
    }

    /// Human-readable name of a region, when one was supplied.
    #[must_use]
    pub fn name_of(&self, label: u32) -> Option<&str> {
        self.region_names
            .binary_search_by_key(&label, |(candidate, _)| *candidate)
            .ok()
            .map(|position| self.region_names[position].1.as_str())
    }

    /// Label at a voxel index, or `None` when the index is outside the grid.
    #[must_use]
    pub fn label_at_index(&self, index: [usize; 3]) -> Option<u32> {
        self.grid.offset_of(index).map(|offset| self.labels[offset])
    }

    /// Label at a physical point by nearest-neighbour lookup.
    ///
    /// Returns `None` when the point falls outside the volume or is not finite;
    /// returns `Some(BACKGROUND)` when it falls on an unlabelled voxel. The two
    /// are distinct answers — outside the field of view is not the same claim as
    /// inside it but unassigned — so they are not collapsed.
    #[must_use]
    pub fn label_at(&self, point: &Point<3>) -> Option<u32> {
        let index = self.grid.voxel_of(point)?;
        self.label_at_index(index)
    }

    /// Per-region volume, centroid, and extent — see [`regions`].
    #[must_use]
    pub fn region_statistics(&self) -> Vec<RegionStatistics> {
        regions::region_statistics(self)
    }

    /// Statistics for one region, or `None` when the label is absent.
    #[must_use]
    pub fn statistics_of(&self, label: u32) -> Option<RegionStatistics> {
        regions::region_statistics_of(self, label)
    }

    /// A copy with every label passed through `remap`.
    ///
    /// The mapping is total, so a label the caller does not mention keeps its
    /// value; returning [`BACKGROUND`] removes a region. This is how a
    /// fine-grained atlas is collapsed to a coarser one — merging cortical
    /// parcels into lobes, or the two hemispheres' matching parcels into one —
    /// without rewriting the volume by hand.
    ///
    /// # Errors
    ///
    /// [`ParcellationError::EmptyParcellation`] when the mapping sends every
    /// voxel to background.
    pub fn remap_labels(
        &self,
        remap: impl Fn(u32) -> u32,
        region_names: Vec<(u32, String)>,
    ) -> Result<Self, ParcellationError> {
        let labels: Box<[u32]> = self
            .labels
            .iter()
            .map(|label| {
                if *label == BACKGROUND {
                    BACKGROUND
                } else {
                    remap(*label)
                }
            })
            .collect();
        Self::new(labels, self.grid.clone(), region_names)
    }

    /// A copy keeping only the listed regions, with everything else background.
    ///
    /// # Errors
    ///
    /// [`ParcellationError::EmptyParcellation`] when none of the listed labels
    /// is present.
    pub fn retain_regions(&self, keep: &[u32]) -> Result<Self, ParcellationError> {
        let mut kept: Vec<u32> = keep.to_vec();
        kept.sort_unstable();
        let names: Vec<(u32, String)> = self
            .region_names
            .iter()
            .filter(|(label, _)| kept.binary_search(label).is_ok())
            .cloned()
            .collect();
        self.remap_labels(
            |label| {
                if kept.binary_search(&label).is_ok() {
                    label
                } else {
                    BACKGROUND
                }
            },
            names,
        )
    }
}

#[cfg(test)]
mod tests;

//! Per-region size, position, and extent.
//!
//! These are the measures a parcellation answers about itself, independent of
//! what produced it. Two of them are load bearing beyond description:
//!
//! * **Volume** normalises a connectome. A raw streamline count between two
//!   regions grows with how big those regions are, because a larger region
//!   presents a larger surface for streamlines to terminate on. Comparing a raw
//!   count across regions of different size, or across subjects whose regions
//!   differ in size, therefore compares anatomy to arithmetic. Dividing by the
//!   node volumes removes the leading part of that dependence.
//! * **Centroid** turns a region into a target point — the position a plan, a
//!   report, or a rendering refers the region by.
//!
//! # Centroid convention
//!
//! The centroid is the mean of the region's voxel *centres* in physical space.
//! For a concave or disconnected region that point can lie outside the region
//! itself, which is a property of centroids and not a defect: a C-shaped gyrus
//! has its centre of mass in the gap. A caller needing a point guaranteed to be
//! inside the region wants a medial or deepest-point query, which is a different
//! computation and not what this reports.

use ritk_spatial::Point;

use crate::{BACKGROUND, Parcellation};

/// Size, position, and extent of one labelled region.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionStatistics {
    label: u32,
    voxel_count: usize,
    volume: f64,
    centroid: Point<3>,
    lower_index: [usize; 3],
    upper_index: [usize; 3],
}

impl RegionStatistics {
    /// The region's label.
    #[must_use]
    pub const fn label(&self) -> u32 {
        self.label
    }

    /// Number of voxels carrying this label.
    #[must_use]
    pub const fn voxel_count(&self) -> usize {
        self.voxel_count
    }

    /// Region volume in mm³ — voxel count times the grid's voxel volume.
    #[must_use]
    pub const fn volume(&self) -> f64 {
        self.volume
    }

    /// Mean of the region's voxel centres, in physical space.
    #[must_use]
    pub const fn centroid(&self) -> &Point<3> {
        &self.centroid
    }

    /// Lowest voxel index the region occupies on each axis, inclusive.
    #[must_use]
    pub const fn lower_index(&self) -> [usize; 3] {
        self.lower_index
    }

    /// Highest voxel index the region occupies on each axis, inclusive.
    #[must_use]
    pub const fn upper_index(&self) -> [usize; 3] {
        self.upper_index
    }

    /// Extent of the axis-aligned index bounding box, in voxels per axis.
    #[must_use]
    pub const fn extent(&self) -> [usize; 3] {
        [
            self.upper_index[0] - self.lower_index[0] + 1,
            self.upper_index[1] - self.lower_index[1] + 1,
            self.upper_index[2] - self.lower_index[2] + 1,
        ]
    }
}

/// Statistics for every non-background region, ordered by label.
///
/// Computed in one pass over the volume rather than one pass per region: a
/// whole-brain atlas has of order a hundred regions and a million voxels, so the
/// per-region form would be a hundred million reads to answer a question one
/// million answer.
pub(crate) fn region_statistics(parcellation: &Parcellation) -> Vec<RegionStatistics> {
    let labels = parcellation.region_label_slice();
    let mut accumulators: Vec<Accumulator> = labels.iter().map(|_| Accumulator::new()).collect();

    accumulate(parcellation, |label, index| {
        if let Ok(position) = labels.binary_search(&label) {
            accumulators[position].add(index);
        }
    });

    labels
        .iter()
        .zip(accumulators)
        .filter_map(|(label, accumulator)| accumulator.finish(*label, parcellation))
        .collect()
}

/// Statistics for a single region, or `None` when the label is absent.
pub(crate) fn region_statistics_of(
    parcellation: &Parcellation,
    label: u32,
) -> Option<RegionStatistics> {
    if label == BACKGROUND || !parcellation.contains_region(label) {
        return None;
    }
    let mut accumulator = Accumulator::new();
    accumulate(parcellation, |candidate, index| {
        if candidate == label {
            accumulator.add(index);
        }
    });
    accumulator.finish(label, parcellation)
}

/// Walk every labelled voxel, handing its label and index to `visit`.
fn accumulate(parcellation: &Parcellation, mut visit: impl FnMut(u32, [usize; 3])) {
    let [nx, ny, nz] = parcellation.grid().shape();
    let labels = parcellation.labels();
    // Iterated in storage order so the walk is sequential in memory; the index
    // is reconstructed from the loop counters rather than recomputed per voxel.
    let mut offset = 0_usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let label = labels[offset];
                offset += 1;
                if label != BACKGROUND {
                    visit(label, [ix, iy, iz]);
                }
            }
        }
    }
}

/// Running totals for one region.
struct Accumulator {
    count: usize,
    index_sum: [f64; 3],
    lower: [usize; 3],
    upper: [usize; 3],
}

impl Accumulator {
    const fn new() -> Self {
        Self {
            count: 0,
            index_sum: [0.0; 3],
            lower: [usize::MAX; 3],
            upper: [0; 3],
        }
    }

    fn add(&mut self, index: [usize; 3]) {
        self.count += 1;
        for (axis, position) in index.into_iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid extents are far below f64's exact-integer range"
            )]
            let coordinate = position as f64;
            self.index_sum[axis] += coordinate;
            self.lower[axis] = self.lower[axis].min(position);
            self.upper[axis] = self.upper[axis].max(position);
        }
    }

    /// `None` when the region turned out to be empty, which the caller's label
    /// list makes impossible but which keeps this total.
    fn finish(self, label: u32, parcellation: &Parcellation) -> Option<RegionStatistics> {
        if self.count == 0 {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "voxel counts are far below f64's exact-integer range"
        )]
        let count = self.count as f64;
        // The centroid is taken in index space and mapped once, rather than
        // mapping every voxel and averaging: the transform is affine, so the two
        // agree exactly, and this does one matrix application instead of N.
        let mean_index = self.index_sum.map(|total| total / count);
        Some(RegionStatistics {
            label,
            voxel_count: self.count,
            volume: count * parcellation.grid().voxel_volume(),
            centroid: parcellation.grid().physical_point_of_continuous(mean_index),
            lower_index: self.lower,
            upper_index: self.upper,
        })
    }
}

#[cfg(test)]
mod tests;

//! Turning streamline endpoints into weighted edges.
//!
//! Two decisions separate a tractogram from a connectome, and both change the
//! answer.
//!
//! # 1. Which region does an endpoint belong to?
//!
//! Reading the label directly under the endpoint discards most of a tractogram.
//! Streamlines are tracked through white matter and terminate where the
//! orientation field stops being coherent — at or just short of the grey-matter
//! boundary — while a cortical parcellation labels grey matter and leaves white
//! matter background. The endpoint therefore lands on an unlabelled voxel, and
//! the streamline is dropped despite ending exactly where it should. Rejection
//! rates above half are ordinary with terminal assignment.
//!
//! [`EndpointAssignment::RadialSearch`] recovers those by assigning the endpoint
//! to the nearest labelled voxel within a radius. What it cannot do is tell the
//! parcel a fibre entered from the parcel across the sulcus: it measures
//! distance, not connectivity, so an over-wide radius manufactures edges. The
//! radius is the caller's, and [`ritk_parcellation::search`] sets out the
//! trade.
//!
//! # 2. What does an edge weigh?
//!
//! A raw streamline count is not proportional to anything anatomical. Two of its
//! dependences are geometric and can be divided out:
//!
//! * **Pathway length.** A streamline is reconstructed step by step, and the
//!   probability of it surviving to its endpoint falls with the number of steps.
//!   Long-range connections are therefore systematically under-counted relative
//!   to short ones, by roughly the pathway length.
//!   [`EdgeWeighting::InverseLength`] divides it back out.
//! * **Region size.** A large region presents a large surface for streamlines to
//!   terminate on, so its edges are heavier for reasons of anatomy-independent
//!   geometry. This confounds comparison between regions of different size, and
//!   between subjects whose regions differ in size.
//!   [`EdgeWeighting::InverseNodeVolume`] divides by the node volumes.
//!
//! [`EdgeWeighting::MeanLength`] is not a connectivity measure at all — it
//! reports the mean length of the streamlines forming each edge, which is a
//! geometric description of the pathway rather than a count of it.
//!
//! None of these makes a weight a measurement of connection strength. They
//! remove known confounds; the residual bias of the tracking algorithm remains.
//!
//! # References
//!
//! * Hagmann, P., Cammoun, L., Gigandet, X., Meuli, R., Honey, C. J., Wedeen,
//!   V. J. & Sporns, O. (2008). Mapping the structural core of human cerebral
//!   cortex. *PLoS Biology* 6(7):e159. — length and surface normalisation.
//! * Jones, D. K., Knösche, T. R. & Turner, R. (2013). White matter integrity,
//!   fiber count, and other fallacies: The do's and don'ts of diffusion MRI.
//!   *NeuroImage* 73:239–254. — why a count is not a strength.

use gaia::Polyline;
use ritk_parcellation::{NearestLabelSearch, Parcellation};
use ritk_spatial::Point;
use serde::{Deserialize, Serialize};

use crate::{ConnectivityMatrix, ConnectomeError, StreamlineAccounting};

/// How a streamline endpoint is attributed to a region.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum EndpointAssignment {
    /// The label of the voxel the endpoint falls in.
    ///
    /// Exact, and exactly what discards streamlines terminating in white matter.
    /// Correct when the parcellation covers the whole brain including white
    /// matter, or when the streamlines were tracked to terminate inside grey
    /// matter by construction.
    #[default]
    Terminal,

    /// The nearest labelled voxel within `radius_mm` of the endpoint.
    ///
    /// The endpoint's own voxel is searched first, so this can only add
    /// assignments that [`Self::Terminal`] would have dropped — it never moves
    /// an endpoint that already sat inside a region.
    RadialSearch {
        /// Search radius in millimetres.
        radius_mm: f64,
    },
}

/// What an edge weight counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EdgeWeighting {
    /// Number of streamlines connecting the two regions.
    #[default]
    StreamlineCount,

    /// Sum of `1 / length` over the connecting streamlines, in mm⁻¹.
    ///
    /// Compensates the length dependence of streamline count described in the
    /// module documentation.
    InverseLength,

    /// Streamline count divided by the summed volume of the two regions, in
    /// mm⁻³.
    ///
    /// Compensates the region-size dependence. Requires the parcellation's
    /// region volumes, which the builder takes from the parcellation itself.
    InverseNodeVolume,

    /// Mean length of the connecting streamlines, in mm.
    ///
    /// A description of the pathway rather than a count of it: an edge of weight
    /// 60 means the streamlines joining those regions averaged 60 mm, whatever
    /// their number. Useful paired with a count matrix, not as a substitute.
    MeanLength,
}

impl EdgeWeighting {
    /// Whether the weight is an average rather than a sum.
    ///
    /// An average needs its accumulator divided by the contributing count at the
    /// end; a sum does not.
    const fn is_mean(self) -> bool {
        matches!(self, Self::MeanLength)
    }
}

/// Assignment and weighting policy for building a connectome.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ConnectomeConfig {
    /// How endpoints are attributed to regions.
    pub assignment: EndpointAssignment,
    /// What the edge weights count.
    pub weighting: EdgeWeighting,
}

impl ConnectomeConfig {
    /// Terminal assignment with streamline-count weighting — the exact,
    /// unnormalised connectome.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            assignment: EndpointAssignment::Terminal,
            weighting: EdgeWeighting::StreamlineCount,
        }
    }

    /// Replace the endpoint assignment.
    #[must_use]
    pub const fn with_assignment(self, assignment: EndpointAssignment) -> Self {
        Self { assignment, ..self }
    }

    /// Replace the edge weighting.
    #[must_use]
    pub const fn with_weighting(self, weighting: EdgeWeighting) -> Self {
        Self { weighting, ..self }
    }
}

/// Build a connectivity matrix from streamlines and a parcellation.
///
/// Every streamline's two endpoints are attributed to regions under
/// `config.assignment`; a streamline joining two different regions contributes
/// to that edge under `config.weighting`. Streamlines that stay inside one
/// region, or whose endpoints cannot be attributed, are counted in the returned
/// matrix's [`StreamlineAccounting`] rather than causing an error — a tractogram
/// always contains some of both, and aborting on the first would make the
/// function unusable on real data.
///
/// # Errors
///
/// [`ConnectomeError::Parcellation`] when the assignment radius is not a usable
/// distance, and [`ConnectomeError::AllocationFailed`] when the matrix does not
/// fit in memory.
pub fn build_connectivity_matrix(
    parcellation: &Parcellation,
    streamlines: &[Polyline<f64>],
    config: &ConnectomeConfig,
) -> Result<ConnectivityMatrix, ConnectomeError> {
    let labels = parcellation.region_label_slice();
    let n = labels.len();

    let mut weights = Vec::new();
    weights
        .try_reserve_exact(n * n)
        .map_err(|_| ConnectomeError::AllocationFailed { regions: n })?;
    weights.resize(n * n, 0.0);
    // Contributions per edge, needed only to finish an averaging weighting.
    let mut contributions = if config.weighting.is_mean() {
        vec![0_u32; n * n]
    } else {
        Vec::new()
    };

    let assign = Assigner::new(parcellation, config.assignment)?;
    let volumes = node_volumes(parcellation, config.weighting);
    let mut accounting = StreamlineAccounting::default();

    for streamline in streamlines {
        let points = streamline.points();
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            accounting.degenerate += 1;
            continue;
        };
        if points.len() < 2 {
            accounting.degenerate += 1;
            continue;
        }
        accounting.total += 1;

        let start = Point::new([first.x, first.y, first.z]);
        let end = Point::new([last.x, last.y, last.z]);
        let (Some(source), Some(target)) = (
            assign.region(parcellation, &start),
            assign.region(parcellation, &end),
        ) else {
            accounting.unassigned += 1;
            continue;
        };

        // Both labels came from the parcellation's own label set, so the search
        // cannot miss; a lookup failure would mean the label set and the volume
        // disagree, which construction rules out.
        let (Ok(i), Ok(j)) = (labels.binary_search(&source), labels.binary_search(&target)) else {
            accounting.unassigned += 1;
            continue;
        };

        if i == j {
            accounting.intra_region += 1;
            // The self-connection is still recorded: intra-region streamlines
            // are real, and a caller comparing within- to between-region
            // connectivity needs them. Every graph measure excludes the
            // diagonal, so recording it changes no other number.
            let contribution =
                contribution_of(streamline, config.weighting, volumes_of(&volumes, i, j));
            weights[i * n + i] += contribution;
            if config.weighting.is_mean() {
                contributions[i * n + i] += 1;
            }
            continue;
        }

        accounting.assigned += 1;
        let contribution =
            contribution_of(streamline, config.weighting, volumes_of(&volumes, i, j));
        weights[i * n + j] += contribution;
        weights[j * n + i] += contribution;
        if config.weighting.is_mean() {
            contributions[i * n + j] += 1;
            contributions[j * n + i] += 1;
        }
    }

    if config.weighting.is_mean() {
        for (weight, count) in weights.iter_mut().zip(&contributions) {
            if *count > 0 {
                *weight /= f64::from(*count);
            }
        }
    }

    Ok(ConnectivityMatrix::from_parts(
        labels.to_vec().into_boxed_slice(),
        weights.into_boxed_slice(),
        accounting,
        config.weighting,
    ))
}

/// Resolves an endpoint to a region label under the configured policy.
enum Assigner {
    Terminal,
    Radial(NearestLabelSearch),
}

impl Assigner {
    fn new(
        parcellation: &Parcellation,
        assignment: EndpointAssignment,
    ) -> Result<Self, ConnectomeError> {
        Ok(match assignment {
            EndpointAssignment::Terminal => Self::Terminal,
            EndpointAssignment::RadialSearch { radius_mm } => {
                Self::Radial(NearestLabelSearch::new(parcellation.grid(), radius_mm)?)
            }
        })
    }

    /// The region a point belongs to, or `None` when none can be attributed.
    ///
    /// Background is not a region, so a point on an unlabelled voxel is
    /// unattributed under terminal assignment rather than assigned to label 0.
    fn region(&self, parcellation: &Parcellation, point: &Point<3>) -> Option<u32> {
        match self {
            Self::Terminal => parcellation
                .label_at(point)
                .filter(|label| *label != ritk_parcellation::BACKGROUND),
            Self::Radial(search) => search.find(parcellation, point).map(|found| found.label),
        }
    }
}

/// Region volumes indexed by matrix position, empty when the weighting does not
/// need them.
///
/// Computed once rather than per streamline: a whole-brain tractogram has
/// millions of streamlines and the volumes are a property of the parcellation.
fn node_volumes(parcellation: &Parcellation, weighting: EdgeWeighting) -> Vec<f64> {
    if weighting != EdgeWeighting::InverseNodeVolume {
        return Vec::new();
    }
    let labels = parcellation.region_label_slice();
    let mut volumes = vec![0.0; labels.len()];
    for statistics in parcellation.region_statistics() {
        if let Ok(index) = labels.binary_search(&statistics.label()) {
            volumes[index] = statistics.volume();
        }
    }
    volumes
}

/// Summed volume of the two nodes, or `None` when the weighting does not use it.
fn volumes_of(volumes: &[f64], i: usize, j: usize) -> Option<f64> {
    if volumes.is_empty() {
        return None;
    }
    // A self-connection has one node, not two, so its normaliser is that node's
    // volume rather than twice it.
    let total = if i == j {
        volumes[i]
    } else {
        volumes[i] + volumes[j]
    };
    (total > 0.0).then_some(total)
}

/// What one streamline adds to its edge under the configured weighting.
fn contribution_of(
    streamline: &Polyline<f64>,
    weighting: EdgeWeighting,
    node_volume: Option<f64>,
) -> f64 {
    match weighting {
        EdgeWeighting::StreamlineCount => 1.0,
        EdgeWeighting::MeanLength => streamline.arc_length(),
        EdgeWeighting::InverseLength => {
            let length = streamline.arc_length();
            // A streamline of zero length has no pathway to normalise against.
            // It cannot arise from an inter-region pair — two distinct regions
            // are at least one voxel apart — so this only guards the diagonal.
            if length > 0.0 { 1.0 / length } else { 0.0 }
        }
        // A region with no volume cannot occur: every label in the set has at
        // least the voxel that put it there. The guard keeps the function total.
        EdgeWeighting::InverseNodeVolume => node_volume.map_or(0.0, |volume| 1.0 / volume),
    }
}

#[cfg(test)]
mod tests;

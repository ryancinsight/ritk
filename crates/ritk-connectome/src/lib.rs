//! Connectome construction and graph analysis.
//!
//! A *connectome* is the graph whose nodes are anatomical regions and whose
//! edges summarise the streamlines running between them. This crate turns a
//! tractogram plus a [`ritk_parcellation::Parcellation`] into that graph, and
//! then measures it.
//!
//! ```text
//! Parcellation (labels) ──┐
//!                         ├──► ConnectivityMatrix ──► graph measures
//! Streamlines (polylines) ┘
//! ```
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`build`] | Turning endpoints into edges: assignment and edge weighting |
//! | [`measures`] | Graph measures over the resulting matrix |
//!
//! # What a connectome edge does and does not mean
//!
//! An edge weight is a property of the *tractogram*, not of the brain.
//! Streamline counts are known not to be proportional to axonal density:
//! tracking systematically favours short, straight, high-anisotropy paths and
//! systematically loses long, curved, or crossing ones. Two of the weightings
//! in [`EdgeWeighting`] exist to remove the leading geometric parts of that
//! dependence — pathway length and region size — and none of them turns a count
//! into a measurement of connection strength. Comparison across subjects
//! processed identically is the defensible use; absolute interpretation of a
//! single weight is not.
//!
//! # Example
//!
//! ```
//! use gaia::Polyline;
//! use leto::geometry::Point3;
//! use ritk_connectome::{ConnectomeConfig, build_connectivity_matrix};
//! use ritk_parcellation::{Parcellation, ParcellationGrid};
//!
//! let grid = ParcellationGrid::axis_aligned([4, 1, 1], [1.0; 3], [0.0; 3])?;
//! let parcellation = Parcellation::new(Box::new([1, 0, 0, 2]), grid, Vec::new())?;
//!
//! // One streamline running from the region-1 voxel to the region-2 voxel.
//! let streamline = Polyline::new(vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(3.0, 0.0, 0.0),
//! ])?;
//!
//! let matrix = build_connectivity_matrix(
//!     &parcellation,
//!     std::slice::from_ref(&streamline),
//!     &ConnectomeConfig::default(),
//! )?;
//!
//! assert_eq!(matrix.weight(1, 2), Some(1.0));
//! assert_eq!(matrix.edge_count(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod build;
pub mod measures;

pub use build::{ConnectomeConfig, EdgeWeighting, EndpointAssignment, build_connectivity_matrix};
pub use measures::{Communities, GraphMeasures};

use ritk_parcellation::ParcellationError;
use serde::{Deserialize, Serialize};

/// Failure while constructing or querying a connectome.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConnectomeError {
    /// The parcellation could not be prepared for lookup.
    #[error("parcellation error: {0}")]
    Parcellation(#[from] ParcellationError),
    /// The matrix could not be allocated.
    #[error("a connectome over {regions} regions could not be allocated")]
    AllocationFailed {
        /// Region count that was attempted.
        regions: usize,
    },
    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// A weighted, undirected edge between two regions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConnectivityEdge {
    /// Source region label.
    pub source: u32,
    /// Target region label.
    pub target: u32,
    /// Edge weight, in the units of the [`EdgeWeighting`] that built it.
    pub weight: f64,
}

/// How many streamlines reached, missed, or stayed inside a region.
///
/// Kept alongside the matrix because a connectome is not interpretable without
/// it: a matrix built from a tractogram of which four fifths were discarded is a
/// different claim from one built from a tractogram of which a twentieth were,
/// and nothing in the weights themselves distinguishes the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StreamlineAccounting {
    /// Streamlines considered — those with at least two points.
    pub total: usize,
    /// Streamlines that produced an inter-region edge.
    pub assigned: usize,
    /// Streamlines whose endpoints resolved to the same region.
    pub intra_region: usize,
    /// Streamlines with at least one endpoint no region could be found for.
    pub unassigned: usize,
    /// Streamlines rejected for having fewer than two points.
    pub degenerate: usize,
}

impl StreamlineAccounting {
    /// Fraction of considered streamlines that produced an inter-region edge,
    /// in `[0, 1]`.
    ///
    /// Returns zero for an empty tractogram.
    #[must_use]
    pub fn assigned_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "streamline counts stay far below f64's exact-integer range"
        )]
        let ratio = self.assigned as f64 / self.total as f64;
        ratio
    }
}

/// Weighted, undirected adjacency between parcellation regions.
///
/// # Storage
///
/// The matrix is stored dense and *fully symmetric*: both `(i, j)` and `(j, i)`
/// carry the weight. A triangular layout would halve the memory — for a
/// whole-brain atlas a few hundred kilobytes either way — at the cost of an
/// index-ordering branch inside every measure that walks a row, and the graph
/// algorithms in [`measures`] walk rows constantly. Symmetry is established at
/// construction instead, so every read is a plain lookup.
///
/// Self-connections sit on the diagonal. They are recorded, because a
/// tractogram's intra-region streamlines are real, but excluded from degree,
/// density, and every path-based measure, where a self-loop is not a connection
/// between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityMatrix {
    /// Region labels, sorted; matrix index `i` is region `labels[i]`.
    labels: Box<[u32]>,
    /// Dense symmetric `n × n` weights, row-major.
    weights: Box<[f64]>,
    accounting: StreamlineAccounting,
    weighting: EdgeWeighting,
}

impl ConnectivityMatrix {
    /// Assemble from sorted labels and a dense symmetric weight matrix.
    pub(crate) fn from_parts(
        labels: Box<[u32]>,
        weights: Box<[f64]>,
        accounting: StreamlineAccounting,
        weighting: EdgeWeighting,
    ) -> Self {
        debug_assert!(
            labels.is_sorted(),
            "invariant: labels index the matrix by binary search"
        );
        debug_assert_eq!(
            weights.len(),
            labels.len() * labels.len(),
            "invariant: the weight matrix covers every region pair"
        );
        Self {
            labels,
            weights,
            accounting,
            weighting,
        }
    }

    /// Number of regions — the node count.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.labels.len()
    }

    /// Region labels in matrix-index order.
    #[must_use]
    pub const fn region_labels(&self) -> &[u32] {
        &self.labels
    }

    /// Matrix index of a region label.
    #[must_use]
    pub fn index_of(&self, label: u32) -> Option<usize> {
        self.labels.binary_search(&label).ok()
    }

    /// How the streamlines were accounted for.
    #[must_use]
    pub const fn accounting(&self) -> StreamlineAccounting {
        self.accounting
    }

    /// The weighting the edge values are expressed in.
    #[must_use]
    pub const fn weighting(&self) -> EdgeWeighting {
        self.weighting
    }

    /// Weight between two region labels, or `None` when either is absent.
    #[must_use]
    pub fn weight(&self, source: u32, target: u32) -> Option<f64> {
        let i = self.index_of(source)?;
        let j = self.index_of(target)?;
        Some(self.weight_at(i, j))
    }

    /// Weight between two matrix indices.
    ///
    /// # Panics
    ///
    /// If either index is outside the matrix.
    #[must_use]
    pub fn weight_at(&self, i: usize, j: usize) -> f64 {
        self.weights[i * self.region_count() + j]
    }

    /// One row of the matrix — every weight incident on region index `i`.
    ///
    /// # Panics
    ///
    /// If the index is outside the matrix.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        let n = self.region_count();
        &self.weights[i * n..(i + 1) * n]
    }

    /// Every edge with a nonzero weight, self-connections included.
    pub fn edges(&self) -> impl Iterator<Item = ConnectivityEdge> + '_ {
        let n = self.region_count();
        (0..n)
            .flat_map(move |i| (i..n).map(move |j| (i, j)))
            .filter_map(move |(i, j)| {
                let weight = self.weight_at(i, j);
                (weight > 0.0).then_some(ConnectivityEdge {
                    source: self.labels[i],
                    target: self.labels[j],
                    weight,
                })
            })
    }

    /// Number of distinct region pairs connected, excluding self-connections.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let n = self.region_count();
        (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .filter(|(i, j)| self.weight_at(*i, *j) > 0.0)
            .count()
    }

    /// Number of distinct neighbours of a region, excluding itself.
    #[must_use]
    pub fn degree(&self, label: u32) -> Option<usize> {
        self.index_of(label).map(|i| self.degree_at(i))
    }

    /// Number of distinct neighbours of a region index, excluding itself.
    #[must_use]
    pub fn degree_at(&self, i: usize) -> usize {
        self.row(i)
            .iter()
            .enumerate()
            .filter(|(j, weight)| *j != i && **weight > 0.0)
            .count()
    }

    /// Sum of the weights incident on a region, excluding its self-connection.
    #[must_use]
    pub fn strength(&self, label: u32) -> Option<f64> {
        self.index_of(label).map(|i| self.strength_at(i))
    }

    /// Sum of the weights incident on a region index, excluding its
    /// self-connection.
    #[must_use]
    pub fn strength_at(&self, i: usize) -> f64 {
        self.row(i)
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, weight)| *weight)
            .sum()
    }

    /// Fraction of possible node pairs that are connected, in `[0, 1]`.
    ///
    /// Zero for a graph with fewer than two nodes, which has no pairs.
    #[must_use]
    pub fn density(&self) -> f64 {
        let n = self.region_count();
        if n <= 1 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "region counts stay far below f64's exact-integer range"
        )]
        let ratio = self.edge_count() as f64 / (n * (n - 1) / 2) as f64;
        ratio
    }

    /// Every graph measure this crate computes — see [`measures`].
    #[must_use]
    pub fn measures(&self) -> GraphMeasures {
        measures::compute(self)
    }

    /// Serialise to a JSON string.
    ///
    /// # Errors
    ///
    /// [`ConnectomeError::Json`] on serialisation failure.
    pub fn to_json(&self) -> Result<String, ConnectomeError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserialise from a JSON string.
    ///
    /// # Errors
    ///
    /// [`ConnectomeError::Json`] on deserialisation failure.
    pub fn from_json(json: &str) -> Result<Self, ConnectomeError> {
        Ok(serde_json::from_str(json)?)
    }
}

#[cfg(test)]
mod tests;

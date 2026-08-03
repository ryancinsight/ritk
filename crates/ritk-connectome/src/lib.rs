//! Parcellation-to-graph construction and graph measures.
//!
//! `ritk-connectome` is the third leaf crate from the
//! [ADR 0036](../../../docs/adr/0036-neuroimaging-and-mr-ownership.md)
//! diffusion-MRI pipeline.  It reduces a set of streamline endpoints and a
//! volumetric parcellation into a weighted undirected
//! [`ConnectivityMatrix`] — the adjacency of anatomical regions — plus
//! per-node and global graph measures.
//!
//! # Data flow
//!
//! ```text
//! Parcellation (label volume) ──┐
//!                               ├──► ConnectivityMatrix ──► graph measures
//! Streamlines (Gaia polylines) ─┘
//! ```
//!
//! Each streamline contributes one unit of weight to the edge connecting
//! the two parcellation regions its endpoints fall in.  Streamlines whose
//! endpoints land in the same region, or outside the parcellation volume,
//! are counted but do not add an inter-region edge.
//!
//! # Persistence
//!
//! Connectivity matrices persist through Consus formats per ADR 0036
//! decision 2.  The current increment provides Serde-based JSON
//! serialisation as a lightweight codec; Consus HDF5 integration is the
//! natural next step.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use gaia::Polyline;
use ritk_spatial::Point;
use serde::{Deserialize, Serialize};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Failure while constructing or querying a connectivity matrix.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConnectomeError {
    /// The parcellation has no non-background regions.
    #[error("parcellation has no labelled regions (only background label {0})")]
    EmptyParcellation(u32),
    /// A region label referenced by an edge does not exist in the parcellation.
    #[error("region label {label} not found in parcellation")]
    UnknownRegion {
        /// The label that was not found.
        label: u32,
    },
    /// The connectivity matrix size does not match the region count.
    #[error("connectivity matrix expects {expected} regions, got {actual}")]
    RegionCountMismatch {
        /// Expected number of regions.
        expected: usize,
        /// Actual number of regions.
        actual: usize,
    },
    /// A streamline endpoint falls outside the parcellation volume.
    #[error(
        "streamline {streamline_index} {endpoint} endpoint at \
         [{x:.2}, {y:.2}, {z:.2}] is outside parcellation bounds"
    )]
    EndpointOutOfBounds {
        /// Index of the streamline in the input slice.
        streamline_index: usize,
        /// Which endpoint ("start" or "end").
        endpoint: &'static str,
        /// Physical x coordinate.
        x: f64,
        /// Physical y coordinate.
        y: f64,
        /// Physical z coordinate.
        z: f64,
    },
    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── Parcellation ──────────────────────────────────────────────────────────────

/// A 3-D parcellation volume mapping each voxel to a region label.
///
/// Labels are stored in z-major (slice-first) order.  The special label `0`
/// conventionally represents background / outside the brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parcellation {
    /// Flat label array in z-major order: `[z][y][x]`.
    labels: Box<[u32]>,
    /// Grid dimensions `[nx, ny, nz]`.
    shape: [usize; 3],
    /// Voxel size in physical units (mm), `[sx, sy, sz]`.
    spacing: [f64; 3],
    /// Physical position of the first voxel centre `[ox, oy, oz]`.
    origin: [f64; 3],
    /// Human-readable names keyed by label ID.  The reserved background
    /// label `0` may appear here (e.g. `"Background"`) or be omitted.
    region_names: Vec<(u32, String)>,
}

impl Parcellation {
    /// Construct a parcellation from a flat label array.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectomeError::EmptyParcellation`] when every voxel is
    /// background (label 0), [`ConnectomeError::RegionCountMismatch`] when
    /// the label count does not match the shape, and
    /// [`ConnectomeError::RegionCountMismatch`] for zero-dimension grids.
    pub fn new(
        labels: Box<[u32]>,
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        region_names: Vec<(u32, String)>,
    ) -> Result<Self, ConnectomeError> {
        let [nx, ny, nz] = shape;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(ConnectomeError::RegionCountMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let expected = nx * ny * nz;
        if labels.len() != expected {
            return Err(ConnectomeError::RegionCountMismatch {
                expected,
                actual: labels.len(),
            });
        }
        let has_nonzero = labels.iter().any(|&label| label != 0);
        if !has_nonzero {
            return Err(ConnectomeError::EmptyParcellation(0));
        }
        Ok(Self {
            labels,
            shape,
            spacing,
            origin,
            region_names,
        })
    }

    /// Grid dimensions `[nx, ny, nz]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Voxel spacing `[sx, sy, sz]` in mm.
    #[must_use]
    pub const fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Origin of the first voxel centre `[ox, oy, oz]`.
    #[must_use]
    pub const fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Sorted list of unique non-background region labels.
    #[must_use]
    pub fn region_labels(&self) -> Vec<u32> {
        let mut labels: Vec<u32> = self
            .labels
            .iter()
            .copied()
            .filter(|&l| l != 0)
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    /// Number of unique non-background regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.region_labels().len()
    }

    /// Human-readable region names.
    #[must_use]
    pub fn region_names(&self) -> &[(u32, String)] {
        &self.region_names
    }

    /// Look up the region label at a physical point.
    ///
    /// Returns `None` when the point maps to a voxel index outside the grid
    /// or when any coordinate is non-finite.
    pub fn label_at(&self, point: &Point<3>) -> Option<u32> {
        let [px, py, pz] = point.to_array();
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            return None;
        }
        let [ox, oy, oz] = self.origin;
        let [sx, sy, sz] = self.spacing;
        let ix = ((px - ox) / sx).round() as isize;
        let iy = ((py - oy) / sy).round() as isize;
        let iz = ((pz - oz) / sz).round() as isize;
        let [nx, ny, nz] = self.shape;
        if ix < 0 || ix >= nx as isize || iy < 0 || iy >= ny as isize || iz < 0 || iz >= nz as isize
        {
            return None;
        }
        let idx = iz as usize * ny * nx + iy as usize * nx + ix as usize;
        Some(self.labels[idx])
    }
}

// ── Connectivity Matrix ───────────────────────────────────────────────────────

/// A weighted, undirected edge between two parcellation regions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConnectivityEdge {
    /// Source region label.
    pub source: u32,
    /// Target region label.
    pub target: u32,
    /// Connection weight (streamline count).
    pub weight: f64,
}

/// Weighted undirected adjacency between parcellation regions.
///
/// Weights are stored as a flat `n × n` row-major matrix accessed with the
/// upper-triangular convention: edge `(i, j)` with `i <= j` is at
/// `i * region_count + j`.  Entries below the diagonal are never written
/// and are always zero.  Self-edges (`i == j`) are stored but do not
/// contribute to degree or density.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityMatrix {
    /// Number of regions (not counting background).
    region_count: usize,
    /// Lookup from region label to internal index `[0, region_count)`.
    label_to_index: Box<[(u32, usize)]>,
    /// Flat upper-triangular weight matrix.
    weights: Box<[f64]>,
    /// Total streamlines that contributed to this matrix.
    total_streamlines: usize,
    /// Streamlines whose endpoints fell in the same region.
    intra_region_count: usize,
    /// Streamlines that were skipped (one or both endpoints out of bounds).
    skipped_count: usize,
}

impl ConnectivityMatrix {
    /// Internal index for a region label.
    fn index_of(&self, label: u32) -> Option<usize> {
        self.label_to_index
            .binary_search_by_key(&label, |(l, _)| *l)
            .ok()
            .map(|pos| self.label_to_index[pos].1)
    }

    /// Number of regions in the matrix.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.region_count
    }

    /// List of region labels in internal index order.
    #[must_use]
    pub fn region_labels(&self) -> Vec<u32> {
        let mut labels: Vec<u32> = self.label_to_index.iter().map(|(l, _)| *l).collect();
        labels.sort_unstable();
        labels
    }

    /// Total streamlines that contributed to this matrix.
    #[must_use]
    pub const fn total_streamlines(&self) -> usize {
        self.total_streamlines
    }

    /// Number of streamlines whose endpoints fell in the same region.
    #[must_use]
    pub const fn intra_region_count(&self) -> usize {
        self.intra_region_count
    }

    /// Number of streamlines skipped due to out-of-bounds endpoints.
    #[must_use]
    pub const fn skipped_count(&self) -> usize {
        self.skipped_count
    }

    /// Get the weight for an edge between two region labels.
    ///
    /// Returns `None` when either label is not in the matrix.
    pub fn weight(&self, source: u32, target: u32) -> Option<f64> {
        let i = self.index_of(source)?;
        let j = self.index_of(target)?;
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        Some(self.weights[a * self.region_count + b])
    }

    /// Iterate over all edges with nonzero weight.
    pub fn edges(&self) -> impl Iterator<Item = ConnectivityEdge> + '_ {
        let labels: Vec<u32> = self.label_to_index.iter().map(|(l, _)| *l).collect();
        (0..self.region_count)
            .flat_map(move |i| (i..self.region_count).map(move |j| (i, j)))
            .filter_map(move |(i, j)| {
                let w = self.weights[i * self.region_count + j];
                if w > 0.0 {
                    Some(ConnectivityEdge {
                        source: labels[i],
                        target: labels[j],
                        weight: w,
                    })
                } else {
                    None
                }
            })
    }

    /// Number of edges with nonzero weight (excluding self-edges).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges().filter(|e| e.source != e.target).count()
    }

    /// Binary degree of a region — the number of distinct neighbours.
    ///
    /// Returns `None` when the label is not in the matrix.
    pub fn degree(&self, label: u32) -> Option<usize> {
        let i = self.index_of(label)?;
        let start = i * self.region_count;
        let row = &self.weights[start..start + self.region_count];
        // Column contributions from rows above.
        let col_count = (0..i)
            .filter(|&r| self.weights[r * self.region_count + i] > 0.0)
            .count();
        let row_count = row.iter().filter(|&&w| w > 0.0).count();
        let self_edge = if self.weights[i * self.region_count + i] > 0.0 {
            1
        } else {
            0
        };
        Some(row_count + col_count - self_edge)
    }

    /// Weighted degree (strength) of a region — sum of incident edge weights.
    ///
    /// Returns `None` when the label is not in the matrix.
    pub fn strength(&self, label: u32) -> Option<f64> {
        let i = self.index_of(label)?;
        let start = i * self.region_count;
        let row_sum: f64 = self.weights[start..start + self.region_count]
            .iter()
            .sum();
        let col_sum: f64 = (0..i)
            .map(|r| self.weights[r * self.region_count + i])
            .sum();
        // Self-edge counted twice (once in row, once in col) — subtract once.
        let self_w = self.weights[i * self.region_count + i];
        Some(row_sum + col_sum - self_w)
    }

    /// Graph density: ratio of actual edges to possible edges.
    ///
    /// For an undirected graph with `n` nodes, the maximum number of edges
    /// is `n·(n-1)/2`.  Density ∈ [0, 1] for n > 1; returns 0.0 when n ≤ 1.
    #[must_use]
    pub fn density(&self) -> f64 {
        let n = self.region_count;
        if n <= 1 {
            return 0.0;
        }
        let max_edges = (n * (n - 1)) / 2;
        self.edge_count() as f64 / max_edges as f64
    }

    /// Serialise to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectomeError::Json`] on serialisation failure.
    pub fn to_json(&self) -> Result<String, ConnectomeError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserialise from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectomeError::Json`] on deserialisation failure.
    pub fn from_json(json: &str) -> Result<Self, ConnectomeError> {
        Ok(serde_json::from_str(json)?)
    }
}

// ── Construction ──────────────────────────────────────────────────────────────

/// Build a connectivity matrix from streamlines and a parcellation.
///
/// For each streamline the endpoints are looked up in the parcellation and
/// the corresponding inter-region edge weight is incremented by one.
/// Streamlines whose endpoints fall outside the parcellation volume are
/// counted as skipped but do not cause an error.  Endpoints that land in
/// the same region increment the intra-region counter.
///
/// # Errors
///
/// Returns [`ConnectomeError::EmptyParcellation`] when the parcellation has
/// no labelled regions.
pub fn build_connectivity_matrix(
    parcellation: &Parcellation,
    streamlines: &[Polyline<f64>],
) -> Result<ConnectivityMatrix, ConnectomeError> {
    let region_labels = parcellation.region_labels();
    if region_labels.is_empty() {
        return Err(ConnectomeError::EmptyParcellation(0));
    }

    let n = region_labels.len();
    let label_to_index: Box<[(u32, usize)]> = region_labels
        .iter()
        .enumerate()
        .map(|(idx, &label)| (label, idx))
        .collect();
    let total = n * n;
    let mut weights = Vec::with_capacity(total);
    weights.try_reserve_exact(total).map_err(|_| {
        ConnectomeError::RegionCountMismatch {
            expected: total,
            actual: 0,
        }
    })?;
    weights.resize(total, 0.0);

    let mut total_streamlines = 0usize;
    let mut intra_region_count = 0usize;
    let mut skipped_count = 0usize;

    for streamline in streamlines.iter() {
        let points = streamline.points();
        if points.len() < 2 {
            continue;
        }
        total_streamlines += 1;

        let first_phys = Point::new([points[0].x, points[0].y, points[0].z]);
        let last_phys = Point::new([
            points[points.len() - 1].x,
            points[points.len() - 1].y,
            points[points.len() - 1].z,
        ]);

        let label_start = parcellation.label_at(&first_phys);
        let label_end = parcellation.label_at(&last_phys);

        let (label_a, label_b) = match (label_start, label_end) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                skipped_count += 1;
                continue;
            }
        };

        // Background labels contribute to skipped, not to edges.
        if label_a == 0 || label_b == 0 {
            skipped_count += 1;
            continue;
        }

        if label_a == label_b {
            intra_region_count += 1;
            // Still record the self-edge.
            if let Some(i) = label_index(&label_to_index, label_a) {
                weights[i * n + i] += 1.0;
            }
            continue;
        }

        let Some(i) = label_index(&label_to_index, label_a) else {
            continue;
        };
        let Some(j) = label_index(&label_to_index, label_b) else {
            continue;
        };

        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        weights[a * n + b] += 1.0;
    }

    Ok(ConnectivityMatrix {
        region_count: n,
        label_to_index,
        weights: weights.into_boxed_slice(),
        total_streamlines,
        intra_region_count,
        skipped_count,
    })
}



/// Binary search the internal index for a given label.
fn label_index(lookup: &[(u32, usize)], label: u32) -> Option<usize> {
    lookup
        .binary_search_by_key(&label, |(l, _)| *l)
        .ok()
        .map(|pos| lookup[pos].1)
}

#[cfg(test)]
mod tests;

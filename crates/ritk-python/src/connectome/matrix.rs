//! Python representation of a connectome and its graph measures.

use gaia::Polyline;
use leto::geometry::Point3;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ritk_connectome::measures::rich_club::{normalised_rich_club, RandomisationConfig};
use ritk_connectome::{
    ConnectivityMatrix, ConnectomeConfig, EdgeWeighting, EndpointAssignment, GraphMeasures,
};

use crate::connectome::parcellation::PyParcellation;
use crate::errors::{RitkPyError, RitkResult};

/// Build a region connectome from streamlines and a parcellation.
///
/// Delegates to `ritk_connectome::build_connectivity_matrix`, so the result is
/// identical to `ritk tract connectome` on the same inputs.
///
/// The GIL is released for the construction, which is the expensive part: a
/// whole-brain tractogram is millions of endpoint lookups.
///
/// Args:
///     parcellation: The regions endpoints are attributed to.
///     streamlines: Sequence of `[N, 3]` float arrays, each a polyline in the
///         parcellation's physical frame. Both inputs must express position the
///         same way; nothing in either says whether they do, and a label volume
///         from a different session produces a full, plausible, entirely wrong
///         connectome.
///     assignment_radius: Endpoint search radius in mm. Zero assigns an endpoint
///         only to the region it lands in, which discards every streamline
///         terminating in white matter — ordinarily most of a tractogram, since
///         tracking stops at the grey/white boundary while a cortical
///         parcellation labels only grey matter. Defaults to 2.0.
///     weighting: One of "count", "inverse_length", "inverse_node_volume", or
///         "mean_length". Defaults to "count".
///
/// Returns:
///     ConnectivityMatrix
///
/// Raises:
///     ValueError: if a streamline array is not `[N, 3]` with at least two
///         points, the radius is not a usable distance, or the weighting is not
///         recognised.
#[pyfunction]
#[pyo3(signature = (parcellation, streamlines, assignment_radius=None, weighting=None))]
pub fn build_connectivity_matrix(
    py: Python<'_>,
    parcellation: PyRef<'_, PyParcellation>,
    streamlines: Vec<PyReadonlyArray2<'_, f64>>,
    assignment_radius: Option<f64>,
    weighting: Option<&str>,
) -> RitkResult<PyConnectivityMatrix> {
    let radius = assignment_radius.unwrap_or(2.0);
    if !radius.is_finite() || radius < 0.0 {
        return Err(RitkPyError::value(format!(
            "assignment_radius must be finite and nonnegative, got {radius}"
        )));
    }

    let config = ConnectomeConfig::new()
        .with_assignment(if radius > 0.0 {
            EndpointAssignment::RadialSearch { radius_mm: radius }
        } else {
            EndpointAssignment::Terminal
        })
        .with_weighting(parse_weighting(weighting.unwrap_or("count"))?);

    let polylines = streamlines
        .iter()
        .enumerate()
        .map(|(index, array)| to_polyline(index, array))
        .collect::<RitkResult<Vec<_>>>()?;

    // The reference is taken before the GIL is released: a `PyRef` is not
    // shareable across threads, but the plain `&Parcellation` behind it is.
    let regions = parcellation.inner();
    let inner = py
        .allow_threads(|| ritk_connectome::build_connectivity_matrix(regions, &polylines, &config))
        .map_err(RitkPyError::value)?;

    Ok(PyConnectivityMatrix { inner })
}

fn parse_weighting(name: &str) -> RitkResult<EdgeWeighting> {
    match name {
        "count" => Ok(EdgeWeighting::StreamlineCount),
        "inverse_length" => Ok(EdgeWeighting::InverseLength),
        "inverse_node_volume" => Ok(EdgeWeighting::InverseNodeVolume),
        "mean_length" => Ok(EdgeWeighting::MeanLength),
        other => Err(RitkPyError::value(format!(
            "unknown weighting {other:?}: expected one of \"count\", \"inverse_length\", \
             \"inverse_node_volume\", \"mean_length\""
        ))),
    }
}

fn to_polyline(index: usize, array: &PyReadonlyArray2<'_, f64>) -> RitkResult<Polyline<f64>> {
    let view = array.as_array();
    let [rows, columns] = [view.shape()[0], view.shape()[1]];
    if columns != 3 {
        return Err(RitkPyError::value(format!(
            "streamline {index} must be [N, 3], got [{rows}, {columns}]"
        )));
    }
    let points: Vec<Point3<f64>> = view
        .rows()
        .into_iter()
        .map(|row| Point3::new(row[0], row[1], row[2]))
        .collect();
    Polyline::new(points)
        .map_err(|error| RitkPyError::value(format!("streamline {index}: {error}")))
}

/// Weighted, undirected adjacency between parcellation regions.
#[pyclass(name = "ConnectivityMatrix", module = "ritk.connectome")]
pub struct PyConnectivityMatrix {
    inner: ConnectivityMatrix,
}

#[pymethods]
impl PyConnectivityMatrix {
    /// Number of regions.
    fn __len__(&self) -> usize {
        self.inner.region_count()
    }

    /// Region labels, in matrix-index order.
    #[getter]
    fn region_labels(&self) -> Vec<u32> {
        self.inner.region_labels().to_vec()
    }

    /// The full symmetric weight matrix, as an `[n, n]` array.
    ///
    /// Both `(i, j)` and `(j, i)` carry the weight. Self-connections sit on the
    /// diagonal; they are recorded because intra-region streamlines are real,
    /// but every graph measure excludes them.
    fn weights<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyArray2<f64>>> {
        let n = self.inner.region_count();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            flat.extend_from_slice(self.inner.row(i));
        }
        PyArray1::<f64>::from_vec_bound(py, flat)
            .reshape([n, n])
            .map_err(|error| RitkPyError::runtime(format!("reshaping the matrix: {error}")))
    }

    /// How the streamlines were accounted for.
    ///
    /// Returns:
    ///     dict: `total`, `assigned`, `intra_region`, `unassigned`, and
    ///     `assigned_fraction`. The three counts partition `total` exactly.
    ///
    ///     Read this before reading the weights: a matrix built from a
    ///     tractogram of which four fifths were discarded is a different claim
    ///     from one built from a twentieth, and the weights do not say which.
    #[getter]
    fn accounting<'py>(&self, py: Python<'py>) -> RitkResult<Bound<'py, PyDict>> {
        let accounting = self.inner.accounting();
        let entry = PyDict::new_bound(py);
        let map = |error: PyErr| RitkPyError::runtime(format!("building the accounting: {error}"));
        entry.set_item("total", accounting.total).map_err(map)?;
        entry
            .set_item("assigned", accounting.assigned)
            .map_err(map)?;
        entry
            .set_item("intra_region", accounting.intra_region)
            .map_err(map)?;
        entry
            .set_item("unassigned", accounting.unassigned)
            .map_err(map)?;
        entry
            .set_item("assigned_fraction", accounting.assigned_fraction())
            .map_err(map)?;
        Ok(entry)
    }

    /// Number of connected region pairs, excluding self-connections.
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Fraction of possible region pairs that are connected.
    #[getter]
    fn density(&self) -> f64 {
        self.inner.density()
    }

    /// Every graph measure, computed together.
    ///
    /// They share the all-pairs shortest-path solution, which dominates the
    /// cost, so deriving them together costs what any one of them costs alone.
    /// The GIL is released for the computation.
    fn measures(&self, py: Python<'_>) -> PyGraphMeasures {
        let matrix = &self.inner;
        PyGraphMeasures {
            inner: py.allow_threads(|| matrix.measures()),
        }
    }

    /// The rich-club curve against a degree-preserving null model.
    ///
    /// Args:
    ///     ensemble_size: Randomised graphs to average over.
    ///     seed: Seed for the swap sequence, so a reported ratio is reproducible.
    ///     swaps_per_edge: Swap attempts per edge, per sample. Defaults to 10.
    ///
    /// Returns:
    ///     tuple[list[dict], float]: one entry per degree threshold, and the
    ///     fraction of attempted swaps the ensemble accepted.
    ///
    ///     A `ratio` above one is the evidence of rich-club organisation; the
    ///     raw coefficient rising is not, since it rises in any graph whose hubs
    ///     simply have more edges. Read the ratio with the acceptance fraction:
    ///     a graph too constrained to rewire yields an ensemble that never left
    ///     where it started.
    ///
    /// Raises:
    ///     ValueError: if the ensemble size or swap count is zero.
    #[pyo3(signature = (ensemble_size, seed, swaps_per_edge=None))]
    fn rich_club<'py>(
        &self,
        py: Python<'py>,
        ensemble_size: usize,
        seed: u64,
        swaps_per_edge: Option<usize>,
    ) -> RitkResult<(Vec<Bound<'py, PyDict>>, f64)> {
        let config = RandomisationConfig {
            ensemble_size,
            swaps_per_edge: swaps_per_edge.unwrap_or(10),
            seed,
        };
        let matrix = &self.inner;
        let (levels, report) = py
            .allow_threads(|| normalised_rich_club(matrix, config))
            .map_err(RitkPyError::value)?;

        let entries = levels
            .into_iter()
            .map(|level| {
                let entry = PyDict::new_bound(py);
                let map = |error: PyErr| RitkPyError::runtime(format!("building a level: {error}"));
                entry
                    .set_item("degree", level.observed.degree)
                    .map_err(map)?;
                entry
                    .set_item("node_count", level.observed.node_count)
                    .map_err(map)?;
                entry
                    .set_item("coefficient", level.observed.coefficient)
                    .map_err(map)?;
                entry
                    .set_item("mean_weight", level.observed.mean_weight)
                    .map_err(map)?;
                entry
                    .set_item("random_mean", level.random_mean)
                    .map_err(map)?;
                entry
                    .set_item("random_deviation", level.random_deviation)
                    .map_err(map)?;
                entry.set_item("ratio", level.ratio).map_err(map)?;
                Ok(entry)
            })
            .collect::<RitkResult<Vec<_>>>()?;

        Ok((entries, report.acceptance()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ConnectivityMatrix(regions={}, edges={}, density={:.4})",
            self.inner.region_count(),
            self.inner.edge_count(),
            self.inner.density()
        )
    }
}

/// Graph measures over one connectome.
#[pyclass(name = "GraphMeasures", module = "ritk.connectome")]
pub struct PyGraphMeasures {
    inner: GraphMeasures,
}

#[pymethods]
impl PyGraphMeasures {
    /// Number of nodes.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges, excluding self-connections.
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Fraction of possible node pairs that are connected.
    #[getter]
    fn density(&self) -> f64 {
        self.inner.density()
    }

    /// Neighbour count per node, in matrix-index order.
    fn degree<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let values: Vec<u64> = self.inner.degree().iter().map(|d| *d as u64).collect();
        PyArray1::from_vec_bound(py, values)
    }

    /// Summed incident weight per node.
    fn strength<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.inner.strength())
    }

    /// Binary clustering coefficient per node.
    fn clustering<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.inner.clustering())
    }

    /// Onnela weighted clustering coefficient per node.
    ///
    /// Reduces exactly to the binary form when every present edge carries the
    /// maximum weight, and separates a triangle closed by heavy edges from one
    /// closed by negligible ones — a distinction the binary form cannot make.
    fn weighted_clustering<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.inner.weighted_clustering())
    }

    /// Normalised betweenness centrality per node, in `[0, 1]`.
    ///
    /// Identifies hubs whose position matters rather than whose degree is large:
    /// a node of modest degree bridging two otherwise separate modules carries
    /// enormous traffic, and degree does not see it.
    fn betweenness<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.inner.betweenness())
    }

    /// Local efficiency per node.
    fn local_efficiency<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, self.inner.local_efficiency())
    }

    /// Mean shortest-path length over reachable node pairs, or None when no
    /// pair is reachable.
    ///
    /// Read with `reachable_pair_fraction`: a graph in fragments can show a
    /// short characteristic path precisely because the long paths do not exist.
    #[getter]
    fn characteristic_path_length(&self) -> Option<f64> {
        self.inner.characteristic_path_length()
    }

    /// Fraction of ordered node pairs with a path between them.
    #[getter]
    fn reachable_pair_fraction(&self) -> f64 {
        self.inner.reachable_pair_fraction()
    }

    /// Mean reciprocal shortest-path length.
    ///
    /// Stays defined on a disconnected graph, where an unreachable pair
    /// contributes zero — which is why it is the better-behaved summary.
    #[getter]
    fn global_efficiency(&self) -> f64 {
        self.inner.global_efficiency()
    }

    /// Connected component sizes, descending.
    ///
    /// A single entry equal to `node_count` means the graph is connected.
    /// Isolated regions appear as components of size one and are the commonest
    /// reason a connectome's path measures look better than the data warrants.
    #[getter]
    fn component_sizes(&self) -> Vec<usize> {
        self.inner.component_sizes().to_vec()
    }

    /// Community index per node, from the deterministic Louvain method.
    ///
    /// Node order is fixed rather than randomised, so the partition is a
    /// function of the matrix alone and can be compared between subjects.
    fn communities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let values: Vec<u64> = self
            .inner
            .communities()
            .assignment()
            .iter()
            .map(|c| *c as u64)
            .collect();
        PyArray1::from_vec_bound(py, values)
    }

    /// Modularity of the detected partition — the value achieved, not a bound.
    #[getter]
    fn modularity(&self) -> f64 {
        self.inner.communities().modularity()
    }

    /// Number of detected communities.
    #[getter]
    fn community_count(&self) -> usize {
        self.inner.communities().count()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphMeasures(nodes={}, edges={}, efficiency={:.3}, communities={})",
            self.inner.node_count(),
            self.inner.edge_count(),
            self.inner.global_efficiency(),
            self.inner.communities().count()
        )
    }
}

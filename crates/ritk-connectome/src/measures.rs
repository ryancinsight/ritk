//! Graph measures over a connectivity matrix.
//!
//! # The one convention everything here rests on
//!
//! A connectome edge weight is a measure of *connection strength*: a heavier
//! edge means more streamlines, so the two regions are more closely linked.
//! Every shortest-path measure needs the opposite — a *distance*, where a
//! stronger link is a shorter step. The two are related by inversion,
//!
//! ```text
//! length(i, j) = 1 / w(i, j)
//! ```
//!
//! and this crate applies that inversion at the single point where the graph is
//! converted for path finding ([`paths`]). Getting it backwards is the classic
//! error in weighted network analysis, and it does not fail loudly: it produces
//! a path length that is largest exactly where the connection is strongest, so
//! every derived measure inverts its meaning while remaining a plausible number.
//!
//! Binary measures ignore the weights and use the topology alone. Both are
//! reported, because they answer different questions and can disagree: a node
//! with many weak connections and one with few strong ones look the same
//! topologically.
//!
//! # Disconnection
//!
//! A real connectome is often not connected — an isolated region, or a
//! parcellation whose regions the tractogram never reached. That makes the
//! characteristic path length infinite, since some pair has no path. Two
//! responses are in use and both are wrong in one direction: averaging over only
//! the reachable pairs understates the cost of disconnection, and dropping the
//! measure loses it entirely. Global efficiency avoids the problem by averaging
//! the *reciprocal* distance, where an unreachable pair contributes exactly
//! zero, which is why it is the better-behaved summary and is reported
//! unconditionally. [`GraphMeasures::characteristic_path_length`] is reported
//! over reachable pairs only, with [`GraphMeasures::reachable_pair_fraction`]
//! stating how much of the graph that covered.
//!
//! # References
//!
//! * Rubinov, M. & Sporns, O. (2010). Complex network measures of brain
//!   connectivity: Uses and interpretations. *NeuroImage* 52(3):1059–1069.
//! * Watts, D. J. & Strogatz, S. H. (1998). Collective dynamics of
//!   "small-world" networks. *Nature* 393:440–442.
//! * Latora, V. & Marchiori, M. (2001). Efficient behavior of small-world
//!   networks. *Physical Review Letters* 87:198701.

pub mod centrality;
pub mod clustering;
pub mod community;
pub mod paths;
pub mod rich_club;

pub use community::Communities;
pub use rich_club::RichClubLevel;

use serde::{Deserialize, Serialize};

use crate::ConnectivityMatrix;

/// Every graph measure this crate computes, over one connectome.
///
/// Computed together rather than one at a time because they share the
/// all-pairs shortest-path solution, which dominates the cost: deriving path
/// length, efficiency, and betweenness from one Dijkstra sweep costs what any
/// one of them costs separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphMeasures {
    node_count: usize,
    edge_count: usize,
    density: f64,
    degree: Box<[usize]>,
    strength: Box<[f64]>,
    clustering: Box<[f64]>,
    weighted_clustering: Box<[f64]>,
    betweenness: Box<[f64]>,
    local_efficiency: Box<[f64]>,
    characteristic_path_length: Option<f64>,
    reachable_pair_fraction: f64,
    global_efficiency: f64,
    component_sizes: Box<[usize]>,
    communities: Communities,
    rich_club: Box<[RichClubLevel]>,
}

impl GraphMeasures {
    /// Number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of edges, excluding self-connections.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Fraction of possible node pairs that are connected.
    #[must_use]
    pub const fn density(&self) -> f64 {
        self.density
    }

    /// Number of neighbours per node, in matrix-index order.
    #[must_use]
    pub const fn degree(&self) -> &[usize] {
        &self.degree
    }

    /// Summed incident weight per node, in matrix-index order.
    #[must_use]
    pub const fn strength(&self) -> &[f64] {
        &self.strength
    }

    /// Binary clustering coefficient per node — see [`clustering`].
    #[must_use]
    pub const fn clustering(&self) -> &[f64] {
        &self.clustering
    }

    /// Onnela weighted clustering coefficient per node — see [`clustering`].
    #[must_use]
    pub const fn weighted_clustering(&self) -> &[f64] {
        &self.weighted_clustering
    }

    /// Normalised betweenness centrality per node — see [`centrality`].
    #[must_use]
    pub const fn betweenness(&self) -> &[f64] {
        &self.betweenness
    }

    /// Local efficiency per node — see [`paths`].
    #[must_use]
    pub const fn local_efficiency(&self) -> &[f64] {
        &self.local_efficiency
    }

    /// Mean shortest-path length over *reachable* node pairs.
    ///
    /// `None` when no pair is reachable at all. Read together with
    /// [`Self::reachable_pair_fraction`]: this average says nothing about the
    /// pairs it excluded, and a graph in fragments can show a short
    /// characteristic path precisely because the long paths do not exist.
    #[must_use]
    pub const fn characteristic_path_length(&self) -> Option<f64> {
        self.characteristic_path_length
    }

    /// Fraction of ordered node pairs with a path between them, in `[0, 1]`.
    #[must_use]
    pub const fn reachable_pair_fraction(&self) -> f64 {
        self.reachable_pair_fraction
    }

    /// Global efficiency — the mean reciprocal shortest-path length.
    ///
    /// Defined for a disconnected graph, where an unreachable pair contributes
    /// zero.
    #[must_use]
    pub const fn global_efficiency(&self) -> f64 {
        self.global_efficiency
    }

    /// Sizes of the connected components, descending.
    ///
    /// A single entry equal to [`Self::node_count`] means the graph is
    /// connected. Isolated regions appear as components of size one, and are the
    /// commonest reason a connectome's path measures look better than the
    /// underlying data.
    #[must_use]
    pub const fn component_sizes(&self) -> &[usize] {
        &self.component_sizes
    }

    /// Community partition found by the deterministic Louvain method — see
    /// [`community`].
    #[must_use]
    pub const fn communities(&self) -> &Communities {
        &self.communities
    }

    /// Rich-club coefficient at each degree threshold — see [`rich_club`].
    ///
    /// Unnormalised: a rising curve is not by itself evidence of a rich club.
    /// The module documentation says what normalisation the claim needs and why
    /// it is the caller's to supply.
    #[must_use]
    pub const fn rich_club(&self) -> &[RichClubLevel] {
        &self.rich_club
    }

    /// Mean of a per-node measure, or zero for an empty graph.
    #[must_use]
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "node counts stay far below f64's exact-integer range"
        )]
        let count = values.len() as f64;
        values.iter().sum::<f64>() / count
    }
}

/// Compute every measure over `matrix`.
pub(crate) fn compute(matrix: &ConnectivityMatrix) -> GraphMeasures {
    let n = matrix.region_count();
    let distances = paths::all_pairs_shortest_paths(matrix);
    let summary = paths::summarise(&distances);

    GraphMeasures {
        node_count: n,
        edge_count: matrix.edge_count(),
        density: matrix.density(),
        degree: (0..n).map(|i| matrix.degree_at(i)).collect(),
        strength: (0..n).map(|i| matrix.strength_at(i)).collect(),
        clustering: clustering::binary(matrix),
        weighted_clustering: clustering::onnela(matrix),
        betweenness: centrality::betweenness(matrix),
        local_efficiency: paths::local_efficiency(matrix),
        characteristic_path_length: summary.characteristic_path_length,
        reachable_pair_fraction: summary.reachable_pair_fraction,
        global_efficiency: summary.global_efficiency,
        component_sizes: paths::component_sizes(matrix),
        communities: community::louvain(matrix),
        rich_club: rich_club::rich_club(matrix).into_boxed_slice(),
    }
}

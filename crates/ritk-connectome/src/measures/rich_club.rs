//! Rich-club organisation — whether the hubs preferentially connect to each
//! other.
//!
//! For a degree threshold `k`, take the subgraph of nodes with degree greater
//! than `k` and ask how densely *those* nodes are connected:
//!
//! ```text
//! Φ(k) = 2·E_{>k} / (N_{>k} · (N_{>k} − 1))
//! ```
//!
//! A rich club is present when `Φ` rises with `k` faster than degree alone would
//! force. In a structural connectome the interpretation is direct: a set of
//! high-degree regions wired preferentially to one another forms a communication
//! backbone, and its edges are the ones whose loss disconnects the most.
//!
//! # Why the raw curve is not the measure
//!
//! `Φ(k)` rises with `k` in *any* graph, because high-degree nodes have more
//! edges and so are likelier to be connected to each other by chance alone. So
//! **a rising `Φ(k)` from [`rich_club`] is not by itself evidence of a rich
//! club.** The published measure is the ratio
//!
//! ```text
//! Φ_norm(k) = Φ(k) / ⟨Φ_random(k)⟩
//! ```
//!
//! against an ensemble of graphs that preserve every node's degree but rewire
//! everything else, and only a ratio above one is evidence.
//! [`normalised_rich_club`] computes it.
//!
//! # The null model, and why its parameters are the caller's
//!
//! The ensemble is built by repeated *double-edge swaps*: take two edges
//! `(a, b)` and `(c, d)` and replace them with `(a, d)` and `(c, b)`. Each node
//! keeps exactly the degree it had, so the club membership at every threshold is
//! identical in every sample and only the edges *among* the club change — which
//! is what makes the ratio a statement about wiring rather than about degree.
//!
//! How many samples, and how hard each is rewired, change the answer's precision
//! and its independence from the original graph. Neither has a defensible
//! library default, so [`RandomisationConfig`] takes both explicitly along with a
//! seed, and the result reports the ensemble it was measured against.
//!
//! Swaps that would create a self-loop or a duplicate edge are rejected rather
//! than applied, so the ensemble stays simple graphs; the accepted fraction is
//! reported, because a low one means the graph is too dense or too constrained
//! to rewire and the ensemble is not independent of where it started.
//!
//! # When the answer is one and that is correct
//!
//! A degree sequence can leave the club no freedom. Four nodes of degree five
//! among eight degree-one leaves has exactly six slots for edges between the
//! four — every pair — so *every* graph with that sequence has a complete club,
//! and a ratio of one is the true answer rather than a failure to detect
//! anything. The same happens in any degree-regular graph, where the club is
//! the whole graph at every threshold.
//!
//! Read the ratio with the acceptance fraction for this reason: a ratio near one
//! means either that the wiring is unremarkable or that the degree sequence
//! never allowed it to be remarkable, and the acceptance fraction is what
//! separates the two.
//!
//! # References
//!
//! * Colizza, V., Flammini, A., Serrano, M. A. & Vespignani, A. (2006).
//!   Detecting rich-club ordering in complex networks. *Nature Physics*
//!   2:110–115.
//! * van den Heuvel, M. P. & Sporns, O. (2011). Rich-club organization of the
//!   human connectome. *Journal of Neuroscience* 31(44):15775–15786.

use serde::{Deserialize, Serialize};

mod normalisation;

pub use normalisation::{
    NormalisedRichClubLevel, RandomisationConfig, RandomisationReport, normalised_rich_club,
};

use crate::ConnectivityMatrix;

/// The rich-club coefficient at one degree threshold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RichClubLevel {
    /// Degree threshold `k`; the club is the nodes with degree strictly above it.
    pub degree: usize,
    /// Number of nodes in the club.
    pub node_count: usize,
    /// Number of edges among them.
    pub edge_count: usize,
    /// Unnormalised coefficient `Φ(k)`, in `[0, 1]`.
    pub coefficient: f64,
    /// Mean weight of the club's edges, or zero when it has none.
    ///
    /// The weighted counterpart of the coefficient: a club can be topologically
    /// complete while its edges are individually weak.
    pub mean_weight: f64,
}

/// Rich-club coefficients for every degree threshold with at least two members.
///
/// Thresholds run from zero upward and stop where the club falls below two
/// nodes, since a club of one has no pairs and its coefficient is undefined
/// rather than zero.
#[must_use]
pub fn rich_club(matrix: &ConnectivityMatrix) -> Vec<RichClubLevel> {
    let n = matrix.region_count();
    let degrees: Vec<usize> = (0..n).map(|i| matrix.degree_at(i)).collect();
    let Some(peak) = degrees.iter().copied().max() else {
        return Vec::new();
    };

    (0..peak)
        .filter_map(|threshold| level_at(matrix, &degrees, threshold))
        .collect()
}

/// One threshold, or `None` when fewer than two nodes qualify.
fn level_at(
    matrix: &ConnectivityMatrix,
    degrees: &[usize],
    threshold: usize,
) -> Option<RichClubLevel> {
    let club: Vec<usize> = degrees
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree > threshold)
        .map(|(node, _)| node)
        .collect();
    if club.len() < 2 {
        return None;
    }

    let mut edge_count = 0_usize;
    let mut weight_total = 0.0;
    for (position, first) in club.iter().enumerate() {
        for second in &club[position + 1..] {
            let weight = matrix.weight_at(*first, *second);
            if weight > 0.0 {
                edge_count += 1;
                weight_total += weight;
            }
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "node and edge counts stay far below f64's exact-integer range"
    )]
    let pairs = (club.len() * (club.len() - 1) / 2) as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "node and edge counts stay far below f64's exact-integer range"
    )]
    let edges = edge_count as f64;

    Some(RichClubLevel {
        degree: threshold,
        node_count: club.len(),
        edge_count,
        coefficient: edges / pairs,
        mean_weight: if edge_count == 0 {
            0.0
        } else {
            weight_total / edges
        },
    })
}

#[cfg(test)]
mod tests;

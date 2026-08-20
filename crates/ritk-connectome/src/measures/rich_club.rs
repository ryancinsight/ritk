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
//! # The normalisation this does not do
//!
//! `Φ(k)` rises with `k` in *any* graph, because high-degree nodes have more
//! edges and so are likelier to be connected to each other by chance alone. The
//! published measure is therefore the ratio `Φ(k)/Φ_random(k)` against
//! degree-preserving randomised graphs, and only a ratio above one is evidence
//! of a rich club.
//!
//! This module returns the unnormalised `Φ(k)`. Producing `Φ_random` requires
//! generating an ensemble of degree-preserving rewirings, which is a random
//! process whose result depends on the ensemble size and the rewiring scheme —
//! parameters that belong to the caller's study design, not to a library
//! default. Reporting the raw curve and saying so is the honest position:
//! **a rising `Φ(k)` from this function is not by itself evidence of a rich
//! club.**
//!
//! # References
//!
//! * Colizza, V., Flammini, A., Serrano, M. A. & Vespignani, A. (2006).
//!   Detecting rich-club ordering in complex networks. *Nature Physics*
//!   2:110–115.
//! * van den Heuvel, M. P. & Sporns, O. (2011). Rich-club organization of the
//!   human connectome. *Journal of Neuroscience* 31(44):15775–15786.

use serde::{Deserialize, Serialize};

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

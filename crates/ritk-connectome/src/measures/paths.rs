//! Shortest paths, and the efficiency measures derived from them.
//!
//! # Weight to distance
//!
//! A connectome weight is a strength, and a shortest-path algorithm needs a
//! length. The conversion is `length = 1/weight`, applied here and nowhere else,
//! so a strong connection becomes a short step. An absent edge has weight zero
//! and therefore infinite length, which is the same statement as "there is no
//! step".
//!
//! # Why Dijkstra rather than Floyd-Warshall
//!
//! A whole-brain connectome has of order a hundred nodes but is far from
//! complete: a typical density is a few tenths, so a node has tens of neighbours
//! rather than hundreds. Dijkstra from every source costs `O(n·m·log n)` against
//! Floyd-Warshall's `O(n³)`, and the sparser the graph the wider the gap. All
//! weights are nonnegative — a streamline count cannot be negative — so
//! Dijkstra's precondition holds by construction.
//!
//! # Efficiency
//!
//! Global efficiency is the mean of `1/d` over ordered node pairs. Its value
//! over path length is that it stays defined when the graph is disconnected: an
//! unreachable pair has `d = ∞` and contributes zero, rather than making the
//! mean infinite. That makes it the summary to reach for on real connectomes,
//! which are frequently fragmented.
//!
//! Local efficiency is the same quantity computed on the subgraph induced by one
//! node's neighbours, with that node removed. It measures how well a node's
//! neighbourhood still communicates without it — the fault-tolerance reading of
//! clustering, and the reason it is reported alongside the clustering
//! coefficient rather than instead of it.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ConnectivityMatrix;

/// A node awaiting expansion, ordered by ascending tentative distance.
///
/// `BinaryHeap` is a max-heap, so the ordering is reversed. `total_cmp` is used
/// rather than `partial_cmp`: distances are finite positive reals here — a NaN
/// would mean a NaN weight, which construction cannot produce — and `total_cmp`
/// makes the ordering total without an unwrap that would panic if one ever did.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Frontier {
    distance: f64,
    node: usize,
}

impl Eq for Frontier {}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Shortest-path distances from every node to every other, row-major `n × n`.
///
/// The diagonal is zero and an unreachable pair is [`f64::INFINITY`].
#[must_use]
pub fn all_pairs_shortest_paths(matrix: &ConnectivityMatrix) -> Vec<f64> {
    let n = matrix.region_count();
    let mut distances = vec![f64::INFINITY; n * n];
    for source in 0..n {
        let row = dijkstra(matrix, source, None);
        distances[source * n..(source + 1) * n].copy_from_slice(&row);
    }
    distances
}

/// Shortest-path distances from `source`, optionally restricted to `subset`.
///
/// `subset` is a membership mask over the node indices; when supplied, only
/// nodes it admits may be visited. That is what lets the local-efficiency
/// computation run on a neighbourhood without materialising a subgraph.
fn dijkstra(matrix: &ConnectivityMatrix, source: usize, subset: Option<&[bool]>) -> Vec<f64> {
    let n = matrix.region_count();
    let mut distances = vec![f64::INFINITY; n];
    if subset.is_some_and(|mask| !mask[source]) {
        return distances;
    }
    distances[source] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(Frontier {
        distance: 0.0,
        node: source,
    });

    while let Some(Frontier { distance, node }) = heap.pop() {
        // A node can be pushed more than once; the first pop is the settled
        // distance and any later one is stale.
        if distance > distances[node] {
            continue;
        }
        for (neighbour, weight) in matrix.row(node).iter().enumerate() {
            if neighbour == node || *weight <= 0.0 {
                continue;
            }
            if subset.is_some_and(|mask| !mask[neighbour]) {
                continue;
            }
            // Strength to distance: the single point of inversion.
            let step = distance + 1.0 / weight;
            if step < distances[neighbour] {
                distances[neighbour] = step;
                heap.push(Frontier {
                    distance: step,
                    node: neighbour,
                });
            }
        }
    }
    distances
}

/// Whole-graph summaries derived from an all-pairs distance matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSummary {
    /// Mean distance over reachable ordered pairs, or `None` when none is.
    pub characteristic_path_length: Option<f64>,
    /// Fraction of ordered node pairs that are reachable.
    pub reachable_pair_fraction: f64,
    /// Mean reciprocal distance over ordered pairs, unreachable counted as zero.
    pub global_efficiency: f64,
}

/// Summarise an all-pairs distance matrix from [`all_pairs_shortest_paths`].
#[must_use]
pub fn summarise(distances: &[f64]) -> PathSummary {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the matrix is square by construction, so its side is an exact integer square root"
    )]
    let n = (distances.len() as f64).sqrt().round() as usize;
    let pairs = n.saturating_sub(1) * n;
    if pairs == 0 {
        return PathSummary {
            characteristic_path_length: None,
            reachable_pair_fraction: 0.0,
            global_efficiency: 0.0,
        };
    }

    let mut reachable = 0_usize;
    let mut distance_total = 0.0;
    let mut reciprocal_total = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let distance = distances[i * n + j];
            if distance.is_finite() {
                reachable += 1;
                distance_total += distance;
                reciprocal_total += 1.0 / distance;
            }
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "pair counts stay far below f64's exact-integer range"
    )]
    let pair_count = pairs as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "pair counts stay far below f64's exact-integer range"
    )]
    let reachable_count = reachable as f64;

    PathSummary {
        characteristic_path_length: (reachable > 0).then(|| distance_total / reachable_count),
        reachable_pair_fraction: reachable_count / pair_count,
        global_efficiency: reciprocal_total / pair_count,
    }
}

/// Local efficiency per node, in matrix-index order.
///
/// For node `i`, the global efficiency of the subgraph induced on `i`'s
/// neighbours with `i` itself removed. A node with fewer than two neighbours has
/// no pairs in its neighbourhood and therefore an efficiency of zero, which is
/// the limiting value rather than a missing one.
#[must_use]
pub fn local_efficiency(matrix: &ConnectivityMatrix) -> Box<[f64]> {
    let n = matrix.region_count();
    (0..n)
        .map(|node| {
            let mut mask = vec![false; n];
            let mut neighbours = Vec::new();
            for (candidate, weight) in matrix.row(node).iter().enumerate() {
                if candidate != node && *weight > 0.0 {
                    mask[candidate] = true;
                    neighbours.push(candidate);
                }
            }
            if neighbours.len() < 2 {
                return 0.0;
            }
            // The node itself is excluded from its own neighbourhood, so paths
            // between its neighbours must route around it. Including it would
            // make every neighbourhood trivially efficient — the node connects
            // to all of them by definition.
            let mut reciprocal_total = 0.0;
            for source in &neighbours {
                let distances = dijkstra(matrix, *source, Some(&mask));
                for target in &neighbours {
                    if source == target {
                        continue;
                    }
                    let distance = distances[*target];
                    if distance.is_finite() {
                        reciprocal_total += 1.0 / distance;
                    }
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "neighbour counts stay far below f64's exact-integer range"
            )]
            let pairs = (neighbours.len() * (neighbours.len() - 1)) as f64;
            reciprocal_total / pairs
        })
        .collect()
}

/// Sizes of the graph's connected components, descending.
///
/// Computed by breadth-first traversal over the binary topology: whether two
/// nodes are in the same component depends on which edges exist, not on how
/// heavy they are.
#[must_use]
pub fn component_sizes(matrix: &ConnectivityMatrix) -> Box<[usize]> {
    let n = matrix.region_count();
    let mut visited = vec![false; n];
    let mut sizes = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut size = 0_usize;
        while let Some(node) = stack.pop() {
            size += 1;
            for (neighbour, weight) in matrix.row(node).iter().enumerate() {
                if neighbour != node && *weight > 0.0 && !visited[neighbour] {
                    visited[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
        sizes.push(size);
    }
    sizes.sort_unstable_by(|left, right| right.cmp(left));
    sizes.into_boxed_slice()
}

#[cfg(test)]
mod tests;

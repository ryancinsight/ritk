//! Betweenness centrality — how much of the graph's traffic a node carries.
//!
//! A node's betweenness is the fraction of all shortest paths that pass through
//! it:
//!
//! ```text
//! B(v) = Σ_{s ≠ v ≠ t} σ_st(v) / σ_st
//! ```
//!
//! where `σ_st` is the number of shortest paths from `s` to `t` and `σ_st(v)` the
//! number of those running through `v`. It is the measure that identifies hubs
//! whose position matters rather than whose degree is large: a node of modest
//! degree bridging two otherwise separate modules carries enormous traffic, and
//! degree does not see it.
//!
//! # Why Brandes
//!
//! Evaluating the definition directly means enumerating shortest paths, which is
//! exponential in the worst case. Brandes' algorithm computes all betweennesses
//! in `O(n·m + n²·log n)` by never enumerating a path. The observation it rests
//! on is that the dependency
//!
//! ```text
//! δ_s(v) = Σ_{w : v ∈ pred(w)} (σ_sv / σ_sw) · (1 + δ_s(w))
//! ```
//!
//! satisfies a recurrence over the shortest-path DAG from `s`. So one Dijkstra
//! per source builds the DAG and records the predecessors and path counts, and
//! one backward sweep in order of *decreasing* distance accumulates the
//! dependencies — each node's contribution is complete by the time it is
//! reached, because every node it feeds is further from the source.
//!
//! # Normalisation
//!
//! Raw betweenness grows quadratically with the node count, so a value is
//! meaningless without knowing `n`. Dividing by `(n−1)(n−2)` — the number of
//! ordered pairs excluding the node — puts it on `[0, 1]`, where `1` is a node
//! every shortest path traverses. That makes the measure comparable between
//! parcellations of different granularity, which is the usual reason to compute
//! it.
//!
//! # References
//!
//! * Brandes, U. (2001). A faster algorithm for betweenness centrality.
//!   *Journal of Mathematical Sociology* 25(2):163–177.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ConnectivityMatrix;

/// A node awaiting expansion, ordered by ascending tentative distance.
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

/// Relative tolerance for calling two path lengths equal.
///
/// Two routes of genuinely equal length reach a node through different sums of
/// reciprocal weights, and floating-point addition is not associative, so the
/// two totals can differ in the last few bits. Without a tolerance one arbitrary
/// route would be recorded as strictly shorter and the other dropped from the
/// path count, which silently biases the centrality of every node on the
/// discarded route. The tolerance is relative because path lengths are sums of
/// reciprocal weights and so have no fixed scale; a few hundred epsilons covers
/// the accumulated rounding of a path of realistic depth while staying far below
/// any real difference in length.
const EQUAL_PATH_TOLERANCE: f64 = 256.0 * f64::EPSILON;

/// Normalised betweenness centrality per node, in matrix-index order.
///
/// Values lie in `[0, 1]`. A graph with fewer than three nodes has no
/// intermediary position to occupy, so every value is zero.
#[must_use]
pub fn betweenness(matrix: &ConnectivityMatrix) -> Box<[f64]> {
    let n = matrix.region_count();
    let mut centrality = vec![0.0_f64; n];
    if n < 3 {
        return centrality.into_boxed_slice();
    }

    // One set of buffers for the whole sweep. Brandes runs a full Dijkstra per
    // source, and allocating its five working arrays — plus a predecessor list
    // per node — inside that loop makes the allocation count quadratic in the
    // node count. Reusing them keeps the predecessor lists' capacity across
    // sources, so the repeated sweeps stop paying for the same growth.
    let mut scratch = Scratch::new(n);
    for source in 0..n {
        accumulate_from(matrix, source, &mut scratch, &mut centrality);
    }

    // Two factors of two cancel. The sweep visits each unordered pair twice,
    // once from each endpoint, so the undirected betweenness is half the
    // accumulated total; and an undirected graph has (n−1)(n−2)/2 unordered
    // intermediary pairs rather than (n−1)(n−2) ordered ones. Dividing the
    // accumulated total by the full (n−1)(n−2) applies both at once.
    #[expect(
        clippy::cast_precision_loss,
        reason = "node counts stay far below f64's exact-integer range"
    )]
    let normaliser = ((n - 1) * (n - 2)) as f64;
    for value in &mut centrality {
        *value /= normaliser;
    }
    centrality.into_boxed_slice()
}

/// Per-source working state, reused across every source in one sweep.
struct Scratch {
    distance: Vec<f64>,
    path_count: Vec<f64>,
    predecessors: Vec<Vec<usize>>,
    /// Nodes in the order they were settled, which is by increasing distance —
    /// exactly the order the backward sweep needs reversed.
    settled_order: Vec<usize>,
    settled: Vec<bool>,
    dependency: Vec<f64>,
    heap: BinaryHeap<Frontier>,
}

impl Scratch {
    fn new(nodes: usize) -> Self {
        Self {
            distance: vec![f64::INFINITY; nodes],
            path_count: vec![0.0; nodes],
            predecessors: vec![Vec::new(); nodes],
            settled_order: Vec::with_capacity(nodes),
            settled: vec![false; nodes],
            dependency: vec![0.0; nodes],
            heap: BinaryHeap::new(),
        }
    }

    /// Return every buffer to its start-of-source state.
    ///
    /// The predecessor lists are cleared rather than reallocated, which is the
    /// point of holding them: their capacity is what a fresh source would
    /// otherwise have to grow again.
    fn reset(&mut self) {
        self.distance.fill(f64::INFINITY);
        self.path_count.fill(0.0);
        for list in &mut self.predecessors {
            list.clear();
        }
        self.settled_order.clear();
        self.settled.fill(false);
        self.dependency.fill(0.0);
        self.heap.clear();
    }
}

/// One Brandes source: build the shortest-path DAG, then sweep it backwards.
fn accumulate_from(
    matrix: &ConnectivityMatrix,
    source: usize,
    scratch: &mut Scratch,
    centrality: &mut [f64],
) {
    scratch.reset();
    let Scratch {
        distance,
        path_count,
        predecessors,
        settled_order,
        settled,
        dependency,
        heap,
    } = scratch;

    distance[source] = 0.0;
    path_count[source] = 1.0;
    heap.push(Frontier {
        distance: 0.0,
        node: source,
    });

    while let Some(Frontier { node, .. }) = heap.pop() {
        if settled[node] {
            continue;
        }
        settled[node] = true;
        settled_order.push(node);

        for (neighbour, weight) in matrix.row(node).iter().enumerate() {
            if neighbour == node || *weight <= 0.0 {
                continue;
            }
            let candidate = distance[node] + 1.0 / weight;
            let known = distance[neighbour];
            // A node not yet reached is at infinite distance, and scaling a
            // relative tolerance by that gives infinity — which would make
            // `candidate < known - tolerance` compare against a NaN and route
            // the first visit into the equal-length branch, leaving the node
            // unreached forever. The first arrival is unambiguously an
            // improvement, so it carries no tolerance.
            let tolerance = if known.is_finite() {
                EQUAL_PATH_TOLERANCE * known.abs().max(candidate.abs())
            } else {
                0.0
            };

            if candidate < known - tolerance {
                // A strictly shorter route: everything recorded for this node so
                // far described a longer path and is now wrong.
                distance[neighbour] = candidate;
                path_count[neighbour] = path_count[node];
                predecessors[neighbour].clear();
                predecessors[neighbour].push(node);
                heap.push(Frontier {
                    distance: candidate,
                    node: neighbour,
                });
            } else if (candidate - known).abs() <= tolerance && !settled[neighbour] {
                // An equally short route: it adds paths rather than replacing
                // them. Settled nodes are skipped because their dependency has
                // already been fixed.
                path_count[neighbour] += path_count[node];
                predecessors[neighbour].push(node);
            }
        }
    }

    // Backward sweep: a node's dependency is complete once every node it feeds
    // has been processed, and those are all further from the source.
    for node in settled_order.iter().rev() {
        for predecessor in &predecessors[*node] {
            if path_count[*node] > 0.0 {
                dependency[*predecessor] +=
                    (path_count[*predecessor] / path_count[*node]) * (1.0 + dependency[*node]);
            }
        }
        if *node != source {
            centrality[*node] += dependency[*node];
        }
    }
}

#[cfg(test)]
mod tests;

//! Community structure — which regions form modules.
//!
//! # Modularity
//!
//! A partition of the nodes is good if it puts more weight inside communities
//! than chance would. "Chance" is made precise by the configuration null model:
//! rewire the graph at random while preserving every node's strength, and the
//! expected weight between `i` and `j` becomes `kᵢkⱼ / 2m`. Modularity is the
//! excess over that expectation, summed inside communities and normalised by the
//! total weight:
//!
//! ```text
//! Q = (1 / 2m) · Σᵢⱼ [ wᵢⱼ − kᵢkⱼ / 2m ] δ(cᵢ, cⱼ)
//! ```
//!
//! `Q` is positive when communities hold more weight than expected, zero for a
//! partition no better than chance, and negative for one worse. Preserving
//! strength in the null model is what stops the measure from simply rediscovering
//! that high-strength nodes have heavy edges.
//!
//! # Louvain
//!
//! Maximising `Q` exactly is NP-hard, so the standard approach is the greedy
//! two-phase scheme of Blondel et al.: repeatedly move single nodes to whichever
//! neighbouring community increases `Q` most, then collapse each community into
//! one node and repeat on the smaller graph. Each phase is cheap and the
//! collapse means the second pass optimises over communities rather than nodes,
//! which is how the method escapes the local optimum a single pass of node moves
//! would settle into.
//!
//! # Determinism
//!
//! The published algorithm visits nodes in random order, which makes the result
//! vary between runs on the same input. That is unacceptable here: a connectome
//! measure that changes when recomputed cannot be compared between subjects or
//! reproduced from a paper. This implementation visits nodes in index order
//! throughout, so the partition is a function of the matrix alone.
//!
//! The cost of fixing the order is a partition that may differ from the best one
//! a randomised search would find, and the modularity reported is the modularity
//! of the partition actually returned — never an optimum it did not reach.
//!
//! # The resolution limit
//!
//! Modularity maximisation cannot resolve communities smaller than roughly
//! `√(2m)` in total weight: below that size, merging two genuinely separate
//! communities *increases* `Q`. This is a property of the objective, not of the
//! search, so no amount of optimisation effort avoids it — a fine-grained
//! parcellation of a strongly connected brain can have its small modules merged
//! by any modularity-based method.
//!
//! # References
//!
//! * Newman, M. E. J. (2004). Analysis of weighted networks. *Physical Review E*
//!   70:056131. — weighted modularity.
//! * Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008). Fast
//!   unfolding of communities in large networks. *Journal of Statistical
//!   Mechanics* 2008:P10008. — the Louvain method.
//! * Fortunato, S. & Barthélemy, M. (2007). Resolution limit in community
//!   detection. *PNAS* 104(1):36–41.

use serde::{Deserialize, Serialize};

use crate::ConnectivityMatrix;

/// A partition of the nodes into communities, with its modularity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Communities {
    /// Community index per node, in matrix-index order. Indices are compacted,
    /// so a partition into `c` communities uses exactly `0..c`.
    assignment: Box<[usize]>,
    /// Number of communities.
    count: usize,
    /// Modularity of this partition.
    modularity: f64,
}

impl Communities {
    /// Community index per node, in matrix-index order.
    #[must_use]
    pub const fn assignment(&self) -> &[usize] {
        &self.assignment
    }

    /// Number of communities.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Modularity of this partition — the value actually achieved, not a bound.
    #[must_use]
    pub const fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Node indices belonging to each community, in community order.
    #[must_use]
    pub fn members(&self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.count];
        for (node, community) in self.assignment.iter().enumerate() {
            groups[*community].push(node);
        }
        groups
    }
}

/// Modularity of an arbitrary partition.
///
/// `assignment` gives a community index per node; the indices need not be
/// contiguous. Returns zero for a graph with no edge weight, which has no
/// structure for a partition to capture.
///
/// # Panics
///
/// If `assignment` does not cover every node.
#[must_use]
pub fn modularity(matrix: &ConnectivityMatrix, assignment: &[usize]) -> f64 {
    let n = matrix.region_count();
    assert_eq!(
        assignment.len(),
        n,
        "a partition must assign every node to a community"
    );

    let total = total_weight(matrix);
    if total <= 0.0 {
        return 0.0;
    }
    let strengths: Vec<f64> = (0..n).map(|i| matrix.strength_at(i)).collect();

    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if assignment[i] != assignment[j] {
                continue;
            }
            // The diagonal contributes no *observed* weight: a self-connection
            // is not a link between two nodes, and counting it would let a
            // region raise the modularity of a community it is alone in. It does
            // contribute to the *expected* term, because the configuration null
            // model admits self-links and the normalisation depends on it —
            // dropping the diagonal there would leave the all-in-one partition
            // at `Σkᵢ²/4m²` instead of the zero the measure is defined against,
            // shifting every score by an amount that varies with the graph.
            let observed = if i == j { 0.0 } else { matrix.weight_at(i, j) };
            sum += observed - strengths[i] * strengths[j] / (2.0 * total);
        }
    }
    sum / (2.0 * total)
}

/// Detect communities by the deterministic Louvain method.
///
/// A graph with no edges returns every node in its own community with a
/// modularity of zero, which is the correct answer rather than a failure.
#[must_use]
pub fn louvain(matrix: &ConnectivityMatrix) -> Communities {
    let n = matrix.region_count();
    if n == 0 {
        return Communities {
            assignment: Box::new([]),
            count: 0,
            modularity: 0.0,
        };
    }

    // The algorithm works on a shrinking graph. `level` is the current one, and
    // `assignment` maps original nodes to their community at the current level.
    let mut level = Level::from_matrix(matrix);
    let mut assignment: Vec<usize> = (0..n).collect();

    loop {
        let moved = level.optimise_locally();
        let local = level.compact();
        // Carry the level's partition back to the original nodes.
        for community in &mut assignment {
            *community = local[*community];
        }
        if !moved || level.community_count(&local) == level.size() {
            break;
        }
        level = level.aggregate(&local);
    }

    let count = compact_in_place(&mut assignment);
    Communities {
        modularity: modularity(matrix, &assignment),
        assignment: assignment.into_boxed_slice(),
        count,
    }
}

/// One level of the Louvain hierarchy: a weighted graph plus a node-to-community
/// map over it.
struct Level {
    size: usize,
    /// Dense symmetric weights over this level's nodes.
    weights: Vec<f64>,
    /// Self-weight per node — the internal weight of the community it collapsed
    /// from. Zero at the first level.
    self_weight: Vec<f64>,
    community: Vec<usize>,
    /// Total weight of every edge, counting each once.
    total: f64,
}

impl Level {
    fn from_matrix(matrix: &ConnectivityMatrix) -> Self {
        let n = matrix.region_count();
        let mut weights = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    weights[i * n + j] = matrix.weight_at(i, j);
                }
            }
        }
        let total = total_weight(matrix);
        Self {
            size: n,
            weights,
            self_weight: vec![0.0; n],
            community: (0..n).collect(),
            total,
        }
    }

    const fn size(&self) -> usize {
        self.size
    }

    fn degree_of(&self, node: usize) -> f64 {
        self.weights[node * self.size..(node + 1) * self.size]
            .iter()
            .sum::<f64>()
            + 2.0 * self.self_weight[node]
    }

    /// Move nodes between communities until no single move improves modularity.
    ///
    /// Returns whether anything moved.
    fn optimise_locally(&mut self) -> bool {
        if self.total <= 0.0 {
            return false;
        }
        let degrees: Vec<f64> = (0..self.size).map(|node| self.degree_of(node)).collect();
        let mut community_degree = vec![0.0; self.size];
        for (node, degree) in degrees.iter().enumerate() {
            community_degree[self.community[node]] += degree;
        }

        let mut moved_ever = false;
        // Sweeps are bounded: each one either improves the partition or ends the
        // loop, and modularity is bounded above, but a cap keeps a pathological
        // oscillation from running unbounded.
        for _ in 0..MAX_SWEEPS {
            let mut moved_this_sweep = false;
            for (node, node_degree) in degrees.iter().copied().enumerate() {
                let origin = self.community[node];
                community_degree[origin] -= node_degree;

                // Weight from this node into each candidate community.
                let mut links: Vec<(usize, f64)> = Vec::new();
                for neighbour in 0..self.size {
                    let weight = self.weights[node * self.size + neighbour];
                    if neighbour == node || weight <= 0.0 {
                        continue;
                    }
                    let target = self.community[neighbour];
                    match links.iter_mut().find(|(candidate, _)| *candidate == target) {
                        Some((_, total)) => *total += weight,
                        None => links.push((target, weight)),
                    }
                }

                let gain_of = |community: usize, link: f64| {
                    // The modularity change from placing the node here, dropping
                    // the terms that do not depend on the choice.
                    link - community_degree[community] * node_degree / (2.0 * self.total)
                };
                let origin_link = links
                    .iter()
                    .find(|(candidate, _)| *candidate == origin)
                    .map_or(0.0, |(_, weight)| *weight);

                let mut best = (origin, gain_of(origin, origin_link));
                for (candidate, link) in &links {
                    let gain = gain_of(*candidate, *link);
                    // Strictly greater keeps the choice deterministic: an equal
                    // alternative never displaces the incumbent, so the outcome
                    // does not depend on the order communities were discovered.
                    if gain > best.1 {
                        best = (*candidate, gain);
                    }
                }

                community_degree[best.0] += node_degree;
                if best.0 != origin {
                    self.community[node] = best.0;
                    moved_this_sweep = true;
                    moved_ever = true;
                }
            }
            if !moved_this_sweep {
                break;
            }
        }
        moved_ever
    }

    /// Renumber this level's communities to `0..c`, returning the map.
    fn compact(&self) -> Vec<usize> {
        let mut map = self.community.clone();
        compact_in_place(&mut map);
        map
    }

    fn community_count(&self, compacted: &[usize]) -> usize {
        compacted.iter().copied().max().map_or(0, |peak| peak + 1)
    }

    /// Collapse each community into a single node.
    fn aggregate(&self, compacted: &[usize]) -> Self {
        let size = self.community_count(compacted);
        let mut weights = vec![0.0; size * size];
        let mut self_weight = vec![0.0; size];

        for (i, ci) in compacted.iter().copied().enumerate() {
            self_weight[ci] += self.self_weight[i];
            for (j, cj) in compacted.iter().copied().enumerate() {
                let weight = self.weights[i * self.size + j];
                if weight <= 0.0 {
                    continue;
                }
                if ci == cj {
                    // Each internal edge is seen twice, once per direction, and
                    // the self-weight counts it once.
                    self_weight[ci] += weight / 2.0;
                } else {
                    weights[ci * size + cj] += weight;
                }
            }
        }

        Self {
            size,
            weights,
            self_weight,
            community: (0..size).collect(),
            total: self.total,
        }
    }
}

/// Bound on local-moving sweeps within one level.
///
/// Each sweep that changes nothing terminates the loop, and modularity increases
/// monotonically across sweeps, so the loop converges on its own. The cap exists
/// so that a floating-point plateau — where two moves each appear to improve on
/// the other by a rounding step — cannot spin forever.
const MAX_SWEEPS: usize = 64;

/// Renumber arbitrary community indices to a contiguous `0..c`, returning `c`.
///
/// The renumbering follows first appearance in node order, so it is a function
/// of the partition alone.
fn compact_in_place(assignment: &mut [usize]) -> usize {
    let mut mapping: Vec<(usize, usize)> = Vec::new();
    for community in assignment.iter_mut() {
        let next = mapping.len();
        let compacted = match mapping.iter().find(|(from, _)| *from == *community) {
            Some((_, to)) => *to,
            None => {
                mapping.push((*community, next));
                next
            }
        };
        *community = compacted;
    }
    mapping.len()
}

/// Total edge weight, counting each undirected edge once and excluding the
/// diagonal.
fn total_weight(matrix: &ConnectivityMatrix) -> f64 {
    let n = matrix.region_count();
    (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .map(|(i, j)| matrix.weight_at(i, j))
        .sum()
}

#[cfg(test)]
mod tests;

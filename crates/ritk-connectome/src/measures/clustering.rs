//! Clustering coefficients — how completely a node's neighbours interconnect.
//!
//! # Binary
//!
//! For a node with `k` neighbours there are `k(k−1)/2` possible links among
//! them. The coefficient is the fraction realised:
//!
//! ```text
//! C = 2t / (k(k − 1))
//! ```
//!
//! with `t` the number of triangles through the node. It is `1` for a node whose
//! neighbours form a clique and `0` for a star. A node with fewer than two
//! neighbours has no possible links, so its coefficient is the limiting `0`
//! rather than undefined — the convention throughout the network literature, and
//! what keeps the mean over all nodes computable.
//!
//! # Weighted
//!
//! The binary form throws away everything the weights say: a triangle closed by
//! three heavy edges counts the same as one closed by three negligible ones. The
//! Onnela form replaces the triangle count with the sum of the geometric means
//! of the three normalised weights,
//!
//! ```text
//! Cʷ = (2 / (k(k − 1))) · Σ_{j,h} (ŵᵢⱼ ŵⱼₕ ŵₕᵢ)^{1/3},   ŵ = w / max(w)
//! ```
//!
//! The geometric mean is the choice that makes the measure behave: it is zero if
//! any of the three edges is absent, so a missing edge closes no triangle, and
//! the cube root keeps the result on the same scale as the weights, so the
//! measure reduces exactly to the binary coefficient when every present edge has
//! the maximum weight. An arithmetic mean would have neither property — one
//! heavy edge would carry a triangle whose other two sides barely exist.
//!
//! Normalising by the maximum weight is what makes the result comparable across
//! matrices: the weights themselves are streamline counts, whose scale depends
//! on how many streamlines were seeded.
//!
//! # References
//!
//! * Onnela, J.-P., Saramäki, J., Kertész, J. & Kaski, K. (2005). Intensity and
//!   coherence of motifs in weighted complex networks. *Physical Review E*
//!   71:065103.

use crate::ConnectivityMatrix;

/// Binary clustering coefficient per node, in matrix-index order.
#[must_use]
pub fn binary(matrix: &ConnectivityMatrix) -> Box<[f64]> {
    let n = matrix.region_count();
    (0..n)
        .map(|node| {
            let neighbours = neighbours_of(matrix, node);
            let Some(pairs) = possible_links(neighbours.len()) else {
                return 0.0;
            };
            let mut triangles = 0_usize;
            for (position, first) in neighbours.iter().enumerate() {
                for second in &neighbours[position + 1..] {
                    if matrix.weight_at(*first, *second) > 0.0 {
                        triangles += 1;
                    }
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "triangle counts stay far below f64's exact-integer range"
            )]
            let closed = triangles as f64;
            closed / pairs
        })
        .collect()
}

/// Onnela weighted clustering coefficient per node, in matrix-index order.
#[must_use]
pub fn onnela(matrix: &ConnectivityMatrix) -> Box<[f64]> {
    let n = matrix.region_count();
    let peak = maximum_weight(matrix);
    if peak <= 0.0 {
        return vec![0.0; n].into_boxed_slice();
    }

    (0..n)
        .map(|node| {
            let neighbours = neighbours_of(matrix, node);
            let Some(pairs) = possible_links(neighbours.len()) else {
                return 0.0;
            };
            let mut intensity = 0.0;
            for (position, first) in neighbours.iter().enumerate() {
                for second in &neighbours[position + 1..] {
                    let closing = matrix.weight_at(*first, *second);
                    if closing <= 0.0 {
                        continue;
                    }
                    let product = (matrix.weight_at(node, *first) / peak)
                        * (matrix.weight_at(*first, *second) / peak)
                        * (matrix.weight_at(*second, node) / peak);
                    intensity += product.cbrt();
                }
            }
            intensity / pairs
        })
        .collect()
}

/// Neighbours of a node, excluding itself.
fn neighbours_of(matrix: &ConnectivityMatrix, node: usize) -> Vec<usize> {
    matrix
        .row(node)
        .iter()
        .enumerate()
        .filter(|(candidate, weight)| *candidate != node && **weight > 0.0)
        .map(|(candidate, _)| candidate)
        .collect()
}

/// `k(k−1)/2` as a float, or `None` when a node has too few neighbours to have
/// any pairs at all.
fn possible_links(degree: usize) -> Option<f64> {
    if degree < 2 {
        return None;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "degrees stay far below f64's exact-integer range"
    )]
    let pairs = (degree * (degree - 1) / 2) as f64;
    Some(pairs)
}

/// Largest off-diagonal weight, used to normalise the weighted form.
fn maximum_weight(matrix: &ConnectivityMatrix) -> f64 {
    let n = matrix.region_count();
    (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .map(|(i, j)| matrix.weight_at(i, j))
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests;

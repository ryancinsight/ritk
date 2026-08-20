//! Shared graph fixtures, and tests of the matrix itself.
//!
//! The fixtures build matrices directly rather than through
//! [`crate::build_connectivity_matrix`], so that a measure's test fails for a
//! defect in the measure rather than in the builder. The builder has its own
//! tests in [`crate::build`].

use super::*;

/// Assemble a matrix from an upper-triangular edge list.
///
/// `edges` are `(i, j, weight)` over matrix indices; each is written to both
/// `(i, j)` and `(j, i)`, so a fixture cannot accidentally build an asymmetric
/// matrix the measures would then misread.
pub(crate) fn matrix_from_edges(
    label_count: usize,
    edges: &[(usize, usize, f64)],
) -> ConnectivityMatrix {
    let mut weights = vec![0.0; label_count * label_count];
    for (i, j, weight) in edges {
        weights[i * label_count + j] = *weight;
        weights[j * label_count + i] = *weight;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixtures use small label counts"
    )]
    let labels: Box<[u32]> = (0..label_count as u32).collect();
    ConnectivityMatrix::from_parts(
        labels,
        weights.into_boxed_slice(),
        StreamlineAccounting::default(),
        EdgeWeighting::StreamlineCount,
    )
}

/// A path graph `0 — 1 — 2` with unit weights.
pub(crate) fn path_graph() -> ConnectivityMatrix {
    matrix_from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)])
}

/// A triangle on three nodes with unit weights.
pub(crate) fn triangle() -> ConnectivityMatrix {
    matrix_from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
}

/// A star: node 0 at the centre, nodes 1..=3 on the rim, unit weights.
pub(crate) fn star() -> ConnectivityMatrix {
    matrix_from_edges(4, &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)])
}

/// Two triangles joined by a single light bridge.
///
/// Nodes `0,1,2` and `3,4,5` are each a triangle of weight 10; the single edge
/// `2 — 3` has weight 1. The modular structure is unambiguous, which is what
/// makes it a usable oracle for community detection.
pub(crate) fn two_modules() -> ConnectivityMatrix {
    matrix_from_edges(
        6,
        &[
            (0, 1, 10.0),
            (1, 2, 10.0),
            (0, 2, 10.0),
            (3, 4, 10.0),
            (4, 5, 10.0),
            (3, 5, 10.0),
            (2, 3, 1.0),
        ],
    )
}

// ── Matrix accessors ─────────────────────────────────────────────────────

#[test]
fn weights_are_symmetric() {
    let matrix = two_modules();
    for i in 0..matrix.region_count() {
        for j in 0..matrix.region_count() {
            assert!(
                (matrix.weight_at(i, j) - matrix.weight_at(j, i)).abs() < f64::EPSILON,
                "({i}, {j}) is not symmetric"
            );
        }
    }
}

#[test]
fn labels_resolve_to_indices() {
    let matrix = path_graph();
    assert_eq!(matrix.index_of(1), Some(1));
    assert_eq!(matrix.index_of(99), None);
    assert_eq!(matrix.weight(0, 1), Some(1.0));
    assert_eq!(matrix.weight(0, 99), None);
}

#[test]
fn degree_and_strength_count_neighbours_and_weights() {
    let matrix = matrix_from_edges(3, &[(0, 1, 2.0), (0, 2, 5.0)]);
    assert_eq!(matrix.degree(0), Some(2));
    assert_eq!(matrix.degree(1), Some(1));
    assert_eq!(matrix.strength(0), Some(7.0));
    assert_eq!(matrix.strength(2), Some(5.0));
}

/// A self-connection is recorded but is not a link between two nodes, so it must
/// not raise the degree, the strength, or the edge count. Getting this wrong
/// would inflate every downstream measure for any region whose tractogram
/// contains intra-region streamlines — which is every region.
#[test]
fn a_self_connection_is_stored_but_excluded_from_degree_and_strength() {
    let matrix = matrix_from_edges(3, &[(0, 0, 7.0), (0, 1, 2.0)]);

    assert_eq!(matrix.weight(0, 0), Some(7.0));
    assert_eq!(matrix.degree(0), Some(1));
    assert_eq!(matrix.strength(0), Some(2.0));
    assert_eq!(matrix.edge_count(), 1);
}

#[test]
fn edges_lists_every_nonzero_pair_once() {
    let matrix = two_modules();
    let edges: Vec<_> = matrix.edges().collect();
    assert_eq!(edges.len(), 7);
    for edge in &edges {
        assert!(
            edge.source <= edge.target,
            "edges are listed once: {edge:?}"
        );
        assert!(edge.weight > 0.0);
    }
}

#[test]
fn density_is_edges_over_possible_pairs() {
    // A triangle is complete: 3 of 3 possible pairs.
    assert!((triangle().density() - 1.0).abs() < f64::EPSILON);
    // The path graph has 2 of 3.
    assert!((path_graph().density() - 2.0 / 3.0).abs() < 1.0e-12);
    // A single node has no pairs at all.
    assert_eq!(matrix_from_edges(1, &[]).density(), 0.0);
}

// ── Accounting ───────────────────────────────────────────────────────────

#[test]
fn the_assigned_fraction_is_zero_for_an_empty_tractogram() {
    assert_eq!(StreamlineAccounting::default().assigned_fraction(), 0.0);
}

#[test]
fn the_assigned_fraction_reports_the_share_that_produced_edges() {
    let accounting = StreamlineAccounting {
        total: 100,
        assigned: 40,
        intra_region: 35,
        unassigned: 25,
        degenerate: 3,
    };
    assert!((accounting.assigned_fraction() - 0.4).abs() < 1.0e-12);
}

// ── Serialisation ────────────────────────────────────────────────────────

#[test]
fn a_json_round_trip_preserves_weights_and_accounting() {
    let matrix = two_modules();
    let decoded =
        ConnectivityMatrix::from_json(&matrix.to_json().expect("serialise")).expect("deserialise");

    assert_eq!(decoded.region_labels(), matrix.region_labels());
    assert_eq!(decoded.accounting(), matrix.accounting());
    assert_eq!(decoded.weighting(), matrix.weighting());
    for i in 0..matrix.region_count() {
        for j in 0..matrix.region_count() {
            assert!(
                (decoded.weight_at(i, j) - matrix.weight_at(i, j)).abs() < f64::EPSILON,
                "({i}, {j})"
            );
        }
    }
}

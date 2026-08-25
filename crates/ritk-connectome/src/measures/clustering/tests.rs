use super::*;

use crate::tests::{matrix_from_edges, path_graph, star, triangle};

const TOLERANCE: f64 = 1.0e-12;

// ── Binary ───────────────────────────────────────────────────────────────

#[test]
fn a_triangle_clusters_completely() {
    for value in binary(&triangle()).iter() {
        assert!((value - 1.0).abs() < TOLERANCE, "got {value}");
    }
}

/// A star's centre has neighbours that are not joined to each other, so none of
/// its possible triangles is closed.
#[test]
fn the_centre_of_a_star_does_not_cluster() {
    assert_eq!(binary(&star())[0], 0.0);
}

/// Fewer than two neighbours means no possible links, so the coefficient takes
/// its conventional limiting value of zero rather than being undefined.
#[test]
fn a_node_with_under_two_neighbours_has_zero_clustering() {
    let coefficients = binary(&path_graph());
    assert_eq!(coefficients[0], 0.0);
    assert_eq!(coefficients[2], 0.0);
}

/// A closed-form intermediate: a node with three neighbours, one of the three
/// possible links among them present, gives `1/3`.
#[test]
fn a_partially_closed_neighbourhood_matches_its_closed_form() {
    let matrix = matrix_from_edges(4, &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0)]);
    assert!((binary(&matrix)[0] - 1.0 / 3.0).abs() < TOLERANCE);
}

/// The binary form ignores weights entirely — that is its defining limitation,
/// and the reason the weighted form exists.
#[test]
fn the_binary_form_is_blind_to_weight() {
    let light = matrix_from_edges(3, &[(0, 1, 0.001), (1, 2, 0.001), (0, 2, 0.001)]);
    let heavy = matrix_from_edges(3, &[(0, 1, 1000.0), (1, 2, 1000.0), (0, 2, 1000.0)]);
    assert_eq!(binary(&light), binary(&heavy));
}

// ── Onnela weighted ──────────────────────────────────────────────────────

/// When every present edge carries the maximum weight, the normalised weights
/// are all one and the geometric mean of three ones is one, so the weighted form
/// must reduce exactly to the binary one. This is the reduction that makes the
/// two comparable, and it holds by construction rather than approximately.
#[test]
fn the_weighted_form_reduces_to_the_binary_one_at_uniform_weight() {
    for matrix in [triangle(), star(), path_graph()] {
        let binary_values = binary(&matrix);
        let weighted_values = onnela(&matrix);
        for (node, (expected, actual)) in
            binary_values.iter().zip(weighted_values.iter()).enumerate()
        {
            assert!(
                (expected - actual).abs() < TOLERANCE,
                "node {node}: binary {expected}, weighted {actual}"
            );
        }
    }
}

/// A triangle closed by weak edges must score lower than one closed by strong
/// ones, which is precisely the distinction the binary form cannot make.
#[test]
fn a_weakly_closed_triangle_scores_below_a_strongly_closed_one() {
    // Node 0 sits in two triangles: one closed heavily, one closed faintly.
    let matrix = matrix_from_edges(
        5,
        &[
            (0, 1, 10.0),
            (0, 2, 10.0),
            (1, 2, 10.0), // heavy triangle
            (0, 3, 10.0),
            (0, 4, 10.0),
            (3, 4, 0.1), // faint closure
        ],
    );
    let weighted = onnela(&matrix);
    let binary_values = binary(&matrix);

    // Topologically the two closures are identical.
    assert!((binary_values[1] - binary_values[3]).abs() < TOLERANCE);
    // By weight they are not: node 1 sits in the heavy triangle, node 3 the faint.
    assert!(
        weighted[1] > weighted[3],
        "heavy closure {} must outscore faint closure {}",
        weighted[1],
        weighted[3]
    );
}

/// The geometric mean is zero if any side is missing, so an absent edge closes
/// no triangle regardless of how heavy the other two sides are.
#[test]
fn a_missing_edge_closes_no_triangle_however_heavy_the_others() {
    let matrix = matrix_from_edges(3, &[(0, 1, 1000.0), (0, 2, 1000.0)]);
    assert_eq!(onnela(&matrix)[0], 0.0);
}

/// Scaling every weight leaves the normalised weights unchanged, so the measure
/// is invariant — which is what makes it comparable between tractograms seeded
/// with different numbers of streamlines.
#[test]
fn the_weighted_form_is_invariant_under_uniform_scaling() {
    let base = matrix_from_edges(4, &[(0, 1, 3.0), (0, 2, 7.0), (1, 2, 2.0), (0, 3, 5.0)]);
    let scaled = matrix_from_edges(4, &[(0, 1, 30.0), (0, 2, 70.0), (1, 2, 20.0), (0, 3, 50.0)]);
    for (node, (left, right)) in onnela(&base).iter().zip(onnela(&scaled).iter()).enumerate() {
        assert!(
            (left - right).abs() < TOLERANCE,
            "node {node}: {left} vs {right}"
        );
    }
}

#[test]
fn a_graph_with_no_edges_has_zero_weighted_clustering() {
    let matrix = matrix_from_edges(3, &[]);
    assert_eq!(&*onnela(&matrix), &[0.0, 0.0, 0.0]);
}

/// Both forms are bounded by one, which the normalisation is there to guarantee.
#[test]
fn both_forms_stay_within_the_unit_range() {
    let matrix = matrix_from_edges(
        5,
        &[
            (0, 1, 4.0),
            (0, 2, 9.0),
            (1, 2, 1.0),
            (2, 3, 6.0),
            (3, 4, 2.0),
            (0, 4, 8.0),
            (1, 4, 3.0),
        ],
    );
    for (node, (binary_value, weighted_value)) in binary(&matrix)
        .iter()
        .zip(onnela(&matrix).iter())
        .enumerate()
    {
        assert!(
            (0.0..=1.0).contains(binary_value),
            "node {node} binary {binary_value}"
        );
        assert!(
            (0.0..=1.0).contains(weighted_value),
            "node {node} weighted {weighted_value}"
        );
        // The weighted form can never exceed the binary one: each triangle
        // contributes a normalised geometric mean of at most one where the
        // binary form contributes exactly one.
        assert!(
            weighted_value <= &(binary_value + TOLERANCE),
            "node {node}: weighted {weighted_value} exceeds binary {binary_value}"
        );
    }
}

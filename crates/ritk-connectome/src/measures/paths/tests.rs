use super::*;

use crate::tests::{matrix_from_edges, path_graph, star, triangle, two_modules};

const TOLERANCE: f64 = 1.0e-12;

fn distance_between(matrix: &ConnectivityMatrix, i: usize, j: usize) -> f64 {
    let n = matrix.region_count();
    all_pairs_shortest_paths(matrix)[i * n + j]
}

// ── Weight is inverted into distance ─────────────────────────────────────

/// The one convention everything rests on: a heavier edge is a *shorter* step.
///
/// Asserting the numeric value rather than an ordering is deliberate — an
/// implementation that used the weight directly as a length would still order
/// two paths consistently in some graphs, so only the value catches it.
#[test]
fn a_heavier_edge_is_a_shorter_step() {
    let matrix = matrix_from_edges(2, &[(0, 1, 4.0)]);
    assert!((distance_between(&matrix, 0, 1) - 0.25).abs() < TOLERANCE);

    let lighter = matrix_from_edges(2, &[(0, 1, 0.5)]);
    assert!((distance_between(&lighter, 0, 1) - 2.0).abs() < TOLERANCE);
}

/// The shortest path is the one minimising summed reciprocal weight, which is
/// not the one with the fewest hops. Here the two-hop route through heavy edges
/// beats the direct light one.
#[test]
fn the_shortest_path_can_take_more_hops_than_the_direct_edge() {
    // Direct 0—2 of weight 0.25 costs 4; via node 1 with weights 1 and 1 costs 2.
    let matrix = matrix_from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.25)]);
    assert!((distance_between(&matrix, 0, 2) - 2.0).abs() < TOLERANCE);
}

#[test]
fn a_node_is_at_zero_distance_from_itself() {
    let matrix = two_modules();
    for node in 0..matrix.region_count() {
        assert_eq!(distance_between(&matrix, node, node), 0.0, "node {node}");
    }
}

/// The graph is undirected, so the distance matrix must be symmetric. An
/// asymmetry would mean the traversal read the matrix in only one direction.
#[test]
fn the_distance_matrix_is_symmetric() {
    let matrix = two_modules();
    let n = matrix.region_count();
    let distances = all_pairs_shortest_paths(&matrix);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (distances[i * n + j] - distances[j * n + i]).abs() < TOLERANCE,
                "({i}, {j})"
            );
        }
    }
}

/// The triangle inequality is a property of shortest paths, so it must hold for
/// every triple. It is the check that catches a traversal that settled a node
/// too early.
#[test]
fn distances_satisfy_the_triangle_inequality() {
    let matrix = matrix_from_edges(
        5,
        &[
            (0, 1, 3.0),
            (1, 2, 0.5),
            (2, 3, 2.0),
            (3, 4, 1.0),
            (0, 4, 0.2),
            (1, 3, 4.0),
        ],
    );
    let n = matrix.region_count();
    let distances = all_pairs_shortest_paths(&matrix);
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                let direct = distances[i * n + j];
                let detour = distances[i * n + k] + distances[k * n + j];
                assert!(
                    direct <= detour + TOLERANCE,
                    "d({i},{j}) = {direct} exceeds d({i},{k}) + d({k},{j}) = {detour}"
                );
            }
        }
    }
}

// ── Disconnection ────────────────────────────────────────────────────────

#[test]
fn an_unreachable_pair_is_at_infinite_distance() {
    let matrix = matrix_from_edges(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
    assert!(distance_between(&matrix, 0, 2).is_infinite());
    assert!(distance_between(&matrix, 0, 1).is_finite());
}

/// Global efficiency stays defined on a disconnected graph, which is the whole
/// reason it is the reported summary. Two isolated unit edges leave 4 of 12
/// ordered pairs reachable at distance 1, so `E = 4/12`.
#[test]
fn global_efficiency_is_defined_when_the_graph_is_disconnected() {
    let matrix = matrix_from_edges(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
    let summary = summarise(&all_pairs_shortest_paths(&matrix));

    assert!((summary.global_efficiency - 4.0 / 12.0).abs() < TOLERANCE);
    assert!((summary.reachable_pair_fraction - 4.0 / 12.0).abs() < TOLERANCE);
    // The characteristic path length averages only what it can reach, which is
    // exactly why it must be read with the reachable fraction.
    assert!(
        (summary
            .characteristic_path_length
            .expect("some pairs are reachable")
            - 1.0)
            .abs()
            < TOLERANCE
    );
}

#[test]
fn a_graph_with_no_edges_has_no_characteristic_path_length() {
    let matrix = matrix_from_edges(3, &[]);
    let summary = summarise(&all_pairs_shortest_paths(&matrix));

    assert_eq!(summary.characteristic_path_length, None);
    assert_eq!(summary.reachable_pair_fraction, 0.0);
    assert_eq!(summary.global_efficiency, 0.0);
}

#[test]
fn a_single_node_graph_has_no_pairs() {
    let summary = summarise(&all_pairs_shortest_paths(&matrix_from_edges(1, &[])));
    assert_eq!(summary.characteristic_path_length, None);
    assert_eq!(summary.global_efficiency, 0.0);
}

/// A closed-form reference. The path graph `0—1—2` with unit weights has
/// distances 1, 1, and 2 among its three unordered pairs, so over the six
/// ordered pairs `L = (1+1+2)·2/6 = 4/3` and `E = (1+1+½)·2/6 = 5/6`.
#[test]
fn the_path_graph_matches_its_closed_form_summary() {
    let summary = summarise(&all_pairs_shortest_paths(&path_graph()));

    assert!((summary.characteristic_path_length.expect("connected") - 4.0 / 3.0).abs() < TOLERANCE);
    assert!((summary.global_efficiency - 5.0 / 6.0).abs() < TOLERANCE);
    assert!((summary.reachable_pair_fraction - 1.0).abs() < TOLERANCE);
}

/// A complete graph of unit weights has every pair at distance 1, so both
/// summaries are exactly 1 — the upper bound of each.
#[test]
fn a_complete_unit_graph_has_unit_path_length_and_efficiency() {
    let summary = summarise(&all_pairs_shortest_paths(&triangle()));
    assert!((summary.characteristic_path_length.expect("connected") - 1.0).abs() < TOLERANCE);
    assert!((summary.global_efficiency - 1.0).abs() < TOLERANCE);
}

// ── Components ───────────────────────────────────────────────────────────

#[test]
fn a_connected_graph_has_one_component() {
    assert_eq!(&*component_sizes(&two_modules()), &[6]);
}

#[test]
fn component_sizes_are_reported_descending() {
    // A triple, a pair, and two isolated nodes.
    let matrix = matrix_from_edges(7, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]);
    assert_eq!(&*component_sizes(&matrix), &[3, 2, 1, 1]);
}

/// An isolated node is its own component, and a self-connection does not join it
/// to anything.
#[test]
fn a_self_connection_does_not_join_a_node_to_the_graph() {
    let matrix = matrix_from_edges(3, &[(0, 1, 1.0), (2, 2, 5.0)]);
    assert_eq!(&*component_sizes(&matrix), &[2, 1]);
}

// ── Local efficiency ─────────────────────────────────────────────────────

/// The centre of a star has neighbours that connect only through the centre
/// itself — which is excluded — so its neighbourhood is entirely disconnected
/// and its local efficiency is zero.
#[test]
fn the_centre_of_a_star_has_no_local_efficiency() {
    let efficiency = local_efficiency(&star());
    assert_eq!(efficiency[0], 0.0);
}

/// In a triangle, each node's two neighbours are directly joined at unit weight,
/// so every neighbourhood is perfectly efficient.
#[test]
fn every_node_of_a_triangle_has_unit_local_efficiency() {
    for value in local_efficiency(&triangle()).iter() {
        assert!((value - 1.0).abs() < TOLERANCE, "got {value}");
    }
}

/// A node with fewer than two neighbours has no pairs in its neighbourhood, so
/// the measure takes its limiting value rather than dividing by zero.
#[test]
fn a_node_with_under_two_neighbours_has_zero_local_efficiency() {
    let efficiency = local_efficiency(&path_graph());
    assert_eq!(efficiency[0], 0.0);
    assert_eq!(efficiency[2], 0.0);
    // The middle node's neighbours are not joined to each other either.
    assert_eq!(efficiency[1], 0.0);
}

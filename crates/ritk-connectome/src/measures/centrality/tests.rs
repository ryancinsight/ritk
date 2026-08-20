use super::*;

use crate::tests::{matrix_from_edges, path_graph, star, triangle};

const TOLERANCE: f64 = 1.0e-9;

// ── Closed-form cases ────────────────────────────────────────────────────

/// In a path `0—1—2` the only pair needing an intermediary is `(0, 2)`, and node
/// 1 is on every shortest path between them. Normalised by the one available
/// intermediary pair, its centrality is exactly 1 and the endpoints' is 0.
#[test]
fn the_middle_of_a_path_carries_all_the_traffic() {
    let values = betweenness(&path_graph());
    assert!((values[1] - 1.0).abs() < TOLERANCE, "got {}", values[1]);
    assert!(values[0].abs() < TOLERANCE);
    assert!(values[2].abs() < TOLERANCE);
}

/// Every shortest path in a star passes through the hub, so it saturates the
/// measure while the rim carries nothing.
#[test]
fn the_hub_of_a_star_saturates_and_the_rim_carries_nothing() {
    let values = betweenness(&star());
    assert!((values[0] - 1.0).abs() < TOLERANCE, "got {}", values[0]);
    for (node, value) in values.iter().enumerate().skip(1) {
        assert!(value.abs() < TOLERANCE, "rim node {node} got {value}");
    }
}

/// In a complete graph every pair is directly joined, so no node is ever an
/// intermediary.
#[test]
fn a_complete_graph_has_no_intermediaries() {
    for value in betweenness(&triangle()).iter() {
        assert!(value.abs() < TOLERANCE, "got {value}");
    }
}

/// A graph too small to have an intermediary position gives zero rather than
/// dividing by zero.
#[test]
fn graphs_under_three_nodes_have_zero_betweenness() {
    assert_eq!(
        &*betweenness(&matrix_from_edges(2, &[(0, 1, 1.0)])),
        &[0.0, 0.0]
    );
    assert_eq!(&*betweenness(&matrix_from_edges(1, &[])), &[0.0]);
}

// ── Ties are shared, not awarded ─────────────────────────────────────────

/// Two equally short routes must split the credit. In a 4-cycle the pair of
/// opposite nodes has two shortest paths, one through each of the other two, so
/// each intermediary earns half a path.
///
/// This is the case that fails if equal-length routes are compared with a bare
/// `<`: floating-point addition would make one route look shorter, the other
/// would be dropped, and one arbitrary node would take all the credit.
#[test]
fn equally_short_routes_split_the_credit() {
    // 0—1—2—3—0 with unit weights.
    let cycle = matrix_from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]);
    let values = betweenness(&cycle);

    // By symmetry every node is equivalent, so every value must be equal.
    for value in values.iter() {
        assert!(
            (value - values[0]).abs() < TOLERANCE,
            "the cycle is symmetric, so all values must agree: {values:?}"
        );
    }
    // Each of the two opposite pairs has two shortest paths; each contributes
    // half a path to each of its two intermediaries, so each node accumulates
    // one half-path. Normalised by (n−1)(n−2) = 6, that is 1/6.
    assert!(
        (values[0] - 1.0 / 6.0).abs() < TOLERANCE,
        "got {}",
        values[0]
    );
}

/// The same tie, reached through *different* reciprocal weights so the two
/// route lengths are equal only after floating-point addition. An exact
/// comparison would call one shorter.
#[test]
fn routes_of_equal_length_reached_by_different_sums_are_still_tied() {
    // 0 to 3 via 1 costs 1/3 + 1/6; via 2 costs 1/6 + 1/3. Equal, different order.
    let matrix = matrix_from_edges(4, &[(0, 1, 3.0), (1, 3, 6.0), (0, 2, 6.0), (2, 3, 3.0)]);
    let values = betweenness(&matrix);
    assert!(
        (values[1] - values[2]).abs() < TOLERANCE,
        "the two intermediaries must tie: {values:?}"
    );
    assert!(values[1] > 0.0, "both must carry traffic: {values:?}");
}

// ── Weights matter ───────────────────────────────────────────────────────

/// Betweenness is a weighted measure, so strengthening a bypass must move
/// traffic off the node it bypasses.
#[test]
fn strengthening_a_bypass_takes_traffic_off_the_intermediary() {
    // 0—1—2 plus a direct 0—2. When the direct edge is light, the route through
    // node 1 is shorter and node 1 carries the traffic; when it is heavy, it does
    // not.
    let via_middle = matrix_from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.25)]);
    let direct = matrix_from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 4.0)]);

    assert!(
        betweenness(&via_middle)[1] > betweenness(&direct)[1],
        "a heavier direct edge must reduce the intermediary's betweenness"
    );
    assert!(betweenness(&direct)[1].abs() < TOLERANCE);
}

// ── Bounds and disconnection ─────────────────────────────────────────────

#[test]
fn values_stay_within_the_unit_range() {
    let matrix = matrix_from_edges(
        6,
        &[
            (0, 1, 2.0),
            (1, 2, 3.0),
            (2, 3, 1.0),
            (3, 4, 5.0),
            (4, 5, 2.0),
            (1, 4, 0.5),
        ],
    );
    for (node, value) in betweenness(&matrix).iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(value),
            "node {node} out of range: {value}"
        );
    }
}

/// A disconnected graph has no path between its parts, so no node earns credit
/// for a pair it cannot serve.
#[test]
fn a_disconnected_graph_credits_only_within_its_components() {
    // A path 0—1—2 and an isolated pair 3—4.
    let matrix = matrix_from_edges(5, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]);
    let values = betweenness(&matrix);

    assert!(
        values[1] > 0.0,
        "the path's middle still carries its own pair"
    );
    assert!(values[3].abs() < TOLERANCE);
    assert!(values[4].abs() < TOLERANCE);
}

#[test]
fn a_graph_with_no_edges_has_zero_betweenness() {
    for value in betweenness(&matrix_from_edges(4, &[])).iter() {
        assert_eq!(*value, 0.0);
    }
}

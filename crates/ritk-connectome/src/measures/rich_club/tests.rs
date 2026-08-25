use super::*;

use crate::tests::{matrix_from_edges, star, triangle};

const TOLERANCE: f64 = 1.0e-12;

fn level_for(levels: &[RichClubLevel], degree: usize) -> Option<&RichClubLevel> {
    levels.iter().find(|level| level.degree == degree)
}

// ── Structure of the returned curve ──────────────────────────────────────

/// Thresholds run from zero and stop where the club drops below two nodes,
/// because a single node has no pairs and its coefficient would be undefined
/// rather than zero.
#[test]
fn levels_stop_when_the_club_falls_below_two_members() {
    let levels = rich_club(&star());
    // The star has one node of degree 3 and three of degree 1. At threshold 0
    // all four qualify; at 1 only the hub does, so that level is not reported.
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].degree, 0);
    assert_eq!(levels[0].node_count, 4);
}

#[test]
fn a_graph_with_no_edges_yields_no_levels() {
    assert!(rich_club(&matrix_from_edges(4, &[])).is_empty());
}

#[test]
fn an_empty_graph_yields_no_levels() {
    assert!(rich_club(&matrix_from_edges(0, &[])).is_empty());
}

// ── Closed-form values ───────────────────────────────────────────────────

/// A complete graph is its own rich club at every threshold: all pairs among the
/// qualifying nodes are present, so `Φ = 1`.
#[test]
fn a_complete_graph_has_a_unit_coefficient() {
    for level in rich_club(&triangle()) {
        assert!(
            (level.coefficient - 1.0).abs() < TOLERANCE,
            "threshold {}: got {}",
            level.degree,
            level.coefficient
        );
    }
}

/// A star's rim nodes are not connected to each other, so at threshold 0 only
/// the three hub edges exist among the six possible pairs: `Φ = 3/6`.
#[test]
fn the_star_matches_its_closed_form_coefficient() {
    let levels = rich_club(&star());
    let level = level_for(&levels, 0).expect("threshold 0");
    assert_eq!(level.edge_count, 3);
    assert!(
        (level.coefficient - 0.5).abs() < TOLERANCE,
        "got {}",
        level.coefficient
    );
}

/// The measure's purpose: a graph where the high-degree nodes are wired to each
/// other must show `Φ` rising with the threshold.
#[test]
fn a_planted_hub_core_shows_a_rising_coefficient() {
    // Four hubs forming a complete core, each with two pendant leaves. The
    // leaves drag the coefficient down at low thresholds and drop out as the
    // threshold rises, leaving the complete core.
    let mut edges = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            edges.push((i, j, 1.0));
        }
    }
    for hub in 0..4 {
        edges.push((hub, 4 + hub * 2, 1.0));
        edges.push((hub, 5 + hub * 2, 1.0));
    }
    let matrix = matrix_from_edges(12, &edges);

    let levels = rich_club(&matrix);
    let low = level_for(&levels, 0).expect("threshold 0");
    let high = levels.last().expect("at least one level");

    assert!(
        high.coefficient > low.coefficient,
        "the coefficient must rise with the threshold: {} at 0, {} at {}",
        low.coefficient,
        high.coefficient,
        high.degree
    );
    // At the highest reported threshold only the four hubs remain, and they are
    // completely interconnected.
    assert_eq!(high.node_count, 4);
    assert!((high.coefficient - 1.0).abs() < TOLERANCE);
}

/// Membership is by degree alone, so raising the threshold can only shrink the
/// club. A non-monotone membership would mean the degree computation disagreed
/// with itself between levels.
#[test]
fn club_membership_shrinks_monotonically_with_the_threshold() {
    let matrix = matrix_from_edges(
        6,
        &[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 4, 1.0),
            (3, 5, 1.0),
        ],
    );
    let levels = rich_club(&matrix);
    for pair in levels.windows(2) {
        assert!(
            pair[1].node_count <= pair[0].node_count,
            "club grew from threshold {} ({}) to {} ({})",
            pair[0].degree,
            pair[0].node_count,
            pair[1].degree,
            pair[1].node_count
        );
    }
}

// ── The weighted companion ───────────────────────────────────────────────

/// A club can be topologically complete while its edges are individually weak,
/// which the coefficient alone cannot show. The mean weight is what separates
/// the two cases.
#[test]
fn mean_weight_separates_a_strong_club_from_a_weak_one_of_equal_topology() {
    let weak = matrix_from_edges(3, &[(0, 1, 0.1), (1, 2, 0.1), (0, 2, 0.1)]);
    let strong = matrix_from_edges(3, &[(0, 1, 50.0), (1, 2, 50.0), (0, 2, 50.0)]);

    let weak_level = rich_club(&weak)[0];
    let strong_level = rich_club(&strong)[0];

    assert!((weak_level.coefficient - strong_level.coefficient).abs() < TOLERANCE);
    assert!((weak_level.mean_weight - 0.1).abs() < TOLERANCE);
    assert!((strong_level.mean_weight - 50.0).abs() < TOLERANCE);
}

#[test]
fn a_club_with_no_internal_edges_has_zero_mean_weight() {
    // Two nodes of equal degree that connect only through a third.
    let matrix = matrix_from_edges(3, &[(0, 2, 1.0), (1, 2, 1.0)]);
    let levels = rich_club(&matrix);
    let level = level_for(&levels, 0).expect("threshold 0");
    // Node 2 has degree 2, nodes 0 and 1 degree 1; at threshold 0 all qualify.
    assert_eq!(level.edge_count, 2);
    assert!(level.mean_weight > 0.0);

    // At threshold 1 only node 2 remains, so no level is reported.
    assert!(level_for(&levels, 1).is_none());
}

#[test]
fn coefficients_stay_within_the_unit_range() {
    let matrix = matrix_from_edges(
        7,
        &[
            (0, 1, 3.0),
            (0, 2, 1.0),
            (1, 2, 4.0),
            (2, 3, 2.0),
            (3, 4, 1.0),
            (4, 5, 5.0),
            (5, 6, 1.0),
            (0, 6, 2.0),
        ],
    );
    for level in rich_club(&matrix) {
        assert!(
            (0.0..=1.0).contains(&level.coefficient),
            "threshold {} out of range: {}",
            level.degree,
            level.coefficient
        );
    }
}

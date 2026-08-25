use super::*;

use crate::tests::{matrix_from_edges, triangle, two_modules};

const TOLERANCE: f64 = 1.0e-12;

/// Community indices are arbitrary labels, so two partitions are the same
/// partition when they group the same nodes, whatever they call the groups.
fn same_partition(left: &[usize], right: &[usize]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().enumerate().all(|(i, _)| {
        right
            .iter()
            .enumerate()
            .all(|(j, _)| (left[i] == left[j]) == (right[i] == right[j]))
    })
}

// ── Modularity ───────────────────────────────────────────────────────────

/// Putting every node in one community leaves nothing outside it, so the
/// observed and expected weights coincide and `Q` is exactly zero. This is the
/// baseline the measure is defined against.
#[test]
fn one_community_containing_everything_has_zero_modularity() {
    let matrix = two_modules();
    let all_together = vec![0; matrix.region_count()];
    assert!(modularity(&matrix, &all_together).abs() < TOLERANCE);
}

/// A partition that cuts the strong internal edges and keeps only the bridge is
/// worse than chance, so `Q` must be negative. A measure that could not go
/// negative would not be distinguishing good partitions from bad ones.
#[test]
fn a_partition_that_splits_the_modules_is_worse_than_chance() {
    let matrix = two_modules();
    // Split each triangle down the middle.
    let scrambled = vec![0, 1, 0, 1, 0, 1];
    assert!(
        modularity(&matrix, &scrambled) < 0.0,
        "got {}",
        modularity(&matrix, &scrambled)
    );
}

/// The planted partition must score higher than any of the alternatives, which
/// is the property that makes maximising `Q` a sensible objective at all.
#[test]
fn the_planted_partition_scores_highest() {
    let matrix = two_modules();
    let planted = vec![0, 0, 0, 1, 1, 1];
    let planted_quality = modularity(&matrix, &planted);

    for alternative in [
        vec![0, 0, 0, 0, 0, 0],
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 0, 1, 1, 0, 0],
        vec![0, 1, 0, 1, 0, 1],
        vec![0, 0, 0, 0, 1, 1],
    ] {
        assert!(
            planted_quality > modularity(&matrix, &alternative),
            "the planted partition ({planted_quality}) must beat {alternative:?} \
             ({})",
            modularity(&matrix, &alternative)
        );
    }
}

/// Modularity is defined against a null model of the same total weight, so
/// scaling every edge leaves it unchanged.
#[test]
fn modularity_is_invariant_under_uniform_scaling() {
    let base = matrix_from_edges(4, &[(0, 1, 2.0), (2, 3, 2.0), (1, 2, 0.5)]);
    let scaled = matrix_from_edges(4, &[(0, 1, 20.0), (2, 3, 20.0), (1, 2, 5.0)]);
    let partition = vec![0, 0, 1, 1];
    assert!((modularity(&base, &partition) - modularity(&scaled, &partition)).abs() < TOLERANCE);
}

/// Community indices carry no meaning, so renaming them cannot change the score.
#[test]
fn modularity_ignores_how_the_communities_are_numbered() {
    let matrix = two_modules();
    assert!(
        (modularity(&matrix, &[0, 0, 0, 1, 1, 1]) - modularity(&matrix, &[7, 7, 7, 2, 2, 2])).abs()
            < TOLERANCE
    );
}

#[test]
fn a_graph_with_no_edges_has_zero_modularity() {
    let matrix = matrix_from_edges(3, &[]);
    assert_eq!(modularity(&matrix, &[0, 1, 2]), 0.0);
}

#[test]
#[should_panic(expected = "a partition must assign every node")]
fn a_partition_that_misses_a_node_is_rejected() {
    let _ = modularity(&two_modules(), &[0, 0, 0]);
}

// ── Louvain ──────────────────────────────────────────────────────────────

/// The planted-partition oracle: on a graph whose module structure is
/// unambiguous, the detector must recover exactly that structure. This is what
/// separates a working implementation from one that returns a plausible-looking
/// partition.
#[test]
fn louvain_recovers_an_unambiguous_planted_partition() {
    let matrix = two_modules();
    let found = louvain(&matrix);

    assert_eq!(found.count(), 2, "assignment {:?}", found.assignment());
    assert!(
        same_partition(found.assignment(), &[0, 0, 0, 1, 1, 1]),
        "expected the two triangles, got {:?}",
        found.assignment()
    );
    assert!(
        (found.modularity() - modularity(&matrix, found.assignment())).abs() < TOLERANCE,
        "the reported modularity must be the partition's own"
    );
}

/// Four cliques joined by single light edges — a harder planted partition that a
/// single pass of node moves would not resolve, so it exercises the aggregation
/// phase rather than only the local one.
#[test]
fn louvain_recovers_four_planted_modules() {
    let mut edges = Vec::new();
    for module in 0..4 {
        let base = module * 4;
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((base + i, base + j, 20.0));
            }
        }
    }
    // A ring of faint bridges between consecutive modules.
    for module in 0..4 {
        edges.push((module * 4, ((module + 1) % 4) * 4 + 1, 0.5));
    }
    let matrix = matrix_from_edges(16, &edges);

    let found = louvain(&matrix);
    let expected: Vec<usize> = (0..16).map(|node| node / 4).collect();
    assert_eq!(found.count(), 4, "assignment {:?}", found.assignment());
    assert!(
        same_partition(found.assignment(), &expected),
        "expected four cliques, got {:?}",
        found.assignment()
    );
}

/// The result must be a function of the matrix alone. A randomised node order
/// would make this fail intermittently, which is exactly the failure mode the
/// deterministic ordering exists to prevent.
#[test]
fn louvain_is_deterministic() {
    let matrix = two_modules();
    let first = louvain(&matrix);
    for _ in 0..8 {
        assert_eq!(louvain(&matrix), first);
    }
}

/// A graph with no structure to find leaves every node alone, at zero
/// modularity — the honest answer rather than an invented grouping.
#[test]
fn a_graph_with_no_edges_leaves_every_node_alone() {
    let matrix = matrix_from_edges(4, &[]);
    let found = louvain(&matrix);

    assert_eq!(found.count(), 4);
    assert_eq!(found.modularity(), 0.0);
}

/// A complete graph has no sub-structure, so any split is worse than chance and
/// the detector must not manufacture one.
#[test]
fn a_complete_graph_is_not_split() {
    let found = louvain(&triangle());
    assert_eq!(found.count(), 1, "assignment {:?}", found.assignment());
    assert!(found.modularity().abs() < TOLERANCE);
}

/// Community indices are compacted, so a partition into `c` groups uses exactly
/// `0..c` — the invariant [`Communities::members`] indexes on.
#[test]
fn community_indices_are_compact_and_members_partition_the_nodes() {
    let matrix = two_modules();
    let found = louvain(&matrix);

    for community in found.assignment() {
        assert!(*community < found.count(), "index {community} out of range");
    }
    let groups = found.members();
    assert_eq!(groups.len(), found.count());
    let total: usize = groups.iter().map(Vec::len).sum();
    assert_eq!(total, matrix.region_count());
}

/// Detected communities must actually be good ones — better than lumping
/// everything together and better than isolating every node.
#[test]
fn the_detected_partition_beats_the_trivial_ones() {
    let matrix = two_modules();
    let found = louvain(&matrix);
    let n = matrix.region_count();

    assert!(found.modularity() > modularity(&matrix, &vec![0; n]));
    assert!(found.modularity() > modularity(&matrix, &(0..n).collect::<Vec<_>>()));
}

#[test]
fn an_empty_graph_has_no_communities() {
    let found = louvain(&matrix_from_edges(0, &[]));
    assert_eq!(found.count(), 0);
    assert_eq!(found.modularity(), 0.0);
    assert!(found.assignment().is_empty());
}

use super::*;

use crate::tests::matrix_from_edges;

/// Nodes 0..HUBS are hubs; the rest are degree-one leaves.
const HUBS: usize = 5;
/// Leaves attached to each hub.
const LEAVES_PER_HUB: usize = 4;
/// Detached leaf pairs, which are what give the null model room to work.
const SPARE_PAIRS: usize = 20;

/// A graph whose hubs are wired to each other by *choice* rather than by
/// arithmetic.
///
/// The distinction matters more than it looks. A degree sequence can force the
/// club's density: four hubs of degree five with eight degree-one leaves leaves
/// exactly six slots for hub-hub edges, which is every pair, so *every* graph
/// with that sequence has a complete club and there is no excess for any null
/// model to find. A fixture like that tests nothing — it reports a ratio of one
/// because one is correct.
///
/// Here the leaves outnumber what the hubs can absorb, and the surplus is
/// carried by detached leaf pairs. A swap can therefore trade a hub-hub edge for
/// two hub-leaf edges, so a random graph with this degree sequence usually has a
/// sparser club than this one does — and the ratio has something to measure.
fn planted_rich_club() -> ConnectivityMatrix {
    let mut edges = Vec::new();
    for i in 0..HUBS {
        for j in (i + 1)..HUBS {
            edges.push((i, j, 1.0));
        }
    }
    let mut next = HUBS;
    for hub in 0..HUBS {
        for _ in 0..LEAVES_PER_HUB {
            edges.push((hub, next, 1.0));
            next += 1;
        }
    }
    for _ in 0..SPARE_PAIRS {
        edges.push((next, next + 1, 1.0));
        next += 2;
    }
    matrix_from_edges(next, &edges)
}

/// The same degree sequence, with the hubs wired only to leaves.
///
/// The negative control: the highest-degree nodes in the graph, with no
/// connection among them at all. A measure that read organisation here would
/// read it everywhere.
fn hubs_without_a_club() -> ConnectivityMatrix {
    let mut edges = Vec::new();
    let mut next = HUBS;
    // Each hub carries the degree it had above, all of it spent on leaves.
    for hub in 0..HUBS {
        for _ in 0..(HUBS - 1 + LEAVES_PER_HUB) {
            edges.push((hub, next, 1.0));
            next += 1;
        }
    }
    for _ in 0..SPARE_PAIRS {
        edges.push((next, next + 1, 1.0));
        next += 2;
    }
    matrix_from_edges(next, &edges)
}

/// A ring, where every node has degree two and no node is a hub at all.
fn ring(nodes: usize) -> ConnectivityMatrix {
    let edges: Vec<(usize, usize, f64)> = (0..nodes)
        .map(|node| (node, (node + 1) % nodes, 1.0))
        .collect();
    matrix_from_edges(nodes, &edges)
}

fn config() -> RandomisationConfig {
    RandomisationConfig::new(64, 0x5eed_0b17)
}

// ── The invariant the whole method rests on ──────────────────────────────

/// Every sample must preserve every node's degree exactly.
///
/// This is what makes the ratio a statement about *wiring* rather than about
/// degree: if the null model changed the degree sequence, the club would be a
/// different set of nodes in every sample and the comparison would be between
/// two unrelated quantities. A swap that folded an edge onto a node or
/// duplicated an existing one would break it, which is why both are rejected.
#[test]
fn randomisation_preserves_every_degree() {
    let matrix = planted_rich_club();
    let n = matrix.region_count();
    let expected: Vec<usize> = (0..n).map(|node| matrix.degree_at(node)).collect();

    let mut rng = StdRng::seed_from_u64(7);
    for sample_index in 0..16 {
        let sample = randomise(&matrix, 10, &mut rng);
        for (node, wanted) in expected.iter().enumerate() {
            let degree = (0..n)
                .filter(|other| *other != node && sample.adjacency[node * n + other])
                .count();
            assert_eq!(
                degree, *wanted,
                "sample {sample_index} changed the degree of node {node}"
            );
        }
    }
}

/// The rewiring must never leave a self-loop or a duplicate edge, since either
/// would put the sample outside the family of simple graphs the observed value
/// is being compared against.
#[test]
fn randomisation_keeps_the_graph_simple() {
    let matrix = planted_rich_club();
    let n = matrix.region_count();
    let mut rng = StdRng::seed_from_u64(11);

    for _ in 0..16 {
        let sample = randomise(&matrix, 10, &mut rng);
        for node in 0..n {
            assert!(
                !sample.adjacency[node * n + node],
                "a self-loop appeared at node {node}"
            );
        }
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    sample.adjacency[i * n + j],
                    sample.adjacency[j * n + i],
                    "the adjacency lost its symmetry at ({i}, {j})"
                );
            }
        }
    }
}

/// The edge count is fixed by the degree sequence, so it too must survive.
#[test]
fn randomisation_preserves_the_edge_count() {
    let matrix = planted_rich_club();
    let n = matrix.region_count();
    let expected = matrix.edge_count();

    let mut rng = StdRng::seed_from_u64(13);
    let sample = randomise(&matrix, 10, &mut rng);
    let edges = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .filter(|(i, j)| sample.adjacency[i * n + j])
        .count();
    assert_eq!(edges, expected);
}

// ── What the ratio says ──────────────────────────────────────────────────

/// The measure's reason to exist: a planted club must show a ratio above one
/// where the raw curve alone could not distinguish it from degree.
#[test]
fn a_planted_club_exceeds_its_null_model() {
    let matrix = planted_rich_club();
    let (levels, report) = normalised_rich_club(&matrix, config()).expect("valid configuration");

    assert!(!levels.is_empty());
    assert!(
        report.acceptance() > 0.1,
        "the ensemble must actually rewire: acceptance {}",
        report.acceptance()
    );

    let top = levels.last().expect("at least one level");
    let ratio = top.ratio.expect("the ensemble found club edges");
    assert!(
        ratio > 1.0,
        "a planted club must exceed chance at the highest threshold: \
         observed {:.3}, random mean {:.3}, ratio {ratio:.3}",
        top.observed.coefficient,
        top.random_mean
    );
}

/// A ring has no hubs, so no threshold isolates a club and the ratio must not
/// claim one. This is the negative control: a measure that reported organisation
/// here would report it everywhere.
#[test]
fn a_ring_shows_no_rich_club() {
    let matrix = ring(12);
    let (levels, _) = normalised_rich_club(&matrix, config()).expect("valid configuration");

    for level in &levels {
        // Every node has the same degree, so the only club is the whole graph
        // and the null model reproduces it exactly.
        if let Some(ratio) = level.ratio {
            assert!(
                (ratio - 1.0).abs() < 1.0e-9,
                "a degree-regular graph is its own null model: threshold {} gave {ratio}",
                level.observed.degree
            );
        }
    }
}

/// The negative control: the highest-degree nodes, wired only to leaves.
///
/// The same degree sequence as the planted case, so the club membership at each
/// threshold is identical — only the wiring differs. A measure that reported
/// organisation here would be reporting degree, which is exactly what the
/// normalisation exists to divide out.
#[test]
fn hubs_wired_only_to_leaves_show_no_club() {
    let matrix = hubs_without_a_club();
    let (levels, report) = normalised_rich_club(&matrix, config()).expect("valid configuration");

    assert!(
        report.acceptance() > 0.1,
        "the ensemble must rewire for the comparison to mean anything: {}",
        report.acceptance()
    );
    let top = levels.last().expect("at least one level");
    assert!(
        top.observed.coefficient.abs() < f64::EPSILON,
        "the fixture must have no hub-hub edge, got {}",
        top.observed.coefficient
    );
    if let Some(ratio) = top.ratio {
        assert!(
            ratio < 1.0,
            "no hub-hub edge cannot exceed chance: ratio {ratio:.3}"
        );
    }
}

/// A degree sequence can leave the club no freedom at all, and the ratio must
/// then report one rather than inventing an excess.
///
/// Four hubs of degree five with eight degree-one leaves has exactly six slots
/// for hub-hub edges — every pair — so every graph with that sequence has a
/// complete club. The honest answer is that there is no excess, and this pins
/// it, because a naive implementation that compared against a *different*
/// degree sequence would report organisation here.
#[test]
fn a_forced_club_reports_no_excess() {
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

    let (levels, _) = normalised_rich_club(&matrix, config()).expect("valid configuration");
    let top = levels.last().expect("at least one level");
    assert!(
        (top.ratio.expect("the club has edges") - 1.0).abs() < 1.0e-9,
        "a forced club has no excess to report: ratio {:?}",
        top.ratio
    );
}

// ── Reproducibility and reporting ────────────────────────────────────────

/// A seeded ensemble must give the same answer twice, or a published ratio
/// cannot be reproduced from the paper that reports it.
#[test]
fn the_ensemble_is_reproducible_from_its_seed() {
    let matrix = planted_rich_club();
    let first = normalised_rich_club(&matrix, config()).expect("valid configuration");
    for _ in 0..4 {
        let again = normalised_rich_club(&matrix, config()).expect("valid configuration");
        assert_eq!(first.0, again.0);
        assert_eq!(first.1, again.1);
    }
}

/// A different seed must actually explore a different ensemble, or the seed is
/// decoration.
#[test]
fn a_different_seed_gives_a_different_ensemble() {
    let matrix = planted_rich_club();
    let (first, _) = normalised_rich_club(&matrix, RandomisationConfig::new(16, 1))
        .expect("valid configuration");
    let (second, _) = normalised_rich_club(&matrix, RandomisationConfig::new(16, 2))
        .expect("valid configuration");
    assert!(
        first
            .iter()
            .zip(&second)
            .any(|(left, right)| { (left.random_mean - right.random_mean).abs() > f64::EPSILON }),
        "two seeds produced identical ensemble means"
    );
}

/// The observed level is carried through untouched, so a caller reading the
/// normalised result never has to recompute the raw one.
#[test]
fn the_observed_level_is_carried_through_unchanged() {
    let matrix = planted_rich_club();
    let raw = rich_club(&matrix);
    let (levels, _) = normalised_rich_club(&matrix, config()).expect("valid configuration");

    assert_eq!(levels.len(), raw.len());
    for (normalised, observed) in levels.iter().zip(&raw) {
        assert_eq!(normalised.observed, *observed);
    }
}

/// The acceptance fraction reports how much rewiring actually happened, which a
/// caller needs before trusting a ratio near one.
#[test]
fn the_report_counts_the_swaps_it_attempted() {
    let matrix = planted_rich_club();
    let (_, report) =
        normalised_rich_club(&matrix, RandomisationConfig::new(8, 3)).expect("valid configuration");

    assert!(report.attempted > 0);
    assert!(report.accepted <= report.attempted);
    assert!((0.0..=1.0).contains(&report.acceptance()));
}

/// A complete graph cannot be rewired at all — every swap would duplicate an
/// edge — so acceptance is zero and the caller is told rather than handed a
/// ratio of one as though it meant something.
#[test]
fn a_complete_graph_accepts_no_swaps() {
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            edges.push((i, j, 1.0));
        }
    }
    let matrix = matrix_from_edges(5, &edges);

    let (_, report) = normalised_rich_club(&matrix, config()).expect("valid configuration");
    assert_eq!(
        report.accepted, 0,
        "a complete graph has no swap that keeps it simple"
    );
    assert!(report.attempted > 0);
}

// ── Configuration is rejected rather than defaulted ──────────────────────

#[test]
fn an_empty_ensemble_is_rejected() {
    let matrix = planted_rich_club();
    let error = normalised_rich_club(
        &matrix,
        RandomisationConfig {
            ensemble_size: 0,
            swaps_per_edge: 10,
            seed: 0,
        },
    )
    .expect_err("the rejected input must yield the typed error");
    assert!(matches!(
        error,
        ConnectomeError::InvalidRandomisation { .. }
    ));
}

/// Zero swaps would hand back the graph as its own null model, and a ratio of
/// exactly one everywhere — a confident answer that means nothing.
#[test]
fn a_zero_swap_count_is_rejected() {
    let matrix = planted_rich_club();
    let error = normalised_rich_club(
        &matrix,
        RandomisationConfig {
            ensemble_size: 8,
            swaps_per_edge: 0,
            seed: 0,
        },
    )
    .expect_err("the rejected input must yield the typed error");
    assert!(matches!(
        error,
        ConnectomeError::InvalidRandomisation { .. }
    ));
}

#[test]
fn a_graph_with_no_levels_yields_no_normalised_levels() {
    let matrix = matrix_from_edges(4, &[]);
    let (levels, report) = normalised_rich_club(&matrix, config()).expect("valid configuration");
    assert!(levels.is_empty());
    assert_eq!(report.attempted, 0);
}

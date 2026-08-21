//! The degree-preserving null model the rich-club ratio is measured against.
//!
//! See the [parent module](super) for why the raw curve is not the measure and
//! why this model's parameters have no library defaults.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::{RichClubLevel, rich_club};
use crate::{ConnectivityMatrix, ConnectomeError};

/// How the null-model ensemble is generated.
///
/// Every field is a study-design choice with no defensible library default, so
/// none of them has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomisationConfig {
    /// Randomised graphs to average over.
    ///
    /// The mean's precision improves as the square root of this, so a handful
    /// gives a shape and a hundred gives a number worth quoting.
    pub ensemble_size: usize,
    /// Swap attempts per edge, per sample.
    ///
    /// Too few and the sample still resembles the graph it came from, which
    /// biases the ratio toward one. Around ten is the usual working figure.
    pub swaps_per_edge: usize,
    /// Seed for the swap sequence.
    ///
    /// Fixed rather than drawn, so a reported ratio can be reproduced: two runs
    /// with the same seed and configuration give the same number.
    pub seed: u64,
}

impl RandomisationConfig {
    /// An ensemble of the given size at ten swaps per edge.
    #[must_use]
    pub const fn new(ensemble_size: usize, seed: u64) -> Self {
        Self {
            ensemble_size,
            swaps_per_edge: 10,
            seed,
        }
    }
}

/// A rich-club level with its null-model comparison.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalisedRichClubLevel {
    /// The level measured on the graph itself.
    pub observed: RichClubLevel,
    /// Mean `Φ(k)` over the randomised ensemble.
    pub random_mean: f64,
    /// Standard deviation of `Φ(k)` over the ensemble.
    ///
    /// The scale the excess should be read against: a ratio of 1.1 means little
    /// when the ensemble itself spreads by 0.2.
    pub random_deviation: f64,
    /// `Φ(k) / ⟨Φ_random(k)⟩`, or `None` when the ensemble produced no club
    /// edges at this threshold and the ratio has no denominator.
    ///
    /// Above one is the evidence of rich-club organisation. At or below one
    /// there is none, however steeply the raw curve rises.
    pub ratio: Option<f64>,
}

/// How well the ensemble managed to rewire.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RandomisationReport {
    /// Swap attempts made across the whole ensemble.
    pub attempted: usize,
    /// Attempts that were applied rather than rejected.
    pub accepted: usize,
}

impl RandomisationReport {
    /// Fraction of attempted swaps that were applied, in `[0, 1]`.
    ///
    /// A low fraction means the graph is too dense or too constrained to rewire
    /// freely, so the ensemble has not moved far from the graph it started at
    /// and the ratio is closer to one than the anatomy warrants. Read it before
    /// reading the ratio.
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.attempted == 0 {
            return 1.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "swap counts stay far below f64's exact-integer range"
        )]
        let ratio = self.accepted as f64 / self.attempted as f64;
        ratio
    }
}

/// The rich-club curve with a degree-preserving null model.
///
/// Returns one entry per threshold [`rich_club`] reports, alongside how much of
/// the rewiring the ensemble actually achieved.
///
/// # Errors
///
/// [`ConnectomeError::InvalidRandomisation`] when the ensemble size is zero —
/// there is nothing to average — or when the swap count is zero, which would
/// hand back the original graph as its own null model and a ratio of exactly
/// one everywhere.
pub fn normalised_rich_club(
    matrix: &ConnectivityMatrix,
    config: RandomisationConfig,
) -> Result<(Vec<NormalisedRichClubLevel>, RandomisationReport), ConnectomeError> {
    if config.ensemble_size == 0 || config.swaps_per_edge == 0 {
        return Err(ConnectomeError::InvalidRandomisation {
            ensemble_size: config.ensemble_size,
            swaps_per_edge: config.swaps_per_edge,
        });
    }

    let observed = rich_club(matrix);
    let report = RandomisationReport {
        attempted: 0,
        accepted: 0,
    };
    if observed.is_empty() {
        return Ok((Vec::new(), report));
    }

    // Double-edge swaps leave every degree exactly as it was, so club
    // membership at each threshold is the same in every sample as in the graph
    // itself. Deriving it once and reusing it is what makes the ratio a
    // statement about wiring rather than about degree.
    let degrees: Vec<usize> = (0..matrix.region_count())
        .map(|node| matrix.degree_at(node))
        .collect();

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(config.ensemble_size); observed.len()];
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut report = report;

    for _ in 0..config.ensemble_size {
        let sample = randomise(matrix, config.swaps_per_edge, &mut rng);
        report.attempted += sample.attempted;
        report.accepted += sample.accepted;
        for (slot, level) in samples.iter_mut().zip(&observed) {
            slot.push(coefficient_over(&sample.adjacency, &degrees, level.degree));
        }
    }

    let levels = observed
        .into_iter()
        .zip(&samples)
        .map(|(level, sample)| {
            let mean = mean_of(sample);
            NormalisedRichClubLevel {
                observed: level,
                random_mean: mean,
                random_deviation: deviation_of(sample, mean),
                ratio: (mean > 0.0).then_some(level.coefficient / mean),
            }
        })
        .collect();

    Ok((levels, report))
}

/// One rewired topology and how it was reached.
struct Sample {
    /// Dense symmetric adjacency over the graph's nodes.
    adjacency: Vec<bool>,
    attempted: usize,
    accepted: usize,
}

/// A degree-preserving rewiring of the graph's topology.
///
/// Only the topology is rewired. The quantity being normalised counts edges
/// among the club rather than weighing them, so the weights play no part and
/// are not carried.
fn randomise(matrix: &ConnectivityMatrix, swaps_per_edge: usize, rng: &mut StdRng) -> Sample {
    let n = matrix.region_count();
    let mut adjacency = vec![false; n * n];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if matrix.weight_at(i, j) > 0.0 {
                adjacency[i * n + j] = true;
                adjacency[j * n + i] = true;
                edges.push((i, j));
            }
        }
    }
    if edges.len() < 2 {
        // A lone edge has nothing to swap against, so the graph is its own
        // randomisation. Reported as no attempts rather than as a failure.
        return Sample {
            adjacency,
            attempted: 0,
            accepted: 0,
        };
    }

    let attempted = swaps_per_edge * edges.len();
    let mut accepted = 0_usize;
    for _ in 0..attempted {
        let first = rng.random_range(0..edges.len());
        let second = rng.random_range(0..edges.len());
        if first == second {
            continue;
        }
        let (a, mut b) = edges[first];
        let (c, mut d) = edges[second];
        // Either pairing of the four endpoints is a valid swap, and choosing
        // between them at random is what lets the chain reach graphs a fixed
        // orientation never would.
        if rng.random::<bool>() {
            std::mem::swap(&mut b, &mut d);
        }
        // A swap folding an edge onto a node would change that node's degree;
        // one duplicating an existing edge would make the graph non-simple.
        // Both are rejected rather than applied.
        if a == d || c == b || adjacency[a * n + d] || adjacency[c * n + b] {
            continue;
        }

        adjacency[a * n + b] = false;
        adjacency[b * n + a] = false;
        adjacency[c * n + d] = false;
        adjacency[d * n + c] = false;
        adjacency[a * n + d] = true;
        adjacency[d * n + a] = true;
        adjacency[c * n + b] = true;
        adjacency[b * n + c] = true;
        edges[first] = (a.min(d), a.max(d));
        edges[second] = (c.min(b), c.max(b));
        accepted += 1;
    }

    Sample {
        adjacency,
        attempted,
        accepted,
    }
}

/// `Φ(k)` over a rewired adjacency, with membership taken from the original
/// degree sequence.
fn coefficient_over(adjacency: &[bool], degrees: &[usize], threshold: usize) -> f64 {
    let n = degrees.len();
    let club: Vec<usize> = degrees
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree > threshold)
        .map(|(node, _)| node)
        .collect();
    if club.len() < 2 {
        return 0.0;
    }
    let mut edges = 0_usize;
    for (position, first) in club.iter().enumerate() {
        for second in &club[position + 1..] {
            if adjacency[first * n + second] {
                edges += 1;
            }
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "node and edge counts stay far below f64's exact-integer range"
    )]
    let ratio = edges as f64 / (club.len() * (club.len() - 1) / 2) as f64;
    ratio
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "ensemble sizes stay far below f64's exact-integer range"
    )]
    let count = values.len() as f64;
    values.iter().sum::<f64>() / count
}

fn deviation_of(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "ensemble sizes stay far below f64's exact-integer range"
    )]
    let count = values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count;
    variance.sqrt()
}

#[cfg(test)]
mod tests;

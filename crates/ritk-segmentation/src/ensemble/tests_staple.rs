//! Tests for the STAPLE ensemble segmentation algorithm.
//!
//! All assertions are value-semantic: computed W, p, q values are verified
//! against analytically derived bounds from the STAPLE EM convergence properties.
//!
//! Analytical basis:
//! - With K identical perfect raters, the E-step log-likelihood ratio for a
//!   positive voxel is K·log(p/(1−q)) → +∞ as p,q → 1, driving W → 1.
//! - The M-step fixed point for K identical raters on a clean binary mask is
//!   p* = q* = 1−ε (clamped), which is a stable attractor reached in O(1) iterations.

use super::*;
use crate::ensemble::staple::EPS as STAPLE_TOL;

// ── Test 1: Perfect raters converge to the true segmentation ────────────────

#[test]
fn test_perfect_raters_converge_to_truth() {
    let n = 100usize;
    // Three identical masks: first half positive, second half negative.
    let mask: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
    let raters = vec![mask.clone(), mask.clone(), mask.clone()];

    let result = staple(&raters, 100, STAPLE_TOL);

    for (i, &w) in result.probabilistic_truth.iter().enumerate() {
        if i < n / 2 {
            assert!(
                w > 0.95,
                "voxel {} (positive region): W = {:.6}, expected > 0.95",
                i,
                w
            );
        } else {
            assert!(
                w < 0.05,
                "voxel {} (negative region): W = {:.6}, expected < 0.05",
                i,
                w
            );
        }
    }

    // Sensitivity and specificity must converge near 1.0 (clamped to 1−ε).
    for (ki, (&p, &q)) in result
        .sensitivity
        .iter()
        .zip(result.specificity.iter())
        .enumerate()
    {
        assert!(
            p > 0.99,
            "rater {}: sensitivity p = {:.8}, expected > 0.99",
            ki,
            p
        );
        assert!(
            q > 0.99,
            "rater {}: specificity q = {:.8}, expected > 0.99",
            ki,
            q
        );
    }
}

// ── Test 2: Single rater – W closely tracks the rater signal ────────────────

#[test]
fn test_single_rater_returns_itself() {
    let n = 20usize;
    // Alternating positive/negative: 1, 0, 1, 0, ...
    let mask: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let raters = vec![mask.clone()];

    let result = staple(&raters, 100, STAPLE_TOL);

    for (i, (&rater_val, &w)) in mask
        .iter()
        .zip(result.probabilistic_truth.iter())
        .enumerate()
    {
        if rater_val > 0.5 {
            assert!(
                w >= 0.9,
                "voxel {} (rater=1): W = {:.6}, expected ≥ 0.9",
                i,
                w
            );
        } else {
            assert!(
                w <= 0.1,
                "voxel {} (rater=0): W = {:.6}, expected ≤ 0.1",
                i,
                w
            );
        }
    }
}

// ── Test 3: Majority-vote approximation ─────────────────────────────────────

#[test]
fn test_majority_vote_approximation() {
    // K=5, N=4.
    // Voxel 0: 4/5 raters say 1  → W[0] > 0.7 (analytically: log-LR >> 0).
    // Voxel 1: 1/5 raters say 1  → W[1] < 0.3 (analytically: log-LR << 0).
    let raters = vec![
        vec![1.0_f32, 1.0, 0.0, 0.0],
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![0.0_f32, 0.0, 0.0, 0.0],
    ];

    let result = staple(&raters, 100, STAPLE_TOL);
    let w = &result.probabilistic_truth;

    assert!(
        w[0] > 0.7,
        "voxel 0 (4/5 agree positive): W = {:.6}, expected > 0.7",
        w[0]
    );
    assert!(
        w[1] < 0.3,
        "voxel 1 (1/5 agree positive): W = {:.6}, expected < 0.3",
        w[1]
    );
    assert!(
        w[0] > w[1],
        "W[0] ({:.6}) must exceed W[1] ({:.6}): more rater agreement must yield higher truth probability",
        w[0],
        w[1]
    );
}

// ── Test 4: Output vector lengths ────────────────────────────────────────────

#[test]
fn test_result_lengths() {
    let n = 8usize;
    let k = 3usize;
    let raters: Vec<Vec<f32>> = (0..k)
        .map(|ki| {
            (0..n)
                .map(|i| if (i + ki) % 2 == 0 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    let result = staple(&raters, 100, STAPLE_TOL);

    assert_eq!(
        result.probabilistic_truth.len(),
        n,
        "probabilistic_truth must have length N = {}",
        n
    );
    assert_eq!(
        result.sensitivity.len(),
        k,
        "sensitivity must have length K = {}",
        k
    );
    assert_eq!(
        result.specificity.len(),
        k,
        "specificity must have length K = {}",
        k
    );
}

// ── Test 5: Sensitivity and specificity are in (0, 1) ────────────────────────

#[test]
fn test_sensitivity_range() {
    let n = 50usize;
    let mask: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
    let raters = vec![mask.clone(), mask.clone(), mask.clone()];

    let result = staple(&raters, 100, STAPLE_TOL);

    for (ki, &p) in result.sensitivity.iter().enumerate() {
        assert!(
            p > 0.0 && p < 1.0,
            "sensitivity[{}] = {:.10} must be strictly in (0, 1)",
            ki,
            p
        );
    }
    for (ki, &q) in result.specificity.iter().enumerate() {
        assert!(
            q > 0.0 && q < 1.0,
            "specificity[{}] = {:.10} must be strictly in (0, 1)",
            ki,
            q
        );
    }
}

// ── Test 6: Convergence flag is set for easy inputs ──────────────────────────

#[test]
fn test_convergence_flag() {
    let n = 50usize;
    let mask: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
    let raters = vec![mask.clone(), mask.clone(), mask.clone()];

    // High tolerance and generous iteration budget; must converge.
    let result = staple(&raters, 200, 1e-3);

    assert_eq!(
        result.convergence,
        StapleConvergence::Converged,
        "STAPLE must converge for identical perfect raters with tol=1e-3 (ran {} iterations)",
        result.iterations
    );
}

// ── Test 7: Iteration count is bounded by max_iter ───────────────────────────

#[test]
fn test_iterations_bounded_by_max_iter() {
    let max_iter = 5usize;
    let n = 20usize;
    let mask: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let raters = vec![mask.clone(), mask];

    let result = staple(&raters, max_iter, 1e-12);

    assert!(
        result.iterations <= max_iter,
        "iterations ({}) must not exceed max_iter ({})",
        result.iterations,
        max_iter
    );
}

// ── Test 8: Empty rater slice panics ─────────────────────────────────────────

#[test]
#[should_panic(expected = "raters must be non-empty")]
fn test_empty_raters_panics() {
    let empty: Vec<Vec<f32>> = Vec::new();
    let _ = staple(&empty, 100, STAPLE_TOL);
}

// ── Test 9: Mismatched mask lengths panics ────────────────────────────────────

#[test]
#[should_panic]
fn test_mismatched_lengths_panics() {
    let raters = vec![
        vec![1.0_f32, 0.0, 1.0], // N = 3
        vec![1.0_f32, 0.0],      // N = 2  ← mismatch
    ];
    let _ = staple(&raters, 100, STAPLE_TOL);
}

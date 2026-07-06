//! Atlas-typed cache property tests (sub-batch #3.b of atlas-meta Batch #3, per
//! `atlas/docs/adr/0012-ritk-burn-trait-rebind.md`).
//!
//! # Sub-batch #3.b atomic-boundary invariant (per ADR 0012 §Decision §1 + §atomic-boundary discipline §2)
//!
//! - **Strictly subtractive on test surface**: drops
//!   `burn_ndarray::NdArray`, `ritk_image::tensor::Backend`,
//!   `ritk_image::tensor::Tensor`, `ParzenJointHistogram<B: Backend>`, and the
//!   `B: Backend` dispatch wrapper layer from this test module. Drops one
//!   source-row from `xtask/burn_surface.allowlist`. The legacy
//!   `direct::build_sparse_w_fixed_transposed` and
//!   `direct::compute_joint_histogram_direct` symbols are accessed through
//!   `atlas_parzen_cache::*` as `build_atlas_sparse_w_fixed_transposed` /
//!   `compute_atlas_joint_histogram_direct` re-export aliases (function
//!   re-exports respect the `pub(in crate::metric::histogram)` visibility
//!   markers). The leaf direct-path CPU algorithms themselves are untouched.
//! - **Strictly additive on production surface**: the companion
//!   `atlas_parzen_cache.rs` sibling module provides the Atlas-prefixed
//!   function re-exports plus `atlas_normalize_intensities` (host-slice
//!   normalisation helper operating on `&[f32]`). The type symbol
//!   `ParzenConfig` (defined as `pub(crate)` in `direct::ParzenConfig`)
//!   is consumed through the crate-local path
//!   `crate::metric::histogram::parzen::direct::ParzenConfig` instead of
//!   being elevated through `pub use as AtlasParzenConfig;` (Rust rejects
//!   visibility-elevation of `pub(crate)` items via `pub use as Alias`).
//! - **Cargo.toml**: zero changes this sub-batch (no deletion/rename/feature
//!   flag change).
//!
//! # Test bodies
//!
//! T1 (`sparse_w_fixed_deterministic`): the call to
//! `build_atlas_sparse_w_fixed_transposed` is invoked twice with identical
//! inputs, and the two cache results must match entry-for-entry (sample index,
//! bin, weight, inv_sum_f).
//!
//! T2 (`histogram_non_negative_all_entries`): the joint histogram flat vector
//! returned by `compute_atlas_joint_histogram_direct` must contain only
//! non-negative entries (Parzen weights are `exp(...)`, always `>= 0`).
//!
//! T3 (`histogram_marginals_sum_correctly`): fixed-axis marginal-sum must
//! equal moving-axis marginal-sum; both per-axis entries must be non-negative.

use crate::metric::histogram::parzen::atlas_parzen_cache::{
    atlas_normalize_intensities, build_atlas_sparse_w_fixed_transposed,
    compute_atlas_joint_histogram_direct,
};
use crate::metric::histogram::parzen::direct::ParzenConfig;

/// T1: Determinism of sparse W_fixed^T cache construction under repeat calls.
#[cfg(feature = "direct-parzen")]
#[test]
fn sparse_w_fixed_deterministic() {
    let num_bins = 32usize;
    let sigma_sq_fix = ParzenConfig::from_intensity_sigma(
        255.0 / 32.0,
        0.0,
        255.0,
        num_bins,
    )
    .sigma_sq(); // SSOT-319-02

    let n = 500;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.51) % 255.0).collect();
    let fixed_norm = atlas_normalize_intensities(&fixed, 0.0, 255.0, num_bins);

    let sparse1 = build_atlas_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);
    let sparse2 = build_atlas_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);

    assert_eq!(
        sparse1.len(),
        sparse2.len(),
        "sparse cache lengths must match"
    );
    for (i, (a, b)) in sparse1.iter().zip(sparse2.iter()).enumerate() {
        // SPARSE-329-01: cache entry is (entries, inv_sum_f) tuple.
        assert_eq!(a.0.len(), b.0.len(), "sample {i}: entry count mismatch");
        assert!(
            (a.1 - b.1).abs() < 1e-10,
            "sample {i}: inv_sum_f mismatch a={} b={}",
            a.1,
            b.1
        );
        for (j, (ea, eb)) in a.0.iter().zip(b.0.iter()).enumerate() {
            assert_eq!(ea.bin, eb.bin, "sample {i} entry {j}: bin mismatch");
            let diff = (ea.weight - eb.weight).abs();
            assert!(diff < 1e-10, "sample {i} entry {j}: weight diff={diff}");
        }
    }
}

/// T2: All entries in the joint histogram must be non-negative.
#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_non_negative_all_entries() {
    let num_bins = 16usize;
    let cfg = ParzenConfig::from_intensity_sigma(255.0 / 16.0, 0.0, 255.0, num_bins);
    let sigma_sq = cfg.sigma_sq();

    let n = 200;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 1.27 + 3.0) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.83 + 7.0) % 255.0).collect();
    let fixed_norm = atlas_normalize_intensities(&fixed, 0.0, 255.0, num_bins);
    let moving_norm = atlas_normalize_intensities(&moving, 0.0, 255.0, num_bins);

    let h = compute_atlas_joint_histogram_direct(
        &fixed_norm,
        &moving_norm,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None, // no histogram_pool for tests
    );

    for (i, &v) in h.iter().enumerate() {
        assert!(v >= 0.0, "histogram entry {i} is negative: {v}");
    }
}

/// T3: Fixed/moving marginal sums must equal; both must be non-negative.
#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_marginals_sum_correctly() {
    let num_bins = 16usize;
    let cfg = ParzenConfig::from_intensity_sigma(255.0 / 16.0, 0.0, 255.0, num_bins);
    let sigma_sq = cfg.sigma_sq();

    let fixed: Vec<f32> = vec![50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0, 40.0];
    let moving: Vec<f32> = vec![60.0, 130.0, 195.0, 25.0, 180.0, 90.0, 215.0, 35.0];
    let fixed_norm = atlas_normalize_intensities(&fixed, 0.0, 255.0, num_bins);
    let moving_norm = atlas_normalize_intensities(&moving, 0.0, 255.0, num_bins);

    let h = compute_atlas_joint_histogram_direct(
        &fixed_norm,
        &moving_norm,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );

    let mut fixed_marginal = vec![0.0f32; num_bins];
    let mut moving_marginal = vec![0.0f32; num_bins];

    for a in 0..num_bins {
        for b in 0..num_bins {
            let v = h[a * num_bins + b];
            fixed_marginal[a] += v;
            moving_marginal[b] += v;
        }
    }

    let sum_fixed: f32 = fixed_marginal.iter().sum();
    let sum_moving: f32 = moving_marginal.iter().sum();
    let diff = (sum_fixed - sum_moving).abs();
    assert!(
        diff < 1e-4,
        "marginal sums must match: fixed={sum_fixed}, moving={sum_moving}, diff={diff}"
    );

    for (i, &v) in fixed_marginal.iter().enumerate() {
        assert!(v >= 0.0, "fixed marginal entry {i} is negative: {v}");
    }
    for (i, &v) in moving_marginal.iter().enumerate() {
        assert!(v >= 0.0, "moving marginal entry {i} is negative: {v}");
    }
}

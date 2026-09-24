//! Differential test: the tridiagonal-QL denoiser against a Jacobi
//! eigensolver reference, one window at a time.
//!
//! With the image shape equal to the window extent, every voxel owns the same
//! window, so the denoiser output is that window's reconstruction averaged
//! over `V` identical copies, and the reference can be computed directly: the
//! same Gram matrix decomposed by [`leto_ops::symmetric_eigen_jacobi_with_tolerance`],
//! the same Marchenko-Pastur boundary, the same projection.
//!
//! # Derived tolerance
//!
//! Both eigensolvers are backward stable on the Gram matrix `G` (`m × m`):
//! Householder tridiagonalization plus QL is exact for `G + E` with
//! `‖E‖_F ≤ m²·ε·‖G‖_F` (Higham 2002, Lemma 19.3, `m` reflectors of `γ̃_m`
//! each), and Jacobi stopped at off-diagonal tolerance `τ` adds at most `m·τ`
//! to its own `m²·ε·‖G‖_F`. Forming `G = YᵀY/n` in floating point perturbs it
//! by at most `ε·‖Y‖²_F` (`γ_n` per sum of `n` products, over `n`). The two
//! spectra therefore agree to
//!
//! ```text
//! δ = (2m² + 1)·ε·‖G‖_F + ε·‖Y‖²_F + m·τ          (Weyl)
//! ```
//!
//! Given matching `P̂`, the signal projectors differ by at most
//! `2δ/(gap − 2δ)` in norm (Davis–Kahan `sin Θ`, gap `λ_P̂ − λ_{P̂+1}`), and
//! the projection sums and the `V`-copy average add `2m·ε` and `V·ε`
//! relative, so every reconstructed sample agrees to
//!
//! ```text
//! ‖Y‖_F·(2δ/(gap − 2δ) + 2m·ε) + V·ε·max|Ŷ|.
//! ```
//!
//! `P̂` is compared only after asserting that the reference criterion clears
//! the perturbation: moving every eigenvalue by `δ` moves the trailing sum by
//! `(m − p)·δ` and the predicted bulk `(λ_{p+1} − λ_m)/(4√γ_p)` by
//! `2δ/(4√γ_p)`, so a criterion margin above `(m − p)·δ·(1 + 1/(2√γ_p))` at
//! `p = P̂ − 1` and `p = P̂` fixes `P̂` for both solvers.

use super::super::threshold::marchenko_pastur_boundary;
use super::super::{MpEstimator, MpPcaDenoiser, PatchExtent};
use super::{low_rank_series, with_noise, DEPTH, SHAPE};
use leto::Array2;
use leto_ops::symmetric_eigen_jacobi_with_tolerance;

/// The first `extent` voxels of every axis of each volume, flattened in the
/// window's `z, y, x` order.
fn crop(series: &[Vec<f64>], extent: [usize; 3]) -> Vec<Vec<f64>> {
    series
        .iter()
        .map(|volume| {
            let mut cropped = Vec::with_capacity(extent.iter().product());
            for z in 0..extent[0] {
                for y in 0..extent[1] {
                    let start = (z * SHAPE[1] + y) * SHAPE[2];
                    cropped.extend_from_slice(&volume[start..start + extent[2]]);
                }
            }
            cropped
        })
        .collect()
}

fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The Marchenko-Pastur criterion margin `Σ_{i>p} λᵢ − (m − p)·σ̂²(p)` and the
/// perturbation it must clear, for the descending spectrum `lambda`.
fn criterion_margin(
    lambda: &[f64],
    n: usize,
    p: usize,
    estimator: MpEstimator,
    delta: f64,
) -> (f64, f64) {
    let m = lambda.len();
    let count = (m - p) as f64;
    let columns = match estimator {
        MpEstimator::Veraart2016 => n,
        MpEstimator::CorderoGrande2019 => n - p,
    };
    let gamma = count / columns as f64;
    let bulk = (lambda[p] - lambda[m - 1]) / (4.0 * gamma.sqrt());
    let sum: f64 = lambda[p..].iter().sum();
    let margin = (sum - count * bulk).abs();
    let perturbation = count * delta * (1.0 + 1.0 / (2.0 * gamma.sqrt()));
    (margin, perturbation)
}

fn check_window_matches_jacobi(extent: [usize; 3], estimator: MpEstimator) {
    let series = crop(&with_noise(&low_rank_series(51), 1.0, 52), extent);
    let voxels: usize = extent.iter().product();
    let (m, n) = (voxels.min(DEPTH), voxels.max(DEPTH));
    let volumes_smaller = DEPTH <= voxels;
    // Casorati matrix: row = voxel, column = volume.
    let y: Vec<f64> = (0..voxels)
        .flat_map(|v| series.iter().map(move |volume| volume[v]))
        .collect();
    let mut gram = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let dot: f64 = if volumes_smaller {
                (0..voxels)
                    .map(|r| y[r * DEPTH + i] * y[r * DEPTH + j])
                    .sum()
            } else {
                (0..DEPTH)
                    .map(|c| y[i * DEPTH + c] * y[j * DEPTH + c])
                    .sum()
            };
            gram[i * m + j] = dot / n as f64;
        }
    }
    let epsilon = f64::EPSILON;
    let gram_norm = frobenius(&gram);
    let tau = m as f64 * epsilon * gram_norm;
    let reference = symmetric_eigen_jacobi_with_tolerance(
        &Array2::from_shape_vec([m, m], gram)
            .expect("invariant: m² entries")
            .view(),
        tau,
    )
    .expect("invariant: a finite symmetric Gram matrix decomposes");
    let descending: Vec<f64> = reference.eigenvalues.iter().rev().copied().collect();
    let boundary = marchenko_pastur_boundary(&descending, n, estimator, &mut Vec::new());
    let p = boundary.signal_components;
    let delta = (2 * m * m + 1) as f64 * epsilon * gram_norm
        + epsilon * frobenius(&y).powi(2)
        + m as f64 * tau;

    for candidate in [p.checked_sub(1), (p + 1 < m).then_some(p)]
        .into_iter()
        .flatten()
    {
        let (margin, perturbation) = criterion_margin(&descending, n, candidate, estimator, delta);
        assert!(
            margin > perturbation,
            "extent {extent:?}: criterion at p = {candidate} is within the solver bound \
             ({margin} ≤ {perturbation}); P̂ is not determined by this seed"
        );
    }

    // Reference reconstruction Ŷ = Y·U·Uᵀ or W·Wᵀ·Y over the top P̂ vectors.
    let vectors = reference
        .eigenvectors
        .as_slice()
        .expect("invariant: leto builds eigenvectors in contiguous row-major storage");
    let mut expected = vec![0.0; voxels * DEPTH];
    for k in m - p..m {
        let u = |i: usize| vectors[i * m + k];
        if volumes_smaller {
            for r in 0..voxels {
                let coefficient: f64 = (0..DEPTH).map(|i| y[r * DEPTH + i] * u(i)).sum();
                for i in 0..DEPTH {
                    expected[r * DEPTH + i] += coefficient * u(i);
                }
            }
        } else {
            for c in 0..DEPTH {
                let coefficient: f64 = (0..voxels).map(|r| u(r) * y[r * DEPTH + c]).sum();
                for r in 0..voxels {
                    expected[r * DEPTH + c] += coefficient * u(r);
                }
            }
        }
    }

    let views: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let extent_value = PatchExtent::new(extent).expect("invariant: a valid test extent");
    let output = MpPcaDenoiser::default()
        .with_extent(extent_value)
        .with_estimator(estimator)
        .denoise(extent, &views)
        .expect("invariant: the window fits the cropped image");

    assert!(output.signal_components().iter().all(|&c| c == p));
    for &sigma in output.noise_sigma() {
        let (variance, reference) = (sigma * sigma, boundary.variance);
        assert!(
            (variance - reference).abs() <= delta,
            "σ̂² = {variance}, reference {reference}, bound {delta}"
        );
    }
    let gap = descending[p - 1] - descending[p];
    let largest = expected.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    let bound = frobenius(&y) * (2.0 * delta / (gap - 2.0 * delta) + 2.0 * m as f64 * epsilon)
        + voxels as f64 * epsilon * largest;
    for (v, volume) in output.volumes().iter().enumerate() {
        for (voxel, &value) in volume.iter().enumerate() {
            let reference = expected[voxel * DEPTH + v];
            assert!(
                (value - reference).abs() <= bound,
                "extent {extent:?}, voxel {voxel}, volume {v}: {value} vs {reference}, bound {bound}"
            );
        }
    }
}

#[test]
fn reconstruction_matches_the_jacobi_reference_within_backward_error() {
    // 4³ = 64 voxels > D = 32: the Gram matrix is formed on the volumes.
    // 2³ = 8 voxels < D: it is formed on the voxels.
    for extent in [[4, 4, 4], [2, 2, 2]] {
        for estimator in [MpEstimator::CorderoGrande2019, MpEstimator::Veraart2016] {
            check_window_matches_jacobi(extent, estimator);
        }
    }
}

//! Analytical-oracle tests for MP-PCA denoising.
//!
//! The synthetic series is rank `P` plus i.i.d. Gaussian noise of known `σ`.
//! Tolerances derive from the estimator's sampling distribution for the window
//! geometry and from the eigensolver's residue floor, stated at each
//! assertion.

use super::threshold::{marchenko_pastur_boundary, NoiseBoundary};
use super::MpEstimator;
use super::{MpPcaDenoiser, MpPcaError, PatchExtent};
use leto_ops::RealScalar;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Image shape of the synthetic series.
const SHAPE: [usize; 3] = [8, 8, 8];
/// Volume count `D`; the derived window is the 4³ = 64-voxel cube.
const DEPTH: usize = 32;
/// Signal rank `P`.
const RANK: usize = 3;
/// Spatial loading amplitude: each signal eigenvalue ≈ `A²·D ≈ 1.3e4 σ²`,
/// far beyond the bulk edge `λ₊ = σ²(1 + √(32/64))² ≈ 2.9 σ²`.
const AMPLITUDE: f64 = 20.0;
/// Absolute off-diagonal residue floor `τ` in the passthrough bounds below:
/// the `10⁻¹²` tolerance of the Jacobi solver they were derived for. The
/// tridiagonal-QL solver deflates at `ε·‖T‖` with no floor; its backward
/// error `p(m)·ε·‖G‖` is of the `m·ε` order these bounds carry in practice,
/// while its worst case `p(m) = m²` is asserted by `jacobi_reference`.
const JACOBI_TOLERANCE: f64 = 1e-12;
/// Bound on the signal condition `λ₁/λ_P`: the three loadings are i.i.d.
/// standard normal over 64 voxels, so each eigenvalue is `A²·D·χ²₆₄/64`;
/// a ratio of 8 lies beyond any seed's reach (χ²₆₄/64 ∈ [0.5, 1.6] at 5σ).
const CONDITION_BOUND: f64 = 8.0;

fn gaussian(rng: &mut StdRng) -> f64 {
    // Box-Muller; `1 − u` keeps the logarithm's argument in (0, 1].
    let u: f64 = rng.random();
    let v: f64 = rng.random();
    (-2.0 * (1.0 - u).ln()).sqrt() * (2.0 * PI * v).cos()
}

/// Clean rank-`RANK` series: `Y[d][v] = A·Σ_k a_k(v)·√2·cos(2π(k+1)d/D)`.
///
/// The volume profiles are orthogonal over `d`, the loadings i.i.d. normal.
fn low_rank_series(seed: u64) -> Vec<Vec<f64>> {
    let voxels: usize = SHAPE.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let loadings: Vec<Vec<f64>> = (0..RANK)
        .map(|_| (0..voxels).map(|_| gaussian(&mut rng)).collect())
        .collect();
    (0..DEPTH)
        .map(|d| {
            (0..voxels)
                .map(|v| {
                    (0..RANK)
                        .map(|k| {
                            let phase = 2.0 * PI * (k + 1) as f64 * d as f64 / DEPTH as f64;
                            AMPLITUDE * loadings[k][v] * 2.0_f64.sqrt() * phase.cos()
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn with_noise(clean: &[Vec<f64>], sigma: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    clean
        .iter()
        .map(|volume| {
            volume
                .iter()
                .map(|&x| x + sigma * gaussian(&mut rng))
                .collect()
        })
        .collect()
}

fn cast<T: RealScalar>(series: &[Vec<f64>]) -> Vec<Vec<T>> {
    series
        .iter()
        .map(|volume| volume.iter().map(|&x| T::from_f64(x)).collect())
        .collect()
}

fn denoise<T: RealScalar>(series: &[Vec<T>]) -> super::MpPcaOutput<T> {
    let views: Vec<&[T]> = series.iter().map(Vec::as_slice).collect();
    MpPcaDenoiser::default()
        .denoise(SHAPE, &views)
        .expect("invariant: the synthetic series is valid for the derived window")
}

fn rmse(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let (sum, count) = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold((0.0, 0_usize), |(s, c), (x, y)| {
            (s + (x - y).powi(2), c + 1)
        });
    (sum / count as f64).sqrt()
}

/// Window geometry of the synthetic series: `(m, n) = (D, V) = (32, 64)`.
fn window_dimensions() -> (f64, f64) {
    let voxels = PatchExtent::for_volume_count(DEPTH).voxels();
    (DEPTH.min(voxels) as f64, DEPTH.max(voxels) as f64)
}

#[test]
fn boundary_separates_a_spike_from_a_flat_bulk() {
    // p = 0: γ = 4/100, σ̂²(0) = 99/(4·0.2) = 123.75, trailing sum 103 < 495.
    // p = 1: zero support width, trailing sum 3 ≥ 0 — P̂ = 1, σ̂² = 3/3 = 1.
    let boundary = marchenko_pastur_boundary(
        &[100.0_f64, 1.0, 1.0, 1.0],
        100,
        MpEstimator::Veraart2016,
        &mut Vec::new(),
    );
    assert_eq!(
        boundary,
        NoiseBoundary {
            signal_components: 1,
            variance: 1.0
        }
    );
}

#[test]
fn boundary_floors_rounding_residue_at_zero() {
    // p = 0 fails (sum < 0 < bound); p = 1 is accepted with mean −1e-18,
    // which is rounding residue of an exactly zero spectrum.
    let boundary = marchenko_pastur_boundary(
        &[1e-20_f64, -1e-18],
        2,
        MpEstimator::Veraart2016,
        &mut Vec::new(),
    );
    assert_eq!(
        boundary,
        NoiseBoundary {
            signal_components: 1,
            variance: 0.0
        }
    );
}

#[test]
fn derived_extent_is_the_smallest_cube_covering_the_volume_count() {
    let sides: Vec<usize> = [1, 8, 9, 27, 28, 60, 64, 65]
        .iter()
        .map(|&d| PatchExtent::for_volume_count(d).extent()[0])
        .collect();
    assert_eq!(sides, [2, 2, 3, 3, 4, 4, 4, 5]);
}

#[test]
fn windows_shift_inward_at_borders() {
    let extent = PatchExtent::new([3, 3, 3]).expect("3³ is a valid window");
    let shape = [5, 5, 5];
    assert_eq!(extent.origin([0, 2, 4], shape), [0, 1, 2]);
    assert_eq!(extent.origin([1, 3, 2], shape), [0, 2, 1]);
}

#[test]
fn invalid_extents_are_rejected() {
    for extent in [[0, 3, 3], [1, 1, 1]] {
        assert!(matches!(
            PatchExtent::new(extent),
            Err(MpPcaError::InvalidPatch { extent: e }) if e == extent
        ));
    }
}

#[test]
fn noise_map_matches_the_true_sigma() {
    let sigma = 1.0;
    let clean = low_rank_series(11);
    let noisy = with_noise(&clean, sigma, 12);
    let output = denoise(&noisy);
    let (m, n) = window_dimensions();
    let p = RANK as f64;
    // Eq. 12 averages m − P trailing eigenvalues whose total is, to first
    // order, σ²·χ²_{(m−P)(n−P)}/n, so E[σ̂²] = σ²(n − P)/n and
    // sd(σ̂²)/σ² = √(2/((m − P)(n − P))). Through the square root:
    // E[σ̂] ≈ σ√((n − P)/n), sd(σ̂) ≈ σ/√(2(m − P)(n − P)) = 0.0168σ.
    let expected = sigma * ((n - p) / n).sqrt();
    let sd = sigma / (2.0 * (m - p) * (n - p)).sqrt();
    // Five standard deviations: per-voxel tail probability 5.7e-7, so the
    // 512-voxel map exceeds it with probability below 3e-4.
    let bound = 5.0 * sd;
    for (voxel, &estimate) in output.noise_sigma().iter().enumerate() {
        assert!(
            (estimate - expected).abs() <= bound,
            "voxel {voxel}: σ̂ = {estimate}, expected {expected} ± {bound}"
        );
    }
    // Every signal eigenvalue (≈ A²·D) is four orders above the bulk, so no
    // window can classify one as noise: P̂ ≥ P everywhere.
    assert!(output.signal_components().iter().all(|&c| c >= RANK));
}

#[test]
fn refined_ratio_never_counts_more_components() {
    // γ_p = (m − p)/(n − p) ≥ (m − p)/n makes the predicted bulk width
    // 4σ²√γ_p at least as wide at every p, so whenever Veraart's Eq. 10 accepts
    // p, the refined criterion accepts it too: P̂_refined ≤ P̂_Veraart per
    // window. Finite-size edge fluctuations make P̂ exceed P in some windows
    // under either ratio; the refinement can only reduce that count.
    let noisy = with_noise(&low_rank_series(41), 1.0, 42);
    let views: Vec<&[f64]> = noisy.iter().map(Vec::as_slice).collect();
    let run = |estimator| {
        MpPcaDenoiser::default()
            .with_estimator(estimator)
            .denoise(SHAPE, &views)
            .expect("invariant: the synthetic series is valid for the derived window")
    };
    let veraart = run(MpEstimator::Veraart2016);
    let refined = run(MpEstimator::CorderoGrande2019);
    for (voxel, (&v, &r)) in veraart
        .signal_components()
        .iter()
        .zip(refined.signal_components())
        .enumerate()
    {
        assert!(
            RANK <= r && r <= v,
            "voxel {voxel}: P̂ refined {r}, Veraart {v}"
        );
    }
}

#[test]
fn denoising_reduces_rmse_below_the_projection_bound() {
    let sigma = 1.0;
    let clean = low_rank_series(21);
    let noisy = with_noise(&clean, sigma, 22);
    let denoised = denoise(&noisy).into_volumes();
    let (m, n) = window_dimensions();
    let p = RANK as f64;
    let noisy_rmse = rmse(&noisy, &clean);
    let denoised_rmse = rmse(&denoised, &clean);
    // Rank-P truncation of one window keeps noise in the P-dimensional
    // column and row subspaces: P(m + n − P) of the m·n degrees of freedom,
    // so its RMSE is σ√(P(m + n − P)/(m·n)) = 0.369σ. Averaging overlapping
    // windows cannot raise the RMSE (Jensen), so this bounds the output.
    let factor = (p * (m + n - p) / (m * n)).sqrt();
    assert!(
        (noisy_rmse - sigma).abs() <= 0.05 * sigma,
        "noisy RMSE {noisy_rmse} should be σ within its χ² sampling spread"
    );
    assert!(
        denoised_rmse <= factor * sigma,
        "denoised RMSE {denoised_rmse} exceeds {factor}·σ (noisy {noisy_rmse})"
    );
}

/// Relative reconstruction error bound for a noise-free window.
///
/// The eigensolver stops at off-diagonal residue `max(τ, m·ε)·λ₁`, which
/// tilts the signal subspace by at most that over the gap `λ_P` (Davis–Kahan),
/// i.e. by `max(τ, m·ε)·κ` with `κ ≤ CONDITION_BOUND`; the projection
/// `Y·U·Uᵀ` doubles it, and the Gram and projection sums add `m·ε` each.
fn passthrough_bound(epsilon: f64, m: f64) -> f64 {
    let residue = JACOBI_TOLERANCE.max(m * epsilon);
    4.0 * residue * CONDITION_BOUND + 4.0 * m * epsilon
}

fn noise_free_series_passes_through<T: RealScalar>(epsilon: f64) {
    let clean = low_rank_series(31);
    let output = denoise(&cast::<T>(&clean));
    let (m, _) = window_dimensions();
    let scale = clean
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    let bound = passthrough_bound(epsilon, m) * scale;
    for (clean, denoised) in clean.iter().zip(output.volumes()) {
        for (a, b) in clean.iter().zip(denoised) {
            let error = (T::from_f64(*a) - *b).abs().to_f64();
            assert!(error <= bound, "|{a} − {b:?}| = {error} > {bound}");
        }
    }
}

#[test]
fn noise_free_low_rank_series_passes_through() {
    noise_free_series_passes_through::<f64>(f64::EPSILON);
    noise_free_series_passes_through::<f32>(f64::from(f32::EPSILON));
}

fn constant_series_is_reproduced<T: RealScalar>(epsilon: f64) {
    let shape = [4, 4, 4];
    let depth = 8;
    let value = 7.5_f64;
    let volumes = vec![vec![T::from_f64(value); 64]; depth];
    let views: Vec<&[T]> = volumes.iter().map(Vec::as_slice).collect();
    let output = MpPcaDenoiser::default()
        .denoise(shape, &views)
        .expect("invariant: a constant series is a valid input");
    // Rank one: λ₁ = c²·D and every other eigenvalue is solver residue of at
    // most max(τ, m·ε)·λ₁, so σ̂ ≤ c·√(max(τ, m·ε)·D).
    let m = depth as f64;
    let residue = JACOBI_TOLERANCE.max(m * epsilon);
    let sigma_bound = value * (residue * depth as f64).sqrt();
    let value_bound = 4.0 * (residue + m * epsilon) * value;
    for denoised in output.volumes() {
        for &x in denoised {
            assert!(
                (x.to_f64() - value).abs() <= value_bound,
                "{x:?} vs {value}"
            );
        }
    }
    for &s in output.noise_sigma() {
        assert!(s.to_f64() <= sigma_bound, "σ̂ = {s:?} > {sigma_bound}");
    }
}

#[test]
fn constant_series_is_reproduced_with_zero_noise() {
    constant_series_is_reproduced::<f64>(f64::EPSILON);
    constant_series_is_reproduced::<f32>(f64::from(f32::EPSILON));
}

#[test]
fn patch_larger_than_the_image_is_rejected() {
    let volumes = vec![vec![1.0_f64; 27]; 100];
    let views: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let error = MpPcaDenoiser::default()
        .denoise([3, 3, 3], &views)
        .expect_err("the derived 5³ window cannot fit a 3³ image");
    assert!(matches!(
        error,
        MpPcaError::PatchExceedsImage {
            extent: [5, 5, 5],
            shape: [3, 3, 3]
        }
    ));
}

#[test]
fn single_volume_is_rejected() {
    let volume = vec![1.0_f64; 27];
    let error = MpPcaDenoiser::default()
        .denoise([3, 3, 3], &[volume.as_slice()])
        .expect_err("one volume has no noise bulk");
    assert!(matches!(
        error,
        MpPcaError::TooFewVolumes {
            count: 1,
            minimum: 2
        }
    ));
}

#[test]
fn malformed_series_is_rejected() {
    let short = vec![0.0_f64; 26];
    let full = vec![0.0_f64; 27];
    let error = MpPcaDenoiser::default()
        .denoise([3, 3, 3], &[full.as_slice(), short.as_slice()])
        .expect_err("volume 1 is one sample short");
    assert!(matches!(
        error,
        MpPcaError::VolumeLength {
            volume: 1,
            len: 26,
            expected: 27,
            ..
        }
    ));

    let mut poisoned = full.clone();
    poisoned[13] = f64::NAN;
    let error = MpPcaDenoiser::default()
        .denoise([3, 3, 3], &[full.as_slice(), poisoned.as_slice()])
        .expect_err("a NaN sample is rejected");
    assert!(matches!(
        error,
        MpPcaError::NonFinite {
            volume: 1,
            sample: 13
        }
    ));
}

/// IEEE-754 bit patterns of a series, widened exactly to `f64`, so `±0` and
/// every other value compare by representation rather than by `==`.
fn bits<T: RealScalar>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.to_f64().to_bits())
        .collect()
}

/// The `[4, 4, 5]` corner of each volume: every window extent below still
/// fits, and the 80 windows keep the debug-build sweep inside the test budget.
fn corner(series: &[Vec<f64>]) -> ([usize; 3], Vec<Vec<f64>>) {
    let shape = [4, 4, 5];
    let cropped = series
        .iter()
        .map(|volume| {
            (0..shape[0])
                .flat_map(|z| (0..shape[1]).map(move |y| (z, y)))
                .flat_map(|(z, y)| {
                    let start = (z * SHAPE[1] + y) * SHAPE[2];
                    volume[start..start + shape[2]].iter().copied()
                })
                .collect()
        })
        .collect();
    (shape, cropped)
}

/// The parallel sweep equals a sequential one bit for bit, for one window
/// per region and for a width that divides nothing, on both Gram
/// orientations: each voxel sums its windows in centre order whatever the
/// partition.
fn check_parallel_sweep_matches_sequential<T: RealScalar>() {
    let (shape, noisy) = corner(&with_noise(&low_rank_series(11), 1.0, 12));
    let noisy = cast::<T>(&noisy);
    let views: Vec<&[T]> = noisy.iter().map(Vec::as_slice).collect();
    let voxels: usize = shape.iter().product();
    // The derived 4³ window has V = 64 > D = 32; a 2³ window has V = 8 < D.
    let small = PatchExtent::new([2, 2, 2]).expect("invariant: 2³ is a valid extent");
    for extent in [PatchExtent::for_volume_count(DEPTH), small] {
        let denoiser = MpPcaDenoiser::default().with_extent(extent);
        let sequential = denoiser
            .sweep::<moirai::Sequential, T>(shape, &views, extent, voxels)
            .expect("invariant: the synthetic series is valid for the window");
        for batch in [1, 7] {
            let parallel = denoiser
                .sweep::<moirai::Parallel, T>(shape, &views, extent, batch)
                .expect("invariant: the synthetic series is valid for the window");
            for (a, b) in sequential.volumes().iter().zip(parallel.volumes()) {
                assert_eq!(bits(a), bits(b), "batch {batch}, extent {extent:?}");
            }
            assert_eq!(bits(sequential.noise_sigma()), bits(parallel.noise_sigma()));
            assert_eq!(sequential.signal_components(), parallel.signal_components());
        }
        // The public entry point stages the whole image in one region.
        let public = denoiser
            .denoise(shape, &views)
            .expect("invariant: the synthetic series is valid for the window");
        for (a, b) in sequential.volumes().iter().zip(public.volumes()) {
            assert_eq!(bits(a), bits(b), "public entry, extent {extent:?}");
        }
    }
}

#[test]
fn parallel_sweep_matches_sequential() {
    check_parallel_sweep_matches_sequential::<f32>();
    check_parallel_sweep_matches_sequential::<f64>();
}

mod jacobi_reference;

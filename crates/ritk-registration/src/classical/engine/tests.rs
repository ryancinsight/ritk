use super::*;
use leto::{Array2, Array3};

#[test]
fn test_rigid_landmark_identity() {
    let reg = ImageRegistration::default();

    let fixed = Array2::from_vec([3, 3], vec![0., 0., 0., 1., 0., 0., 0., 1., 0.])
        .expect("valid dimension");
    let result = reg
        .rigid_registration_landmarks(&fixed, &fixed)
        .expect("infallible: validated precondition");

    // Identity transform should have zero FRE
    let fre = result
        .quality
        .fre
        .expect("infallible: validated precondition");
    assert!(
        fre < 1e-10,
        "FRE for identity transform should be ~0, got {}",
        fre
    );
}

#[test]
fn test_rigid_landmark_known_rotation() {
    let reg = ImageRegistration::default();

    // Fixed points: unit vectors along axes
    let fixed = Array2::from_vec([3, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1.])
        .expect("valid dimension");
    // Moving points: same points rotated 90 deg around Z-axis
    let moving = Array2::from_vec([3, 3], vec![0., 1., 0., -1., 0., 0., 0., 0., 1.])
        .expect("valid dimension");

    let result = reg
        .rigid_registration_landmarks(&fixed, &moving)
        .expect("infallible: validated precondition");

    let fre = result
        .quality
        .fre
        .expect("infallible: validated precondition");
    assert!(
        fre < 1e-6,
        "FRE for 90 deg rotation should be ~0, got {}",
        fre
    );
}

// ── Mutual-information metric: algebraic properties ──────────────────────────
//
// `MutualInformationMetric::compute` returns the normalized mutual information
//   NMI = 2·MI / (H(X) + H(Y)),   MI = H(X) + H(Y) − H(X,Y),
// estimated from a plug-in `num_bins × num_bins` joint histogram.  The tests
// below assert the identities that definition implies.  Each is falsifiable by
// a real defect in the binning, the marginalisation, or the entropy sums —
// unlike a non-negativity or `< 1` bound, which the formula's structure and the
// degenerate-constant early return satisfy regardless of what was computed.

/// Worst-case floating-point tolerance for the exact NMI identities below.
///
/// Derivation.  The default metric uses `num_bins = 32`, so the joint entropy
/// sums 32² = 1024 terms sequentially and each marginal entropy sums 32.  The
/// worst-case accumulated rounding of a sequential sum of `n` f64 terms is
/// `n·ε·Σ|term|` with `ε = 2⁻⁵³ ≈ 1.11e-16`.  The largest entropy representable
/// on 1024 joint bins is `ln(1024) = 6.93` nats, so
///   `|δH(X,Y)| ≤ 1024 · 1.11e-16 · 6.93 ≈ 7.9e-13`,
/// and the two 32-term marginal sums contribute strictly less.  The numerator
/// `2·MI` therefore carries at most `≈ 3.2e-12` of error, and the denominator
/// `H(X)+H(Y) ≥ 2·ln(2) = 1.386` for any pair of non-constant images used here
/// (two occupied bins is the minimum), so the propagated NMI error is bounded
/// by `3.2e-12 / 1.386 ≈ 2.3e-12`.  Rounded up one order of magnitude:
const NMI_ROUNDING_TOLERANCE: f64 = 1e-11;

/// Intensity landing exactly in histogram bin `bin` of the default metric.
///
/// The default metric bins `[0, 255)` into 32 bins of width `255/32 = 7.96875`.
/// The centre of bin `b` is `(b + 0.5)·width`, and `floor((b + 0.5)·w / w) = b`
/// exactly, so a value built this way is never ambiguous at a bin edge.
fn bin_centre(bin: usize) -> f64 {
    (bin as f64 + 0.5) * (255.0 / 32.0)
}

/// Sinusoid periods and phase offsets (voxels) for [`smooth_volume`].
///
/// The periods are far longer than the 12-voxel test grid, so each axis samples
/// less than a third of one cycle and the pattern is **aperiodic over the
/// volume**: no shift within the tested range can bring the image back into
/// partial self-alignment.  A period comparable to the grid (7–11 voxels) makes
/// the misalignment ladder non-monotonic through exactly that mechanism.  The
/// phases place each axis's arc across the sinusoid's peak, maximising the
/// intensity span and hence the marginal entropy.
const SMOOTH_PERIODS: [f64; 3] = [37.0, 41.0, 43.0];
const SMOOTH_PHASES: [f64; 3] = [9.0, 10.0, 11.0];

/// Smooth analytic intensity volume on `dims`, sampled at `index + shift`.
///
/// `I(z, y, x) = 127.5 + 40·Σ sin(2π(cᵢ + φᵢ + sᵢ)/Lᵢ)`
///
/// The range is at worst `127.5 ± 120 = [7.5, 247.5]`, strictly inside the
/// default metric's `[0, 255)` binning window, so no sample is ever dropped by
/// the bin-range guard.  Over the 12³ grid the realised span is about 136
/// intensity units — 17 of the 32 bins — so the marginals carry ample entropy.
fn smooth_volume(dims: [usize; 3], shift: [f64; 3]) -> Array3<f64> {
    let [nz, ny, nx] = dims;
    let values = (0..nz * ny * nx)
        .map(|index| {
            let coords = [
                (index / (ny * nx)) as f64,
                ((index / nx) % ny) as f64,
                (index % nx) as f64,
            ];
            127.5
                + 40.0
                    * (0..3)
                        .map(|axis| {
                            let c = coords[axis] + SMOOTH_PHASES[axis] + shift[axis];
                            (std::f64::consts::TAU * c / SMOOTH_PERIODS[axis]).sin()
                        })
                        .sum::<f64>()
        })
        .collect();
    Array3::from_vec(dims, values).expect("dimensions match the generated element count")
}

/// `MI(X, X) = H(X)`, so `NMI(X, X) = 2·H/(H + H) = 1` **exactly** for any
/// non-constant image.
///
/// Unlike a constant volume — which returns the literal `1.0` from the
/// documented `H(X) + H(Y) == 0` early return without touching the histogram —
/// this input spreads over the histogram's full intensity range and exercises
/// the joint-histogram, marginalisation and entropy code paths in full.
#[test]
fn nmi_of_a_non_constant_volume_with_itself_is_one() {
    let metric = MutualInformationMetric::default();
    let volume = smooth_volume([12, 12, 12], [0.0; 3]);
    let nmi = metric.compute(&volume, &volume);

    assert!(
        (nmi - 1.0).abs() < NMI_ROUNDING_TOLERANCE,
        "NMI(X, X) must be exactly 1.0 for a non-constant volume, got {nmi:.15}"
    );
}

/// `MI` is symmetric: swapping the arguments transposes the joint histogram,
/// which exchanges the two marginals and leaves `H(X,Y)` unchanged, so both
/// `MI = H(X)+H(Y)−H(X,Y)` and the normaliser `H(X)+H(Y)` are invariant.
///
/// A defect that marginalises over the wrong histogram axis, or that indexes
/// the joint histogram as `[moving, fixed]` in one place and `[fixed, moving]`
/// in another, breaks this identity.
#[test]
fn nmi_is_symmetric_in_its_arguments() {
    let metric = MutualInformationMetric::default();
    let dims = [12, 12, 12];
    let a = smooth_volume(dims, [0.0; 3]);
    let b = smooth_volume(dims, [1.3, -0.7, 2.1]);

    let forward = metric.compute(&a, &b);
    let reverse = metric.compute(&b, &a);

    assert!(
        (forward - reverse).abs() < NMI_ROUNDING_TOLERANCE,
        "NMI must be symmetric: NMI(A,B) = {forward:.15}, NMI(B,A) = {reverse:.15}"
    );
    // Guard against the identity holding trivially because both sides collapsed
    // to a degenerate constant: a misaligned pair must still share information.
    assert!(
        forward > 0.1,
        "misaligned smooth volumes must retain measurable shared information, got {forward:.6}"
    );
}

/// Mutual information is invariant under a bijective intensity remapping — the
/// property that distinguishes it from correlation-based metrics and the reason
/// it is the metric of choice for multi-modal registration.
///
/// Bin inversion `b ↦ 31 − b` is a bijection on the 32-bin alphabet, so it
/// preserves the marginal entropy (`H(σ(X)) = H(X)`) and maps the diagonal
/// joint histogram of `(X, X)` onto the anti-diagonal without merging or
/// splitting any cell.  Hence `MI(X, σ(X)) = H(X)` and `NMI = 1` exactly, even
/// though the two volumes are perfectly *anti*-correlated (NCC would give −1
/// and MSE would be maximal).
#[test]
fn nmi_is_invariant_under_a_bijective_intensity_remapping() {
    let metric = MutualInformationMetric::default();
    let dims = [10, 10, 10];
    // Deterministic pattern occupying 8 of the 32 bins with unequal counts, so
    // H(X) > 0 and the marginal is non-uniform.
    let bins: Vec<usize> = (0..dims[0] * dims[1] * dims[2])
        .map(|index| (index * index / 3 + index) % 8)
        .collect();
    let direct = Array3::from_vec(dims, bins.iter().map(|&b| bin_centre(b)).collect())
        .expect("dimensions match the generated element count");
    let inverted = Array3::from_vec(dims, bins.iter().map(|&b| bin_centre(31 - b)).collect())
        .expect("dimensions match the generated element count");

    let nmi = metric.compute(&direct, &inverted);

    assert!(
        (nmi - 1.0).abs() < NMI_ROUNDING_TOLERANCE,
        "NMI must be invariant under the bijective bin remapping b ↦ 31−b, got {nmi:.15}"
    );
}

/// Statistically independent fields carry zero mutual information.
///
/// The construction makes independence *exact*, not asymptotic, so no
/// finite-sample estimator bias enters and the tolerance is pure floating-point
/// rounding.  `a` varies only along x and `b` only along y, so for every bin
/// pair `(i, j)` the joint count is `nz · nₐ(i) · n_b(j)`, giving
///   `p(i,j) = nₐ(i)·n_b(j) / (nx·ny) = p(i)·p(j)`
/// identically — the plug-in estimate of `MI` is exactly zero rather than the
/// usual `(Bₓ−1)(B_y−1)/2N` positive bias of a randomly sampled independent pair.
#[test]
fn nmi_of_exactly_independent_fields_is_zero() {
    let metric = MutualInformationMetric::default();
    let dims = [6usize, 8, 10];
    let [nz, ny, nx] = dims;
    // 5 x-bins × 2 voxels each, and 4 y-bins × 2 voxels each, so both marginals
    // are uniform and every (bin_a, bin_b) combination is realised.
    let a = Array3::from_vec(
        dims,
        (0..nz * ny * nx)
            .map(|index| bin_centre(index % nx % 5))
            .collect(),
    )
    .expect("dimensions match the generated element count");
    let b = Array3::from_vec(
        dims,
        (0..nz * ny * nx)
            .map(|index| bin_centre(10 + (index / nx) % ny % 4))
            .collect(),
    )
    .expect("dimensions match the generated element count");

    let nmi = metric.compute(&a, &b);

    assert!(
        nmi.abs() < NMI_ROUNDING_TOLERANCE,
        "NMI of exactly independent fields must be 0, got {nmi:.15}"
    );
}

/// Mutual information peaks at alignment and decays monotonically as the
/// misalignment grows — the property that makes it usable as a registration
/// objective at all.  A metric that is flat, non-monotonic, or maximised away
/// from zero shift would steer the hill-climb in `registration.rs` to the wrong
/// optimum while still satisfying every bound-style assertion.
///
/// Shifts are sampled analytically (the volume is regenerated at `c + s`), so no
/// resampling error is folded into the comparison.  The ladder stays far below
/// half the shortest sinusoid period (37 voxels), so the decay cannot wrap
/// around into a periodic re-alignment — with grid-scale periods it does: a
/// 10³ volume built from periods 7/9/11 reverses between shift 2 and 3
/// (NMI 0.375 → 0.430) because the pattern partially re-aligns with itself.
/// The step of 2 voxels keeps each rung's intensity change (`2A·sin(πs/L)`,
/// 11.7 units at `s = 2`) above the 7.97-unit bin width, so consecutive rungs
/// are separated by more than the histogram's quantisation.
#[test]
fn nmi_decreases_as_misalignment_grows() {
    let metric = MutualInformationMetric::default();
    let dims = [12, 12, 12];
    let reference = smooth_volume(dims, [0.0; 3]);

    let shifts = [0.0_f64, 2.0, 4.0, 6.0];
    let scores: Vec<f64> = shifts
        .iter()
        .map(|&shift| metric.compute(&reference, &smooth_volume(dims, [shift; 3])))
        .collect();

    assert!(
        (scores[0] - 1.0).abs() < NMI_ROUNDING_TOLERANCE,
        "zero shift must reproduce the self-NMI of 1.0, got {:.15}",
        scores[0]
    );
    for (rung, pair) in scores.windows(2).enumerate() {
        assert!(
            pair[1] < pair[0],
            "NMI must decrease from shift {} to {} voxels: {:.6} -> {:.6}",
            shifts[rung],
            shifts[rung + 1],
            pair[0],
            pair[1]
        );
    }
}

/// The documented degenerate branch: when both images are constant every
/// entropy is zero and `NMI = 2·MI/0` is undefined, so `compute` short-circuits
/// to `1.0` when the two constants share a histogram bin and `0.0` otherwise.
///
/// This covers the early return explicitly rather than letting a constant-volume
/// input masquerade as a test of the mutual-information computation, which it
/// never reaches.
#[test]
fn nmi_of_constant_volumes_takes_the_documented_degenerate_branch() {
    let metric = MutualInformationMetric::default();
    let low = Array3::from_elem([10, 10, 10], 100.0);
    let high = Array3::from_elem([10, 10, 10], 200.0);

    assert_eq!(
        metric.compute(&low, &low),
        1.0,
        "equal constant volumes share a bin and must return exactly 1.0"
    );
    assert_eq!(
        metric.compute(&low, &high),
        0.0,
        "constant volumes in different bins must return exactly 0.0"
    );
}

#[test]
fn intensity_registration_reports_final_transform_metric() {
    let volume = Array3::from_vec([3, 3, 3], (0..27).map(|value| value as f64 * 8.0).collect())
        .expect("infallible: validated precondition");
    let initial = crate::types::AffineTransform::new([
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let config = ClassicalConfig {
        max_iterations: 0,
        ..ClassicalConfig::default()
    };
    let metric = MutualInformationMetric::default();
    let registration = ImageRegistration::with_config(config, metric.clone());
    let transformed = crate::classical::spatial::apply_transform(&volume, &initial);
    let expected = metric.compute(&transformed, &volume);
    let untransformed = metric.compute(&volume, &volume);

    let rigid = registration
        .rigid_registration_mutual_info(&volume, &volume, &initial)
        .expect("infallible: validated precondition");
    let affine = registration
        .affine_registration_mutual_info(&volume, &volume, &initial)
        .expect("infallible: validated precondition");

    assert_eq!(rigid.quality.mutual_information, expected);
    assert_eq!(affine.quality.mutual_information, expected);
    assert_ne!(expected, untransformed);
}

#[test]
fn rigid_and_affine_registration_reject_invalid_step_multiplier() {
    let volume = Array3::from_elem([3, 3, 3], 1.0);
    for invalid_multiplier in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid_config = ClassicalConfig {
            max_iterations: 1,
            step_multiplier: invalid_multiplier,
            ..ClassicalConfig::default()
        };
        let registration =
            ImageRegistration::with_config(invalid_config, MutualInformationMetric::default());

        let rigid_error = registration
            .rigid_registration_mutual_info(
                &volume,
                &volume,
                &crate::types::AffineTransform::IDENTITY,
            )
            .expect_err("invalid rigid step multiplier must be rejected");
        assert!(rigid_error
            .to_string()
            .contains("rigid step_multiplier must be finite and positive"));

        let affine_error = registration
            .affine_registration_mutual_info(
                &volume,
                &volume,
                &crate::types::AffineTransform::IDENTITY,
            )
            .expect_err("invalid affine step multiplier must be rejected");
        assert!(affine_error
            .to_string()
            .contains("affine step_multiplier must be finite and positive"));
    }
}

#[test]
fn translation_mutual_information_recovers_known_shift() {
    let fixed = Array3::from_vec(
        [5, 5, 5],
        (0..125)
            .map(|index| {
                let z = index / 25;
                let y = (index / 5) % 5;
                let x = index % 5;
                f64::from((z * z + 3 * y + 7 * x) as u32)
            })
            .collect(),
    )
    .expect("infallible: validated precondition");
    let generating_transform = crate::types::AffineTransform::new([
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let moving = crate::classical::spatial::apply_transform(&fixed, &generating_transform);
    let metric = MutualInformationMetric::new(16, 0.0, 60.0);
    let initial_similarity = metric.compute(&moving, &fixed);
    let registration = ImageRegistration::with_config(
        ClassicalConfig {
            max_iterations: 4,
            tolerance: 0.0,
            step_multiplier: 1.0,
        },
        metric,
    );

    let result = registration
        .translation_registration_mutual_info(
            &moving,
            &fixed,
            &crate::types::AffineTransform::IDENTITY,
        )
        .expect("infallible: validated precondition");

    assert_eq!(result.transform.0[3], -1.0);
    assert_eq!(result.transform.0[7], 0.0);
    assert_eq!(result.transform.0[11], 0.0);
    assert!(result.quality.mutual_information > initial_similarity);
}

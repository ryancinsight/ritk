use std::num::NonZeroU8;

use super::*;
use crate::dti::{DtiConfig, estimate_dti};
use crate::test_support::{
    add_rician_noise, dti_signal, mean, rmse, scheme, schemes_with_references, seeded_rng,
};

/// A prolate white-matter tensor aligned with `+z`.
///
/// Eigenvalues are the conventional corpus-callosum figures — `λ₁ ≈ 1.7`,
/// `λ₂ = λ₃ ≈ 0.25`, in units of `10⁻³ mm²/s` — which put FA near 0.8 and make
/// the attenuation strongly direction dependent. That dependence is the point:
/// it is what gives the log-transformed measurements unequal variances, which
/// is the thing weighting corrects.
const WHITE_MATTER: [f64; 6] = [0.25e-3, 0.25e-3, 1.7e-3, 0.0, 0.0, 0.0];

/// Signal-to-noise ratio of the unweighted reference.
///
/// At `b = 1000 s/mm²` the along-fibre attenuation is `exp(−1.7) ≈ 0.18`, so
/// the least reliable measurement still sits near SNR 5 where the Rician
/// distribution is close enough to Gaussian that the log-transform bias — not
/// the noise floor — dominates. Below that the comparison would be measuring
/// the rectification instead of the estimator.
const BASELINE_SNR: f64 = 30.0;

/// Enough realisations that the RMSE difference between two estimators is
/// larger than the sampling error of the RMSE itself.
const TRIALS: usize = 400;

fn ordinary() -> DtiConfig {
    DtiConfig::default().with_fit(TensorFit::Ordinary)
}

fn weighted(passes: u8) -> DtiConfig {
    DtiConfig::default().with_fit(TensorFit::Weighted {
        reweight_passes: NonZeroU8::new(passes).expect("nonzero pass count"),
    })
}

/// Truth values implied by [`WHITE_MATTER`].
fn truth() -> (f64, f64, f64) {
    let eigenvalues = [1.7e-3, 0.25e-3, 0.25e-3];
    (
        crate::dti::invariants::fractional_anisotropy(eigenvalues),
        crate::dti::invariants::mean_diffusivity(eigenvalues),
        eigenvalues[0],
    )
}

/// Fit `TRIALS` noisy realisations and return the recovered `(FA, MD, λ₁)`.
///
/// Both estimators are driven from the same seed, so they see the identical
/// noise realisations and the comparison isolates the estimator rather than the
/// draw.
fn sweep(config: DtiConfig) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let scheme = schemes_with_references(30, 5);
    let clean = dti_signal(&scheme, WHITE_MATTER, 1.0);
    let sigma = 1.0 / BASELINE_SNR;
    let mut rng = seeded_rng(0x5eed_d71f_17a3);

    let mut fa = Vec::with_capacity(TRIALS);
    let mut md = Vec::with_capacity(TRIALS);
    let mut axial = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let noisy = add_rician_noise(&clean, sigma, &mut rng);
        let Ok(tensor) = estimate_dti(&scheme, &noisy, config) else {
            continue;
        };
        fa.push(tensor.fa());
        md.push(tensor.md());
        axial.push(tensor.ad());
    }
    assert!(
        fa.len() * 100 >= TRIALS * 95,
        "at SNR {BASELINE_SNR} nearly every realisation must fit; got {} of {TRIALS}",
        fa.len()
    );
    (fa, md, axial)
}

// ── The weighting claim ──────────────────────────────────────────────────

/// The estimator's reason to exist: on noisy data, weighting the log-linear
/// system by the predicted signal recovers the tensor more accurately than
/// leaving it unweighted.
///
/// This is the falsifiable form of the module's derivation. `var(ln Sᵢ) ∝ 1/Ŝᵢ²`
/// means ordinary least squares over-trusts the strongly attenuated
/// measurements; if that reasoning is right, restoring the inverse-variance
/// weights must reduce the error of the recovered anisotropy over a sample of
/// noise realisations. If it did not, the weighting would be ceremony.
#[test]
fn weighting_reduces_anisotropy_error_under_noise() {
    let (fa_truth, md_truth, axial_truth) = truth();

    let (fa_ols, md_ols, axial_ols) = sweep(ordinary());
    let (fa_wls, md_wls, axial_wls) = sweep(weighted(1));

    let fa_error = (rmse(&fa_ols, fa_truth), rmse(&fa_wls, fa_truth));
    let md_error = (rmse(&md_ols, md_truth), rmse(&md_wls, md_truth));
    let axial_error = (rmse(&axial_ols, axial_truth), rmse(&axial_wls, axial_truth));

    assert!(
        fa_error.1 < fa_error.0,
        "weighted least squares must recover FA more accurately than ordinary: \
         RMSE ordinary {:.5} vs weighted {:.5}",
        fa_error.0,
        fa_error.1
    );
    assert!(
        axial_error.1 < axial_error.0,
        "weighted least squares must recover the axial diffusivity more \
         accurately than ordinary: RMSE ordinary {:.3e} vs weighted {:.3e}",
        axial_error.0,
        axial_error.1
    );
    assert!(
        md_error.1 <= md_error.0 * 1.02,
        "weighting must not degrade mean diffusivity: \
         RMSE ordinary {:.3e} vs weighted {:.3e}",
        md_error.0,
        md_error.1
    );
}

/// Where the gain actually is: weighting narrows the *spread* of the estimate.
///
/// It is worth separating this from bias, because the two are different claims
/// and only one of them holds. Generalised least squares is the minimum-variance
/// unbiased estimator of a linear model — its guarantee is about variance — and
/// that is what shows up: over the same noise realisations, the weighted fit's
/// standard deviation in the axial diffusivity is materially smaller than the
/// ordinary fit's.
///
/// The systematic error does *not* shrink, and this test asserts that honestly
/// rather than claiming an improvement the estimator does not deliver. Measured
/// at SNR 30, both estimators sit within a fraction of a percent of truth, with
/// the residual offset an order of magnitude or more below their own spread, so
/// neither is bias limited here. The remaining offset is the Rician
/// rectification — the magnitude reconstruction folds noise upward, which no
/// weighting of a *log-linear* system can undo, because the bias is in the data
/// before the log is taken. Removing it needs an estimator that models the
/// Rician likelihood; that is a different model, not a different weighting.
#[test]
fn weighting_narrows_the_spread_without_claiming_a_bias_correction() {
    let (_, _, axial_truth) = truth();
    let (_, _, axial_ols) = sweep(ordinary());
    let (_, _, axial_wls) = sweep(weighted(1));

    let ordinary_spread = rmse(&axial_ols, mean(&axial_ols));
    let weighted_spread = rmse(&axial_wls, mean(&axial_wls));
    assert!(
        weighted_spread < ordinary_spread,
        "weighting must narrow the axial-diffusivity spread:          sd ordinary {ordinary_spread:.3e} vs weighted {weighted_spread:.3e}"
    );

    // Neither estimator is bias limited: what is left after averaging is small
    // against both the truth and the estimator's own spread.
    for (label, estimates, spread) in [
        ("ordinary", &axial_ols, ordinary_spread),
        ("weighted", &axial_wls, weighted_spread),
    ] {
        let bias = (mean(estimates) - axial_truth).abs();
        assert!(
            bias < 0.01 * axial_truth,
            "{label} bias {bias:.3e} must stay under 1% of the true              axial diffusivity {axial_truth:.3e}"
        );
        assert!(
            bias < spread,
            "{label} bias {bias:.3e} must stay below its own spread              {spread:.3e}; if it did not, the comparison would be measuring              the Rician floor rather than the weighting"
        );
    }
}

/// A second reweighting pass changes the answer by far less than the first.
///
/// The documented recommendation is a single pass, which is only sound if the
/// iteration has essentially converged there. Comparing the two step sizes
/// tests that claim rather than assuming it.
#[test]
fn reweighting_converges_after_the_first_pass() {
    let (fa_truth, _, _) = truth();
    let (fa_ols, _, _) = sweep(ordinary());
    let (fa_one, _, _) = sweep(weighted(1));
    let (fa_two, _, _) = sweep(weighted(2));

    let first_step = (rmse(&fa_ols, fa_truth) - rmse(&fa_one, fa_truth)).abs();
    let second_step = (rmse(&fa_one, fa_truth) - rmse(&fa_two, fa_truth)).abs();

    assert!(
        second_step < first_step,
        "the second reweighting must move the estimate less than the first: \
         first {first_step:.3e}, second {second_step:.3e}"
    );
}

// ── Consistency of the two estimators where they must agree ──────────────

/// On noiseless data the system is consistent, so every positive weighting has
/// the same exact solution.
///
/// This is what makes the weighted form a safe default: it changes the answer
/// only where the answer was uncertain.
#[test]
fn estimators_agree_exactly_on_noiseless_data() {
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, WHITE_MATTER, 1_000.0);

    let ordinary_tensor = estimate_dti(&scheme, &signals, ordinary()).expect("ordinary fit");
    let weighted_tensor = estimate_dti(&scheme, &signals, weighted(1)).expect("weighted fit");

    for (index, (from_ordinary, from_weighted)) in ordinary_tensor
        .elements()
        .iter()
        .zip(weighted_tensor.elements())
        .enumerate()
    {
        assert!(
            (from_ordinary - from_weighted).abs() < 1.0e-14,
            "element {index}: ordinary {from_ordinary:.12e} vs weighted {from_weighted:.12e}"
        );
    }
}

/// The configuration reports which estimator produced a tensor.
#[test]
fn fitted_tensor_records_its_estimator() {
    let scheme = scheme(20);
    let signals = dti_signal(&scheme, WHITE_MATTER, 1_000.0);

    let tensor = estimate_dti(&scheme, &signals, ordinary()).expect("fit");
    assert_eq!(tensor.fit(), TensorFit::Ordinary);
    assert_eq!(tensor.fit().reweight_passes(), 0);

    let tensor = estimate_dti(&scheme, &signals, weighted(3)).expect("fit");
    assert_eq!(tensor.fit().reweight_passes(), 3);
}

/// The default estimator is the weighted one.
#[test]
fn default_estimator_is_weighted() {
    assert_eq!(
        TensorFit::default(),
        TensorFit::Weighted {
            reweight_passes: NonZeroU8::MIN
        }
    );
    assert_eq!(DtiConfig::default().fit(), TensorFit::default());
}

// ── Row scaling ──────────────────────────────────────────────────────────

/// Scaling a row by `√w` is what turns the unweighted solver into a weighted
/// one, so the scaling itself is checked directly.
#[test]
fn row_scaling_multiplies_both_sides() {
    let mut design = Array2::zeros([2, 2]);
    design[[0, 0]] = 1.0;
    design[[0, 1]] = 2.0;
    design[[1, 0]] = 3.0;
    design[[1, 1]] = 4.0;
    let mut rhs = Array1::zeros([2]);
    rhs[0] = 5.0;
    rhs[1] = 6.0;

    let (scaled_design, scaled_rhs) = scale_rows(&design, &rhs, &[2.0, 0.5]);

    assert_eq!(scaled_design[[0, 0]], 2.0);
    assert_eq!(scaled_design[[0, 1]], 4.0);
    assert_eq!(scaled_design[[1, 0]], 1.5);
    assert_eq!(scaled_design[[1, 1]], 2.0);
    assert_eq!(scaled_rhs[0], 10.0);
    assert_eq!(scaled_rhs[1], 3.0);
}

/// A weight is a signal amplitude, so no row may be scaled out of the system or
/// given more say than the unattenuated baseline.
#[test]
fn predicted_weights_stay_within_the_attenuation_range() {
    let mut design = Array2::zeros([3, 1]);
    // Rows chosen to drive the exponent far negative, near zero, and positive.
    design[[0, 0]] = -50.0;
    design[[1, 0]] = -0.5;
    design[[2, 0]] = 50.0;
    let mut solution = Array1::zeros([1]);
    solution[0] = 1.0;

    let scales = predicted_relative_signals(&design, &solution);

    assert_eq!(scales[0], MINIMUM_RELATIVE_PREDICTION);
    assert!((scales[1] - (-0.5_f64).exp()).abs() < 1.0e-15);
    assert_eq!(scales[2], 1.0);
}

/// The reported residual is the unweighted one whichever estimator ran, so the
/// number is comparable across configurations.
#[test]
fn residual_is_measured_against_the_unweighted_system() {
    let mut design = Array2::zeros([2, 1]);
    design[[0, 0]] = 1.0;
    design[[1, 0]] = 1.0;
    let mut solution = Array1::zeros([1]);
    solution[0] = 1.0;
    let mut log_signals = Array1::zeros([2]);
    log_signals[0] = 0.0;
    log_signals[1] = 2.0;

    // Residuals are (1 − 0) and (1 − 2), so the norm is √2.
    let norm = residual_norm(&design, &solution, &log_signals);
    assert!((norm - std::f64::consts::SQRT_2).abs() < 1.0e-15);
}

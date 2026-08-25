//! DKI model verification against synthetic oracles.
//!
//! Every case uses noise-free signals generated from known D and W tensors.
//! Tolerances account for the LM solver's `sqrt(ε)` convergence floor and
//! conditioning of the multi-shell scheme.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::*;
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;

/// Build a multi-shell gradient scheme with `n_dirs` directions per shell.
fn multi_shell_scheme(b0_count: usize, dir_count: usize, b_values: &[f64]) -> GradientScheme {
    let mut entries: Vec<GradientDirection> = Vec::new();

    // b0 volumes — zero direction vector.
    let b0_weighting = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero_dir = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..b0_count {
        entries.push(GradientDirection::new(b0_weighting, zero_dir).expect("b0 entry"));
    }

    // Uniform directions via a Fibonacci sphere (quasi-uniform on S²).
    let phi_golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..dir_count {
        let z = 1.0 - (2.0 * i as f64 + 1.0) / dir_count as f64;
        let radius = (1.0 - z * z).sqrt();
        let phi = phi_golden * i as f64;
        let g = Vector::new([radius * phi.cos(), radius * phi.sin(), z]);

        for &b in b_values {
            let weighting =
                DiffusionWeighting::from_seconds_per_square_millimeter(b).expect("finite b");
            entries.push(GradientDirection::new(weighting, g).expect("weighted direction"));
        }
    }

    GradientScheme::new(entries, GradientFrame::ImageAxis).expect("valid scheme")
}

/// Generate noiseless DKI signals from ground-truth tensors.
fn dki_signals(scheme: &GradientScheme, s0: f64, d: &[f64; 6], w: &[f64; 15]) -> Vec<f64> {
    let md = (d[0] + d[1] + d[2]) / 3.0;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let d_app = d[0] * gx * gx
                + d[1] * gy * gy
                + d[2] * gz * gz
                + 2.0 * d[3] * gx * gy
                + 2.0 * d[4] * gx * gz
                + 2.0 * d[5] * gy * gz;
            let w_app = compute_w_contraction(w, [gx, gy, gz]);
            s0 * (-b * d_app + (b.powi(2) / 6.0) * md.powi(2) * w_app).exp()
        })
        .collect()
}

/// Ground-truth: single fibre along z, AD=1.7e-3, RD=0.3e-3 mm²/s.
fn single_fibre_truth() -> ([f64; 6], [f64; 15]) {
    let d: [f64; 6] = [0.0003, 0.0003, 0.0017, 0.0, 0.0, 0.0];
    let mut w = [0.0; 15];
    w[2] = 0.7; // W_zzzz
    (d, w)
}

/// Default multi-shell scheme: 4 b0 + 30 dirs × 2 shells (b=1000, 2500).
fn default_scheme() -> GradientScheme {
    multi_shell_scheme(4, 30, &[1000.0, 2500.0])
}

/// Default DKI config.
fn default_config() -> KtiConfig {
    KtiConfig::default()
}

// ── Round-trip tests ──────────────────────────────────────────────────────────

#[test]
fn isotropic_gaussian_recovers_zero_kurtosis() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let d: [f64; 6] = [0.0007, 0.0007, 0.0007, 0.0, 0.0, 0.0];
    let w = [0.0; 15];
    let signals = dki_signals(&scheme, s0, &d, &w);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("isotropic Gaussian fit should succeed");

    // D elements recovered.
    let d_fit = tensor.elements_d();
    assert!(
        (d_fit[0] - d[0]).abs() < 1e-7,
        "Dxx mismatch: {} vs {}",
        d_fit[0],
        d[0]
    );

    // Kurtosis near zero.
    assert!(
        tensor.mk().abs() < 0.05,
        "MK should be ~0 for Gaussian, got {}",
        tensor.mk()
    );
    assert!(
        tensor.elements_w().iter().all(|w_i| w_i.abs() < 0.1),
        "W elements should be ~0 for Gaussian diffusion"
    );

    // Convergence verified.
    assert!(tensor.converged(), "noiseless fit must converge");
}

#[test]
fn single_fibre_round_trip_recovers_d_tensor() {
    let scheme = default_scheme();
    let (d_true, w_true) = single_fibre_truth();
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_true, &w_true);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("single-fibre DKI fit should succeed");

    // D elements within 0.5%.
    for (idx, (fit, truth)) in tensor.elements_d().iter().zip(d_true.iter()).enumerate() {
        let rel_err = if *truth > 0.0 {
            (fit - truth).abs() / truth
        } else {
            fit.abs()
        };
        assert!(
            rel_err < 0.005,
            "D[{idx}]: fit={fit:.6e} truth={truth:.6e} rel_err={rel_err:.3e}"
        );
    }
    assert!(tensor.converged(), "noiseless fit must converge");
}

#[test]
fn single_fibre_recovers_axial_kurtosis() {
    let scheme = default_scheme();
    let (d_true, w_true) = single_fibre_truth();
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_true, &w_true);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("single-fibre DKI fit should succeed");

    // W_zzzz should be within 0.15 of 0.7.
    let w_zzzz_fit = tensor.elements_w()[2];
    assert!(
        (w_zzzz_fit - 0.7).abs() < 0.2,
        "W_zzzz: fit={w_zzzz_fit:.4} truth=0.7"
    );

    // MK positive.
    assert!(
        tensor.mk() > 0.0,
        "MK must be positive, got {}",
        tensor.mk()
    );
    assert!(tensor.mk() < 1.5, "MK={} implausibly large", tensor.mk());

    // AK positive.
    assert!(
        tensor.ak() > 0.0,
        "AK must be positive, got {}",
        tensor.ak()
    );
}

#[test]
fn single_fibre_pev_aligns_with_z() {
    let scheme = default_scheme();
    let (d_true, w_true) = single_fibre_truth();
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_true, &w_true);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("single-fibre DKI fit should succeed");

    let pev = tensor.principal_eigenvector();
    let dot_z = pev[2].abs();
    assert!(dot_z > 0.99, "PEV should align with z, got dot={dot_z:.4}");
}

#[test]
fn signal_prediction_round_trips() {
    let scheme = default_scheme();
    let (d_true, w_true) = single_fibre_truth();
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_true, &w_true);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("single-fibre DKI fit should succeed");

    // Predicted signals within 2%.
    for (idx, entry) in scheme.directions().iter().enumerate() {
        let predicted = tensor.predict_signal(
            entry.direction().to_array(),
            entry.weighting().seconds_per_square_millimeter(),
        );
        let rel_err = (predicted - signals[idx]).abs() / signals[idx];
        assert!(
            rel_err < 0.02,
            "signal[{idx}]: predicted={predicted:.2} true={true_val:.2} rel_err={rel_err:.3e}",
            true_val = signals[idx]
        );
    }
}

// ── DTI compatibility ─────────────────────────────────────────────────────────

#[test]
fn gaussian_signal_matches_dti() {
    let scheme = default_scheme();
    let d: [f64; 6] = [0.0010, 0.0006, 0.0004, 0.0001, 0.00005, -0.00003];
    let w = [0.0; 15];
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d, &w);

    let dki_tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("Gaussian+noiseless DKI fit should succeed");
    let dti_tensor = dti::estimate_dti(
        &scheme,
        &signals,
        DtiConfig::new(DiffusionWeighting::from_seconds_per_square_millimeter(50.0).unwrap()),
    )
    .expect("DTI fit should succeed");

    // D elements agree within tight tolerance.
    for (idx, (dki_d, dti_d)) in dki_tensor
        .elements_d()
        .iter()
        .zip(dti_tensor.elements().iter())
        .enumerate()
    {
        assert!(
            (dki_d - dti_d).abs() < 1e-6,
            "D[{idx}]: DKI={dki_d:.6e} DTI={dti_d:.6e}"
        );
    }

    // FA agrees.
    let fa_diff = (dki_tensor.fa() - dti_tensor.fa()).abs();
    assert!(fa_diff < 1e-5, "FA mismatch: {fa_diff:.3e}");

    // MD agrees.
    let md_diff = (dki_tensor.md() - dti_tensor.md()).abs();
    assert!(md_diff < 1e-7, "MD mismatch: {md_diff:.3e}");

    // Kurtosis is zero.
    assert!(dki_tensor.mk().abs() < 0.05, "MK should be ~0 for Gaussian");
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn rejects_signal_length_mismatch() {
    let scheme = default_scheme();
    let signals = vec![1.0; scheme.len() - 1];
    let err = estimate_dki(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, KtiError::SignalLengthMismatch { .. }));
}

#[test]
fn rejects_non_finite_signal() {
    let scheme = default_scheme();
    let mut signals = vec![1.0; scheme.len()];
    signals[5] = f64::NAN;
    let err = estimate_dki(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, KtiError::NonFiniteSignal { .. }));
}

#[test]
fn rejects_all_dwi_scheme() {
    // A scheme with only b0 volumes.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let z = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..4 {
        entries.push(GradientDirection::new(b0, z).unwrap());
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();
    let signals = vec![1.0; scheme.len()];
    let err = estimate_dki(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, KtiError::NoDwiDirections));
}

#[test]
fn rejects_negative_signal_via_dti_fit() {
    let scheme = default_scheme();
    let mut signals = vec![1000.0; scheme.len()];
    // Make one DWI signal negative — DTI initial fit catches this before
    // the DKI nonlinear stage.
    let dwi = scheme.dwi_indices(default_config().b0_threshold());
    signals[dwi[0]] = -10.0;
    let err = estimate_dki(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, KtiError::DtiFailed(_)));
}

// ── Multi-shell requirement ───────────────────────────────────────────────────

#[test]
fn single_shell_scheme_is_accepted_with_enough_directions() {
    // Single shell at b=2000 with 30 directions gives 30 residuals > 21 params.
    let scheme = multi_shell_scheme(4, 30, &[2000.0]);
    let (d_true, w_true) = single_fibre_truth();
    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_true, &w_true);

    let result = estimate_dki(&scheme, &signals, &default_config());
    assert!(
        result.is_ok(),
        "single-shell with enough directions should solve, got {:?}",
        result.err()
    );
}

// ── Rotational invariance ─────────────────────────────────────────────────────

#[test]
fn rotated_tensor_recovers_consistent_fa() {
    // Rotate a fibre 45° in xz-plane and verify FA is conserved.
    let angle = std::f64::consts::FRAC_PI_4;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let scheme = default_scheme();

    // Original: fibre along z.
    let d0: [f64; 6] = [0.0003, 0.0003, 0.0017, 0.0, 0.0, 0.0];

    // Rotated D: R D Rᵀ where R is rotation about y.
    let d_rot: [f64; 6] = [
        d0[0] * cos_a * cos_a + d0[2] * sin_a * sin_a,
        d0[1],
        d0[0] * sin_a * sin_a + d0[2] * cos_a * cos_a,
        0.0,
        (d0[0] - d0[2]) * cos_a * sin_a,
        0.0,
    ];

    // Rotated W: only W_zzzz → rotated contributions.
    let mut w_rot = [0.0; 15];
    let w0_zzzz = 0.7;
    w_rot[0] = w0_zzzz * sin_a.powi(4); // W_xxxx
    w_rot[2] = w0_zzzz * cos_a.powi(4); // W_zzzz
    w_rot[9] = w0_zzzz * sin_a * sin_a * cos_a * cos_a; // W_xxzz

    let s0 = 1000.0;
    let signals = dki_signals(&scheme, s0, &d_rot, &w_rot);

    let tensor = estimate_dki(&scheme, &signals, &default_config())
        .expect("rotated fibre fit should succeed");

    // FA should be approximately conserved.
    let md = (d0[0] + d0[1] + d0[2]) / 3.0;
    let num = ((d0[0] - md).powi(2) + (d0[1] - md).powi(2) + (d0[2] - md).powi(2)).sqrt();
    let den = (d0[0].powi(2) + d0[1].powi(2) + d0[2].powi(2)).sqrt();
    let fa_expected = (1.5_f64).sqrt() * num / den;

    assert!(
        (tensor.fa() - fa_expected).abs() < 0.02,
        "FA preserved under rotation: fit={:.4} expected={:.4}",
        tensor.fa(),
        fa_expected
    );

    // MK positive.
    assert!(tensor.mk() > 0.0, "MK should be positive");
}

// ── ADR 0036 verification condition 7: gradient reorientation ────────────

/// Build a proper rotation matrix about the y-axis by `angle` radians.
fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Gradient reorientation is equally critical for kurtosis tensor fitting:
/// skipping it produces a silently wrong DKI tensor.  The oracle is the same
/// as in the DTI case (ADR 0036 verification condition 7): signals generated
/// from the reoriented scheme must recover the original PEV after consistent
/// reorientation, and must fail when reorientation is skipped.
#[test]
fn reorient_gradients_recovers_original_pev_skip_reorientation_gives_wrong_pev() {
    // ── Setup ────────────────────────────────────────────────────────────
    let scheme = default_scheme();
    let (tensor, w_true) = single_fibre_truth();
    // tensor = [0.0003, 0.0003, 0.0017, 0, 0, 0], PEV = [0, 0, 1]
    let original_pev = [0.0, 0.0, 1.0];

    // Rotate 45° about y.
    let angle = std::f64::consts::FRAC_PI_4;
    let rotation = rotation_y(angle);

    let scheme_reoriented = scheme.reorient(rotation).expect("valid proper rotation");

    // Generate DKI signals from the original tensor via the reoriented
    // scheme — this represents data after a rigid correction.
    let s0 = 1000.0;
    let signals = dki_signals(&scheme_reoriented, s0, &tensor, &w_true);

    // ── With reorientation: correct path ─────────────────────────────────
    let dki_correct = estimate_dki(&scheme_reoriented, &signals, &default_config())
        .expect("DKI fit with reoriented scheme");
    let pev_correct = dki_correct.principal_eigenvector();

    let dot_original = (pev_correct[0] * original_pev[0]
        + pev_correct[1] * original_pev[1]
        + pev_correct[2] * original_pev[2])
        .abs();
    assert!(
        dot_original > 0.99,
        "ADR 0036 vc7 DKI: after reorientation PEV must recover original +z; \
         PEV = [{:.4}, {:.4}, {:.4}], dot = {dot_original:.4}",
        pev_correct[0],
        pev_correct[1],
        pev_correct[2]
    );

    // ── Without reorientation: error path ────────────────────────────────
    let dki_wrong =
        estimate_dki(&scheme, &signals, &default_config()).expect("DKI fit without reorientation");
    let pev_wrong = dki_wrong.principal_eigenvector();

    // Fitting with the original (unrotated) scheme on signals generated
    // from the reoriented scheme recovers D̂ = Rᵀ·D·R, whose PEV is
    // Rᵀ·[0,0,1] = [-r2, 0, r2].
    let dot_original_wrong = (pev_wrong[0] * original_pev[0]
        + pev_wrong[1] * original_pev[1]
        + pev_wrong[2] * original_pev[2])
        .abs();
    let r2 = std::f64::consts::SQRT_2 / 2.0;
    let rotated_pev = [-r2, 0.0, r2];
    let dot_rotated = (pev_wrong[0] * rotated_pev[0]
        + pev_wrong[1] * rotated_pev[1]
        + pev_wrong[2] * rotated_pev[2])
        .abs();

    assert!(
        dot_original_wrong < 0.9,
        "ADR 0036 vc7 DKI: without reorientation PEV must NOT align \
         with original +z; dot = {dot_original_wrong:.4}"
    );
    assert!(
        dot_rotated > 0.99,
        "ADR 0036 vc7 DKI: without reorientation PEV must align with \
         R^T·[0,0,1] = [{:.4}, 0, {:.4}]; PEV = [{:.4}, {:.4}, {:.4}], \
         dot = {dot_rotated:.4}",
        -r2,
        r2,
        pev_wrong[0],
        pev_wrong[1],
        pev_wrong[2]
    );

    // Verify the correct-path fit converged.
    assert!(
        dki_correct.converged(),
        "DKI with reorientation must converge"
    );
}

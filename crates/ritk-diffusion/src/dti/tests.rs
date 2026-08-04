use super::*;
use ritk_diffusion_scheme::{GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

fn scheme(direction_count: usize) -> GradientScheme {
    let mut entries = vec![
        GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("valid b0"),
    ];
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for index in 0..direction_count {
        let z = 1.0 - 2.0 * (index as f64 + 0.5) / direction_count as f64;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * index as f64;
        entries.push(
            GradientDirection::new(
                weighting(1_000.0),
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .expect("unit Fibonacci direction"),
        );
    }
    GradientScheme::new(entries, GradientFrame::Lps).expect("valid scheme")
}

fn dti_signal(scheme: &GradientScheme, tensor_elements: [f64; 6], s0: f64) -> Vec<f64> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = tensor_elements;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let q = dxx * gx * gx
                + dyy * gy * gy
                + dzz * gz * gz
                + 2.0 * dxy * gx * gy
                + 2.0 * dxz * gx * gz
                + 2.0 * dyz * gy * gz;
            s0 * (-b * q).exp()
        })
        .collect()
}

// ── Round-trip tests ─────────────────────────────────────────────────────

#[test]
fn isotropic_tensor_round_trips() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let tensor = [0.0007, 0.0007, 0.0007, 0.0, 0.0, 0.0];
    let signals = dti_signal(&scheme, tensor, 1000.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;

    // Diffusivities should recover within 2%.
    let [dxx, dyy, dzz, dxy, dxz, dyz] = dti.elements();
    assert!((dxx - 0.0007).abs() < 2e-5, "Dxx = {dxx}");
    assert!((dyy - 0.0007).abs() < 2e-5, "Dyy = {dyy}");
    assert!((dzz - 0.0007).abs() < 2e-5, "Dzz = {dzz}");
    assert!(dxy.abs() < 2e-5);
    assert!(dxz.abs() < 2e-5);
    assert!(dyz.abs() < 2e-5);

    // FA should be ≈0 for isotropic tensor.
    assert!(dti.fa() < 0.01, "isotropic FA = {}", dti.fa());
    assert!((dti.md() - 0.0007).abs() < 1e-5);
    Ok(())
}

#[test]
fn anisotropic_tensor_round_trips() -> Result<(), DtiError> {
    let scheme = scheme(60);
    let tensor = [0.0017, 0.0003, 0.0003, 0.0, 0.0, 0.0];
    let signals = dti_signal(&scheme, tensor, 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;

    let [dxx, dyy, dzz, _dxy, _dxz, _dyz] = dti.elements();
    assert!((dxx - 0.0017).abs() < 5e-5);
    assert!((dyy - 0.0003).abs() < 5e-5);
    assert!((dzz - 0.0003).abs() < 5e-5);

    // FA should be high for a 1.7/0.3 prolate tensor.
    assert!(dti.fa() > 0.7, "anisotropic FA = {}", dti.fa());
    Ok(())
}

#[test]
fn rotated_tensor_round_trips() -> Result<(), DtiError> {
    // A tensor whose principal axis is at 45° in the xy-plane.
    let d_parallel = 0.0017;
    let d_perp = 0.0003;
    let cos45 = std::f64::consts::SQRT_2 / 2.0;
    let dxx = d_parallel * cos45.powi(2) + d_perp * (1.0 - cos45.powi(2));
    let dyy = d_parallel * (1.0 - cos45.powi(2)) + d_perp * cos45.powi(2);
    let dxy = (d_parallel - d_perp) * cos45 * cos45;
    let tensor = [dxx, dyy, d_perp, dxy, 0.0, 0.0];

    let scheme = scheme(60);
    let signals = dti_signal(&scheme, tensor, 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;

    let [tdxx, tdyy, tdzz, tdxy, tdxz, tdyz] = dti.elements();
    assert!((tdxx - dxx).abs() < 5e-5);
    assert!((tdyy - dyy).abs() < 5e-5);
    assert!((tdzz - d_perp).abs() < 5e-5);
    assert!((tdxy - dxy).abs() < 5e-5);
    assert!(tdxz.abs() < 5e-5);
    assert!(tdyz.abs() < 5e-5);
    Ok(())
}

// ── Scalar metrics ───────────────────────────────────────────────────────

#[test]
fn fa_of_isotropic_is_zero() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, [0.0007, 0.0007, 0.0007, 0.0, 0.0, 0.0], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    assert!(dti.fa() < 0.01, "isotropic FA should be near zero");
    Ok(())
}

#[test]
fn fa_of_maximally_anisotropic_is_near_one() -> Result<(), DtiError> {
    let scheme = scheme(60);
    let signals = dti_signal(&scheme, [0.002, 0.0001, 0.0001, 0.0, 0.0, 0.0], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    assert!(dti.fa() > 0.9, "highly anisotropic FA = {}", dti.fa());
    Ok(())
}

#[test]
fn md_is_trace_over_three() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let tensor = [0.0010, 0.0008, 0.0006, 0.0, 0.0, 0.0];
    let signals = dti_signal(&scheme, tensor, 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    let expected_md = (0.0010 + 0.0008 + 0.0006) / 3.0;
    assert!((dti.md() - expected_md).abs() < 1e-5);
    Ok(())
}

#[test]
fn pev_aligns_with_principal_axis() -> Result<(), DtiError> {
    let scheme = scheme(60);
    let signals = dti_signal(&scheme, [0.0017, 0.0003, 0.0003, 0.0, 0.0, 0.0], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    let pev = dti.principal_eigenvector();
    let x_component = pev[0].abs();
    assert!(
        x_component > 0.99,
        "PEV x = {x_component}, should be near 1"
    );
    Ok(())
}

#[test]
fn pev_aligns_with_diagonal_axis() -> Result<(), DtiError> {
    // Tensor with principal axis at (1,1,0)/√2.
    let ad = 0.0017;
    let rd = 0.0003;
    let s = 1.0 / 2.0_f64.sqrt();
    let dxx = ad * s * s + rd * (1.0 - s * s);
    let dyy = ad * s * s + rd * (1.0 - s * s);
    let dzz = rd;
    let dxy = (ad - rd) * s * s;

    let scheme = scheme(60);
    let signals = dti_signal(&scheme, [dxx, dyy, dzz, dxy, 0.0, 0.0], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    let pev = dti.principal_eigenvector();
    // PEV should be near (s, s, 0) or (-s, -s, 0).
    let dot = (pev[0] * s + pev[1] * s).abs();
    assert!(
        dot > 0.99,
        "PEV dot with diagonal = {dot}, should be near 1"
    );
    Ok(())
}

// ── Signal prediction ────────────────────────────────────────────────────

#[test]
fn predict_signal_matches_synthesis() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let tensor = [0.0012, 0.0005, 0.0005, 0.0001, 0.0, 0.0];
    let signals = dti_signal(&scheme, tensor, 100.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;

    let entry = &scheme.directions()[1];
    let dir = entry.direction().to_array();
    let b = entry.weighting().seconds_per_square_millimeter();
    let predicted = dti.predict_signal(dir, b);
    assert!((predicted - signals[1]).abs() < 1e-3);
    Ok(())
}

#[test]
fn quadratic_form_reconstructs_apparent_diffusivity() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let tensor = [0.0010, 0.0005, 0.0005, 0.0, 0.0, 0.0];
    let signals = dti_signal(&scheme, tensor, 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;

    // At (1,0,0): ADC should be ≈ Dxx.
    let adc_x = dti.quadratic_form([1.0, 0.0, 0.0]);
    assert!((adc_x - 0.0010).abs() < 1e-5);

    // At (0,1,0): ADC should be ≈ Dyy.
    let adc_y = dti.quadratic_form([0.0, 1.0, 0.0]);
    assert!((adc_y - 0.0005).abs() < 1e-5);
    Ok(())
}

// ── Baseline and residual ────────────────────────────────────────────────

#[test]
fn baseline_signal_is_mean_of_b0() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, [0.0007; 6], 300.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    assert!((dti.baseline_signal() - 300.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn residual_is_small_for_noiseless_data() -> Result<(), DtiError> {
    let scheme = scheme(40);
    let signals = dti_signal(&scheme, [0.0010, 0.0005, 0.0005, 0.0, 0.0, 0.0], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    assert!(dti.residual_norm() < 1e-8);
    Ok(())
}

// ── Error cases ──────────────────────────────────────────────────────────

#[test]
fn signal_length_mismatch_errors() {
    let scheme = scheme(30);
    let err = estimate_dti(&scheme, &[1.0; 5], DtiConfig::default()).unwrap_err();
    assert!(matches!(err, DtiError::SignalLengthMismatch { .. }));
}

#[test]
fn non_finite_signal_errors() {
    let scheme = scheme(30);
    let mut signals = vec![1.0; scheme.len()];
    signals[3] = f64::NAN;
    let err = estimate_dti(&scheme, &signals, DtiConfig::default()).unwrap_err();
    assert!(matches!(err, DtiError::NonFiniteSignal { index: 3, .. }));
}

#[test]
fn fewer_than_six_dwi_errors() {
    // 5 DWI volumes + 1 b0 = 6 total; only 5 DWI < 6 needed.
    let mut entries =
        vec![GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("b0")];
    for i in 0..5 {
        entries.push(
            GradientDirection::new(
                weighting(1_000.0),
                Vector::new([(i as f64 * 0.5).cos(), (i as f64 * 0.5).sin(), 0.0]),
            )
            .expect("DWI"),
        );
    }
    let scheme = GradientScheme::new(entries, GradientFrame::Lps).expect("5-dir scheme");
    let signals = vec![1.0; scheme.len()];
    let err = estimate_dti(&scheme, &signals, DtiConfig::default()).unwrap_err();
    assert!(matches!(err, DtiError::Underdetermined { .. }));
}

#[test]
fn zero_log_signal_errors() {
    // Signal = 0 produces -inf in log domain.
    let scheme = scheme(10);
    let mut signals = vec![1.0; scheme.len()];
    signals[1] = 0.0; // first DWI
    let err = estimate_dti(&scheme, &signals, DtiConfig::default()).unwrap_err();
    assert!(matches!(
        err,
        DtiError::InvalidNormalisedSignal { index: 1, .. }
    ));
}

#[test]
fn frame_is_preserved() -> Result<(), DtiError> {
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, [0.0007; 6], 1.0);
    let dti = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    assert_eq!(dti.frame(), GradientFrame::Lps);
    Ok(())
}

// ── ADR 0036 verification condition 7: gradient reorientation ────────────

/// Build a proper rotation matrix about the y-axis by `angle` radians.
fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// A rigid correction applied to a diffusion series must reorient the
/// gradient scheme with it.  Skipping the reorientation produces a
/// silently wrong tensor — the single most common defect class in this
/// domain (ADR 0036, decision 7).
///
/// The oracle: signals are generated from the reoriented scheme so they
/// represent data after a rigid transform.  Fitting with the reoriented
/// scheme recovers the original PEV; fitting with the original scheme
/// (i.e., skipping reorientation) recovers a different, wrong PEV.
#[test]
fn reorient_gradients_recover_original_pev_skip_reorientation_gives_wrong_pev()
-> Result<(), DtiError> {
    // ── Setup ────────────────────────────────────────────────────────────
    let scheme = scheme(60);
    // Tensor with PEV along +z: D = diag(0.0003, 0.0003, 0.0017).
    let tensor: [f64; 6] = [0.0003, 0.0003, 0.0017, 0.0, 0.0, 0.0];
    let original_pev = [0.0, 0.0, 1.0];

    // Rotate 45° about y.  A proper rotation (det = +1, orthonormal).
    let angle = std::f64::consts::FRAC_PI_4;
    let rotation = rotation_y(angle);

    // Reoriented scheme: gradients are rotated with the data.
    let scheme_reoriented = scheme.reorient(rotation).expect("valid proper rotation");

    // Generate signals from the ORIGINAL tensor using the REORIENTED
    // scheme.  These signals represent data acquired after a rigid
    // correction has been applied to the image.
    let signals = dti_signal(&scheme_reoriented, tensor, 1000.0);

    // ── With reorientation: correct path ─────────────────────────────────
    let dti_correct = estimate_dti(&scheme_reoriented, &signals, DtiConfig::default())?;
    let pev_correct = dti_correct.principal_eigenvector();

    // The fitted PEV must recover the original direction — the rotation
    // was undone by the consistent reorientation.
    let dot_original = (pev_correct[0] * original_pev[0]
        + pev_correct[1] * original_pev[1]
        + pev_correct[2] * original_pev[2])
        .abs();
    assert!(
        dot_original > 0.99,
        "ADR 0036 vc7: after reorientation PEV must recover original; \
         PEV = [{:.4}, {:.4}, {:.4}], dot with +z = {dot_original:.4}",
        pev_correct[0],
        pev_correct[1],
        pev_correct[2]
    );

    // ── Without reorientation: error path ────────────────────────────────
    // Fit the same signals with the ORIGINAL (unrotated) scheme.  This
    // is the defect: the gradients were not rotated to match the data.
    let dti_wrong = estimate_dti(&scheme, &signals, DtiConfig::default())?;
    let pev_wrong = dti_wrong.principal_eigenvector();

    // The fitted PEV must NOT align with the original direction.  Without
    // reorientation, the PEV is rotated away from +z — it should be near
    // the rotation of +z, i.e., [sin(45°), 0, cos(45°)] = [r2, 0, r2].
    let dot_original_wrong = (pev_wrong[0] * original_pev[0]
        + pev_wrong[1] * original_pev[1]
        + pev_wrong[2] * original_pev[2])
        .abs();
    let r2 = std::f64::consts::SQRT_2 / 2.0;
    // D̂ = Rᵀ·D·R ⇒ PEV = Rᵀ·[0,0,1] = [-r2, 0, r2]
    let rotated_pev = [-r2, 0.0, r2];
    let dot_rotated = (pev_wrong[0] * rotated_pev[0]
        + pev_wrong[1] * rotated_pev[1]
        + pev_wrong[2] * rotated_pev[2])
        .abs();

    assert!(
        dot_original_wrong < 0.9,
        "ADR 0036 vc7: without reorientation PEV must NOT align with \
         original +z; got dot = {dot_original_wrong:.4}"
    );
    assert!(
        dot_rotated > 0.99,
        "ADR 0036 vc7: without reorientation PEV must align with rotated \
         +z = [{r2:.4}, 0, {r2:.4}]; PEV = [{:.4}, {:.4}, {:.4}], \
         dot = {dot_rotated:.4}",
        pev_wrong[0],
        pev_wrong[1],
        pev_wrong[2]
    );

    Ok(())
}

#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::super::entropy::marginal_entropy;
use super::super::mutual_information::{
    conditional_mutual_information, interaction_information, mutual_information,
    mutual_information_mattes, normalized_mutual_information, symmetric_uncertainty,
};

// ── mutual_information tests ──────────────────────────────────────────────────

#[test]
fn mi_identical_equals_marginal_entropy() {
    // I(X;X) = H(X) + H(X) - H(X,X) = 2H(X) - H(X) = H(X).
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let h_a = marginal_entropy(&a, 8).unwrap();
    let mi = mutual_information(&a, &a, 8).unwrap();
    assert!(
        (mi - h_a).abs() < 1e-9,
        "I(X;X)={mi:.6} must equal H(X)={h_a:.6}"
    );
}

/// `MI(X, X) = H(X)`: a signal shares all of its information with itself.
///
/// Replaces a `mi >= 0.0` assertion that `mutual_information` guaranteed by
/// construction and could therefore never fail. This pins the value instead:
/// eight bins populated uniformly give `H = ln 8`, so the estimator has to
/// build a real histogram to satisfy it.
///
/// Tolerance: 64 joint-histogram terms summed in f64, each bounded by
/// `ln 8 = 2.08`, so worst-case accumulation is `64 * 2.2e-16 * 2.08 = 2.9e-14`.
#[test]
fn mi_of_a_signal_with_itself_equals_its_entropy() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let mi = mutual_information(&a, &a, 8).unwrap();
    let entropy = 8.0_f64.ln();
    assert!(
        (mi - entropy).abs() < 1e-12,
        "MI(X, X) must equal H(X) = {entropy}, got {mi}"
    );
}

/// `MI = 0` exactly when the joint distribution factorises.
///
/// `a` cycles every 8 samples and `b` every 64, so across 256 samples each of
/// the 64 (a, b) pairs occurs exactly 4 times: `p(a, b) = p(a) * p(b)` by
/// construction. Independence is then a property the estimator must reproduce,
/// not a bound it is clamped into.
#[test]
fn mi_of_exactly_independent_channels_is_zero() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let mi = mutual_information(&a, &b, 8).unwrap();
    assert!(
        mi.abs() < 1e-12,
        "independent channels must share no information, got {mi}"
    );
}

#[test]
fn mi_is_symmetric() {
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let mi_ab = mutual_information(&a, &b, 8).unwrap();
    let mi_ba = mutual_information(&b, &a, 8).unwrap();
    assert!(
        (mi_ab - mi_ba).abs() < 1e-10,
        "I(X;Y)={mi_ab:.10} must equal I(Y;X)={mi_ba:.10}"
    );
}

#[test]
fn mi_bounded_above_by_min_marginals() {
    // I(X;Y) ≤ min(H(X), H(Y)) (data processing inequality consequence).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let h_a = marginal_entropy(&a, 8).unwrap();
    let h_b = marginal_entropy(&b, 8).unwrap();
    let mi = mutual_information(&a, &b, 8).unwrap();
    let upper = h_a.min(h_b);
    assert!(
        mi <= upper + 1e-9,
        "I(X;Y)={mi:.6} must be ≤ min(H(X),H(Y))={upper:.6}"
    );
}

#[test]
fn mi_rejects_length_mismatch() {
    let a = vec![1.0_f32; 10];
    let b = vec![1.0_f32; 8];
    assert!(mutual_information(&a, &b, 4).is_err());
}

#[test]
fn mi_rejects_empty() {
    assert!(mutual_information(&[], &[], 4).is_err());
}

// ── normalized_mutual_information tests ──────────────────────────────────────

#[test]
fn nmi_identical_non_constant_is_two() {
    // NMI(X,X) = (H(X)+H(X)) / H(X,X) = 2H(X)/H(X) = 2.0.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let nmi = normalized_mutual_information(&a, &a, 8).unwrap();
    assert!(
        (nmi - 2.0).abs() < 1e-9,
        "NMI(X,X)={nmi:.6} must equal 2.0 for non-constant X"
    );
}

#[test]
fn nmi_constant_channel_returns_one() {
    // Both channels constant → H(X,Y) < ε → return 1.0.
    let a = vec![5.0_f32; 100];
    let b = vec![3.0_f32; 100];
    let nmi = normalized_mutual_information(&a, &b, 8).unwrap();
    assert!(
        (nmi - 1.0).abs() < 1e-9,
        "NMI(const,const)={nmi:.6} must be 1.0"
    );
}

#[test]
fn nmi_at_least_one() {
    // NMI ≥ 1.0 always (Studholme bound).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let nmi = normalized_mutual_information(&a, &b, 8).unwrap();
    assert!(nmi >= 1.0 - 1e-9, "NMI={nmi:.6} must be ≥ 1.0");
}

#[test]
fn nmi_at_most_two() {
    // NMI ≤ 2.0 (achieved only for identical non-constant channels).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let nmi = normalized_mutual_information(&a, &b, 8).unwrap();
    assert!(nmi <= 2.0 + 1e-9, "NMI={nmi:.6} must be ≤ 2.0");
}

// ── mutual_information_mattes tests ──────────────────────────────────────────

#[test]
fn mi_mattes_identical_positive() {
    // I_mattes(X;X) > 0 for non-constant X.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let mi = mutual_information_mattes(&a, &a, 8).unwrap();
    assert!(mi > 0.0, "Mattes MI(X,X) must be positive, got {mi}");
}

#[test]
fn mi_mattes_constant_channel_is_zero() {
    // I(X, const) = 0 since H(const) = 0.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let b_const = vec![3.0_f32; 64];
    let mi = mutual_information_mattes(&a, &b_const, 8).unwrap();
    assert!(mi.abs() < 1e-9, "Mattes MI(X, const) must be 0, got {mi}");
}

/// Soft binning must agree with hard binning where the two cannot differ.
///
/// Replaces a `mi >= 0.0` assertion the implementation guaranteed. Mattes
/// spreads each sample bilinearly over neighbouring cells, which changes the
/// estimate in general — but a signal against itself puts every sample on the
/// diagonal, where both estimators see the same distribution. Agreement there
/// constrains the soft-assignment weights: a normalisation error in the
/// bilinear split breaks it while leaving the value non-negative.
///
/// Tolerance: as `mi_of_a_signal_with_itself_equals_its_entropy`, widened one
/// decade for the four-way weight split per sample.
#[test]
fn mi_mattes_agrees_with_hard_binning_on_the_diagonal() {
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let soft = mutual_information_mattes(&a, &a, 8).unwrap();
    let hard = mutual_information(&a, &a, 8).unwrap();
    assert!(
        (soft - hard).abs() < 1e-11,
        "on the diagonal soft binning must match hard binning: {soft} vs {hard}"
    );
}

#[test]
fn mi_mattes_rejects_length_mismatch() {
    let a = vec![1.0_f32; 10];
    let b = vec![1.0_f32; 8];
    assert!(mutual_information_mattes(&a, &b, 4).is_err());
}

// ── symmetric_uncertainty tests ───────────────────────────────────────────────

#[test]
fn su_identical_non_constant_is_one() {
    // SU(X,X) = 2·H(X)/(H(X)+H(X)) = 1.0.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let su = symmetric_uncertainty(&a, &a, 8).unwrap();
    assert!(
        (su - 1.0).abs() < 1e-9,
        "SU(X,X)={su:.6} must equal 1.0 for non-constant X"
    );
}

/// Symmetric uncertainty reaches its endpoints, rather than merely lying
/// between them.
///
/// Replaces two assertions that `symmetric_uncertainty` guaranteed by clamping
/// to `[0, 1]`. Both extremes are constructed exactly here: a signal against
/// itself is total dependence, the independent pair is none. An estimator
/// returning a constant mid-range value passed the old bounds and fails these.
#[test]
fn su_reaches_one_for_identity_and_zero_for_independence() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();

    let identical = symmetric_uncertainty(&a, &a, 8).unwrap();
    assert!(
        (identical - 1.0).abs() < 1e-12,
        "SU of a signal with itself must be 1, got {identical}"
    );

    let independent = symmetric_uncertainty(&a, &b, 8).unwrap();
    assert!(
        independent.abs() < 1e-12,
        "SU of independent channels must be 0, got {independent}"
    );
}

#[test]
fn su_constant_channels_returns_zero() {
    let a = vec![5.0_f32; 100];
    let b = vec![3.0_f32; 100];
    let su = symmetric_uncertainty(&a, &b, 8).unwrap();
    assert!(su.abs() < 1e-9, "SU(const,const)={su:.6} must be 0.0");
}

// ── conditional_mutual_information tests ─────────────────────────────────────

/// Conditioning on a copy of X removes everything X and Y share.
///
/// Replaces a `cmi >= 0.0` assertion the implementation guaranteed. With Z set
/// to X itself, `I(X;Y|Z) = 0` exactly: whatever Y knows about X, Z already
/// told us. The four-entropy expression has to produce that value — a sign
/// error or a mismatched joint histogram gives a non-zero result while staying
/// non-negative.
#[test]
fn cmi_vanishes_when_conditioned_on_one_of_its_arguments() {
    let x: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let cmi = conditional_mutual_information(&x, &y, &x, 8).unwrap();
    assert!(
        cmi.abs() < 1e-12,
        "conditioning on X must leave X and Y sharing nothing, got {cmi}"
    );
}

#[test]
fn cmi_constant_z_equals_mi() {
    // I(X;Y|const) = I(X;Y): when Z=const, H(Z)=0, H(X,Z)=H(X), H(Y,Z)=H(Y), H(X,Y,Z)=H(X,Y).
    let x: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..128).map(|i| ((i / 8) % 8) as f32).collect();
    let z_const = vec![3.0_f32; 128];
    let cmi = conditional_mutual_information(&x, &y, &z_const, 8).unwrap();
    let mi = mutual_information(&x, &y, 8).unwrap();
    assert!(
        (cmi - mi).abs() < 1e-9,
        "CMI(X,Y|const)={cmi:.9} must equal MI(X,Y)={mi:.9}"
    );
}

#[test]
fn cmi_knowing_z_equal_to_y_is_zero() {
    // I(X;Y|Y) = H(X,Y) + H(Y,Y) − H(X,Y,Y) − H(Y) = H(X,Y) + H(Y) − H(X,Y) − H(Y) = 0.
    let x: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..128).map(|i| ((i / 8) % 8) as f32).collect();
    let cmi = conditional_mutual_information(&x, &y, &y, 8).unwrap();
    assert!(cmi.abs() < 1e-9, "CMI(X;Y|Y)={cmi:.10} must be 0");
}

#[test]
fn cmi_rejects_length_mismatch() {
    let x = vec![1.0_f32; 10];
    let y = vec![1.0_f32; 10];
    let z = vec![1.0_f32; 8];
    assert!(conditional_mutual_information(&x, &y, &z, 4).is_err());
}

#[test]
fn cmi_rejects_empty() {
    assert!(conditional_mutual_information(&[], &[], &[], 4).is_err());
}

// ── interaction_information tests ─────────────────────────────────────────────

#[test]
fn ii_constant_z_gives_zero() {
    // II(X;Y;const) = I(X;Y) − I(X;Y|const) = I(X;Y) − I(X;Y) = 0.
    let x: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..128).map(|i| ((i / 8) % 8) as f32).collect();
    let z_const = vec![3.0_f32; 128];
    let ii = interaction_information(&x, &y, &z_const, 8).unwrap();
    assert!(ii.abs() < 1e-9, "II(X;Y;const)={ii:.10} must be 0");
}

#[test]
fn ii_identical_triple_is_positive() {
    // II(X;X;X) = I(X;X) − I(X;X|X) = H(X) − 0 = H(X) > 0.
    let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let ii = interaction_information(&x, &x, &x, 8).unwrap();
    assert!(ii > 0.0, "II(X;X;X)={ii:.6} must be positive (= H(X))");
}

#[test]
fn ii_matches_formula_mi_minus_cmi() {
    // II = I(X;Y) − I(X;Y|Z); verify internal consistency.
    let x: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let z: Vec<f32> = (0..128).map(|i| ((i / 16) % 8) as f32).collect();
    let ii = interaction_information(&x, &y, &z, 8).unwrap();
    let mi_xy = mutual_information(&x, &y, 8).unwrap();
    let cmi = conditional_mutual_information(&x, &y, &z, 8).unwrap();
    let manual = mi_xy - cmi;
    assert!(
        (ii - manual).abs() < 1e-12,
        "II={ii:.12} must equal I(X;Y)−CMI(X;Y|Z)={manual:.12}"
    );
}

#[test]
fn ii_rejects_length_mismatch() {
    let x = vec![1.0_f32; 10];
    let y = vec![1.0_f32; 10];
    let z = vec![1.0_f32; 8];
    assert!(interaction_information(&x, &y, &z, 4).is_err());
}

#[test]
fn ii_rejects_empty() {
    assert!(interaction_information(&[], &[], &[], 4).is_err());
}

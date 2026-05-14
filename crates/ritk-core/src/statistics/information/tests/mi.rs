use super::super::mutual_information::{
    conditional_mutual_information, interaction_information, mutual_information,
    mutual_information_mattes, normalized_mutual_information, symmetric_uncertainty,
};
use super::super::entropy::marginal_entropy;

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

#[test]
fn mi_is_non_negative() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let mi = mutual_information(&a, &b, 8).unwrap();
    assert!(mi >= 0.0, "MI must be non-negative, got {mi}");
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

#[test]
fn mi_mattes_non_negative() {
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let mi = mutual_information_mattes(&a, &b, 8).unwrap();
    assert!(mi >= 0.0, "Mattes MI must be non-negative, got {mi}");
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

#[test]
fn su_in_zero_one() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let su = symmetric_uncertainty(&a, &b, 8).unwrap();
    assert!(su >= 0.0 - 1e-9, "SU={su:.6} must be ≥ 0");
    assert!(su <= 1.0 + 1e-9, "SU={su:.6} must be ≤ 1");
}

#[test]
fn su_constant_channels_returns_zero() {
    let a = vec![5.0_f32; 100];
    let b = vec![3.0_f32; 100];
    let su = symmetric_uncertainty(&a, &b, 8).unwrap();
    assert!(
        su.abs() < 1e-9,
        "SU(const,const)={su:.6} must be 0.0"
    );
}

// ── conditional_mutual_information tests ─────────────────────────────────────

#[test]
fn cmi_is_non_negative() {
    // I(X;Y|Z) ≥ 0 by the data processing inequality.
    let x: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let y: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let z: Vec<f32> = (0..256).map(|i| ((i / 16) % 8) as f32).collect();
    let cmi = conditional_mutual_information(&x, &y, &z, 8).unwrap();
    assert!(cmi >= 0.0, "CMI must be ≥ 0, got {cmi}");
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

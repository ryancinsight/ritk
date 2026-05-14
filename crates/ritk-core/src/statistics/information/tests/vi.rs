use super::super::variation_of_information::{
    multivariate_variation_of_information, variation_of_information,
};
use super::super::entropy::marginal_entropy;

// ── variation_of_information tests ───────────────────────────────────────────

#[test]
fn vi_identical_is_zero() {
    // VI(X,X) = H(X) + H(X) - 2·I(X;X) = 2H(X) - 2H(X) = 0.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let vi = variation_of_information(&a, &a, 8).unwrap();
    assert!(vi.abs() < 1e-9, "VI(X,X)={vi:.10} must be 0");
}

#[test]
fn vi_is_non_negative() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let vi = variation_of_information(&a, &b, 8).unwrap();
    assert!(vi >= 0.0, "VI must be non-negative, got {vi}");
}

#[test]
fn vi_is_symmetric() {
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let vi_ab = variation_of_information(&a, &b, 8).unwrap();
    let vi_ba = variation_of_information(&b, &a, 8).unwrap();
    assert!(
        (vi_ab - vi_ba).abs() < 1e-10,
        "VI(X,Y)={vi_ab:.10} must equal VI(Y,X)={vi_ba:.10}"
    );
}

#[test]
fn vi_against_constant_equals_marginal_entropy() {
    // VI(X, const) = H(X) + H(const) - 2·I(X;const) = H(X) + 0 - 0 = H(X).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b = vec![0.0_f32; 256];
    let h_a = marginal_entropy(&a, 8).unwrap();
    let vi = variation_of_information(&a, &b, 8).unwrap();
    assert!(
        (vi - h_a).abs() < 0.01,
        "VI(X,const)={vi:.6} must ≈ H(X)={h_a:.6}"
    );
}

#[test]
fn vi_bounded_above_by_sum_of_marginals() {
    // VI(X,Y) ≤ H(X) + H(Y) (triangle inequality consequence).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let h_a = marginal_entropy(&a, 8).unwrap();
    let h_b = marginal_entropy(&b, 8).unwrap();
    let vi = variation_of_information(&a, &b, 8).unwrap();
    assert!(
        vi <= h_a + h_b + 1e-9,
        "VI={vi:.6} must be ≤ H(X)+H(Y)={:.6}", h_a + h_b
    );
}

#[test]
fn vi_rejects_length_mismatch() {
    let a = vec![1.0_f32; 10];
    let b = vec![1.0_f32; 8];
    assert!(variation_of_information(&a, &b, 4).is_err());
}

#[test]
fn vi_rejects_empty() {
    assert!(variation_of_information(&[], &[], 4).is_err());
}

// ── multivariate_variation_of_information tests ───────────────────────────────

#[test]
fn mvi_identical_channels_is_zero() {
    // VI_n(X,X,X) = avg of VI(X,X) pairs = 0.
    let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let mvi = multivariate_variation_of_information(
        &[x.as_slice(), x.as_slice(), x.as_slice()],
        8,
    )
    .unwrap();
    assert!(mvi.abs() < 1e-9, "MVI(X,X,X)={mvi:.10} must be 0");
}

#[test]
fn mvi_is_non_negative() {
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..128).map(|i| ((i / 16) % 8) as f32).collect();
    let mvi = multivariate_variation_of_information(&[&a, &b, &c], 8).unwrap();
    assert!(mvi >= 0.0, "MVI must be ≥ 0, got {mvi}");
}

#[test]
fn mvi_two_channels_equals_bivariate_vi() {
    // VI_n with n=2 has exactly one pair: avg = VI(X,Y).
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let mvi = multivariate_variation_of_information(&[&a, &b], 8).unwrap();
    let vi = variation_of_information(&a, &b, 8).unwrap();
    assert!(
        (mvi - vi).abs() < 1e-12,
        "MVI([X,Y])={mvi:.12} must equal VI(X,Y)={vi:.12}"
    );
}

#[test]
fn mvi_rejects_single_channel() {
    let x: Vec<f32> = vec![1.0; 8];
    assert!(multivariate_variation_of_information(&[x.as_slice()], 4).is_err());
}

#[test]
fn mvi_rejects_empty_channels() {
    assert!(multivariate_variation_of_information(&[], 4).is_err());
}

#[test]
fn mvi_rejects_length_mismatch() {
    let a = vec![1.0_f32; 10];
    let b = vec![1.0_f32; 8];
    assert!(multivariate_variation_of_information(&[&a, &b], 4).is_err());
}

use super::super::entropy::{joint_entropy, joint_entropy_n, marginal_entropy};

// ── marginal_entropy tests ────────────────────────────────────────────────────

#[test]
fn marginal_entropy_uniform_eight_values_equals_ln8() {
    // H(X) = ln(8) when X takes 8 equally probable values.
    let data: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let h = marginal_entropy(&data, 8).unwrap();
    let expected = (8.0_f64).ln();
    assert!(
        (h - expected).abs() < 0.01,
        "H(uniform-8) = {h:.6}, expected ln(8) = {expected:.6}"
    );
}

#[test]
fn marginal_entropy_constant_is_zero() {
    // H(constant) = 0.
    let data = vec![3.0_f32; 100];
    let h = marginal_entropy(&data, 8).unwrap();
    assert!(h.abs() < 1e-10, "H(constant) must be 0, got {h}");
}

#[test]
fn marginal_entropy_rejects_empty() {
    assert!(marginal_entropy(&[], 8).is_err());
}

#[test]
fn marginal_entropy_rejects_one_bin() {
    let data = vec![1.0_f32; 10];
    assert!(marginal_entropy(&data, 1).is_err());
}

// ── joint_entropy tests ───────────────────────────────────────────────────────

#[test]
fn joint_entropy_identical_channels_equals_marginal() {
    // H(X,X) = H(X) because knowing X determines X.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let h_a = marginal_entropy(&a, 8).unwrap();
    let h_aa = joint_entropy(&a, &a, 8).unwrap();
    assert!(
        (h_aa - h_a).abs() < 1e-9,
        "H(X,X)={h_aa:.6} must equal H(X)={h_a:.6}"
    );
}

#[test]
fn joint_entropy_independent_channels_near_sum_of_marginals() {
    // H(X,Y) ≈ H(X)+H(Y) for independent X,Y (upper bound tightens with more bins).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let h_a = marginal_entropy(&a, 8).unwrap();
    let h_b = marginal_entropy(&b, 8).unwrap();
    let h_ab = joint_entropy(&a, &b, 8).unwrap();
    assert!(h_ab <= h_a + h_b + 1e-9, "H(X,Y) must be ≤ H(X)+H(Y)");
}

#[test]
fn joint_entropy_rejects_length_mismatch() {
    let a = vec![1.0_f32; 10];
    let b = vec![1.0_f32; 8];
    assert!(joint_entropy(&a, &b, 4).is_err());
}

// ── joint_entropy_n tests ─────────────────────────────────────────────────────

#[test]
fn joint_entropy_n_single_channel_equals_marginal() {
    let a: Vec<f32> = (0..32).map(|i| (i % 8) as f32).collect();
    let h1 = joint_entropy_n(&[a.as_slice()], 8).unwrap();
    let h_a = marginal_entropy(&a, 8).unwrap();
    assert!((h1 - h_a).abs() < 1e-9, "H(X) via joint_n={h1:.6} vs marginal={h_a:.6}");
}

#[test]
fn joint_entropy_n_rejects_empty_channels() {
    assert!(joint_entropy_n(&[], 8).is_err());
}

#[test]
fn joint_entropy_n_rejects_excessive_size() {
    // 65^3 = 274625 which is fine, but 65^5 > 4M
    let a = vec![1.0_f32; 10];
    // num_bins=65 > 64 so joint_entropy_n should reject due to size if n is large enough
    // 65^4 = 17,850,625 > 4,194,304
    let refs: Vec<&[f32]> = (0..4).map(|_| a.as_slice()).collect();
    assert!(joint_entropy_n(&refs, 65).is_err());
}

use super::super::total_correlation::total_correlation;
use super::super::mutual_information::mutual_information;

// ── total_correlation tests ───────────────────────────────────────────────────

#[test]
fn tc_single_channel_is_zero() {
    // TC(X) = H(X) - H(X) = 0 by definition.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let tc = total_correlation(&[a.as_slice()], 8).unwrap();
    assert!(tc.abs() < 1e-9, "TC(X)={tc:.10} must be 0");
}

#[test]
fn tc_two_identical_channels_equals_mutual_information() {
    // TC(X,X) = I(X;X) = H(X).
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let tc = total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    let mi = mutual_information(&a, &a, 8).unwrap();
    assert!(
        (tc - mi).abs() < 1e-9,
        "TC(X,X)={tc:.6} must equal MI(X,X)={mi:.6}"
    );
}

#[test]
fn tc_is_non_negative() {
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let tc = total_correlation(&[a.as_slice(), b.as_slice()], 8).unwrap();
    assert!(tc >= 0.0, "TC must be non-negative, got {tc}");
}

#[test]
fn tc_three_channels_at_least_two_channel_tc() {
    // TC(X,X,X) ≥ TC(X,X): more channels cannot reduce total correlation for identical inputs.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let tc2 = total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    let tc3 = total_correlation(&[a.as_slice(), a.as_slice(), a.as_slice()], 8).unwrap();
    assert!(
        tc3 >= tc2 - 1e-9,
        "TC(X,X,X)={tc3:.6} must be ≥ TC(X,X)={tc2:.6}"
    );
}

#[test]
fn tc_independent_channels_near_zero() {
    // Channels cycling through disjoint patterns have lower TC than identical channels.
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let tc_indep = total_correlation(&[a.as_slice(), b.as_slice()], 8).unwrap();
    let tc_ident = total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    assert!(
        tc_ident > tc_indep - 1e-9,
        "TC(X,X)={tc_ident:.6} must be ≥ TC(X,Y)={tc_indep:.6}"
    );
}

#[test]
fn tc_rejects_empty_channels() {
    assert!(total_correlation(&[], 8).is_err());
}

#[test]
fn tc_rejects_excessive_histogram_size() {
    // num_bins=65, n=4 → 65^4 = 17,850,625 > 4,194,304.
    let a = vec![1.0_f32; 10];
    let refs: Vec<&[f32]> = (0..4).map(|_| a.as_slice()).collect();
    assert!(total_correlation(&refs, 65).is_err());
}

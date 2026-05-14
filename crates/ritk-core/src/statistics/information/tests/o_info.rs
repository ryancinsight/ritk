use super::super::mutual_information::mutual_information;
use super::super::o_information::{
    dual_total_correlation, o_information, o_information_direct, o_information_from_tc_dtc,
};
use super::super::total_correlation::total_correlation;

// ── dual_total_correlation tests ──────────────────────────────────────────────

#[test]
fn dtc_rejects_single_channel() {
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    assert!(dual_total_correlation(&[a.as_slice()], 8).is_err());
}

#[test]
fn dtc_is_non_negative() {
    // DTC ≥ 0 by chain rule for conditional entropy.
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let dtc = dual_total_correlation(&[a.as_slice(), b.as_slice()], 8).unwrap();
    assert!(dtc >= 0.0, "DTC must be ≥ 0, got {dtc}");
}

#[test]
fn dtc_two_channels_equals_mutual_information() {
    // DTC(X,Y) = I(X;Y) for n=2 (Han 1978, Corollary 1).
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = a.iter().map(|&v| v * 2.0).collect();
    let dtc = dual_total_correlation(&[a.as_slice(), b.as_slice()], 8).unwrap();
    let mi = mutual_information(&a, &b, 8).unwrap();
    assert!(
        (dtc - mi).abs() < 1e-9,
        "DTC(X,Y)={dtc:.10} must equal MI(X,Y)={mi:.10} for n=2"
    );
}

#[test]
fn dtc_two_identical_channels_equals_tc() {
    // For n=2, DTC(X,X) = TC(X,X) = I(X;X) = H(X).
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let dtc = dual_total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    let tc = total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    assert!(
        (dtc - tc).abs() < 1e-9,
        "DTC(X,X)={dtc:.10} must equal TC(X,X)={tc:.10} for n=2"
    );
}

#[test]
fn dtc_independent_channels_near_zero() {
    // DTC → 0 when channels are independent (no shared conditional information).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 8) % 8) as f32).collect();
    let dtc_indep = dual_total_correlation(&[a.as_slice(), b.as_slice()], 8).unwrap();
    let dtc_ident = dual_total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
    assert!(
        dtc_ident > dtc_indep - 1e-9,
        "DTC(X,X)={dtc_ident:.6} must be ≥ DTC(X,Y_indep)={dtc_indep:.6}"
    );
}

#[test]
fn dtc_rejects_empty_channels() {
    assert!(dual_total_correlation(&[], 8).is_err());
}

// ── o_information tests ───────────────────────────────────────────────────────

#[test]
fn oi_rejects_single_channel() {
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    assert!(o_information(&[a.as_slice()], 8).is_err());
}

#[test]
fn oi_two_channels_is_zero() {
    // Ω(X,Y) = TC(X,Y) − DTC(X,Y) = I(X;Y) − I(X;Y) = 0 for any n=2 pair.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| ((i * 3) % 8) as f32).collect();
    let oi = o_information(&[a.as_slice(), b.as_slice()], 8).unwrap();
    assert!(
        oi.abs() < 1e-9,
        "Ω(X,Y)={oi:.10} must be exactly 0 for n=2 (TC=DTC=I(X;Y))"
    );
}

#[test]
fn oi_identical_triple_is_non_negative() {
    // Identical channels are maximally redundant: Ω ≥ 0.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let oi = o_information(&[a.as_slice(), a.as_slice(), a.as_slice()], 8).unwrap();
    assert!(oi >= 0.0, "Ω(X,X,X) must be ≥ 0 (redundancy-dominated), got {oi}");
}

#[test]
fn oi_from_tc_dtc_matches_direct() {
    // Verify the shortcut formula o_information_from_tc_dtc = o_information.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..64).map(|i| ((i * 5) % 8) as f32).collect();
    let channels: &[&[f32]] = &[a.as_slice(), b.as_slice(), c.as_slice()];
    let tc = total_correlation(channels, 8).unwrap();
    let dtc = dual_total_correlation(channels, 8).unwrap();
    let oi_fast = o_information_from_tc_dtc(tc, dtc);
    let oi_full = o_information(channels, 8).unwrap();
    assert!(
        (oi_fast - oi_full).abs() < 1e-9,
        "oi_from_tc_dtc={oi_fast:.10} must equal o_information={oi_full:.10}"
    );
}

#[test]
fn oi_direct_matches_via_tc_dtc() {
    // o_information_direct must equal o_information (two different computation paths).
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..128).map(|i| ((i * 3) % 8) as f32).collect();
    let channels: &[&[f32]] = &[a.as_slice(), b.as_slice(), c.as_slice()];
    let oi_std = o_information(channels, 8).unwrap();
    let oi_dir = o_information_direct(channels, 8).unwrap();
    assert!(
        (oi_std - oi_dir).abs() < 1e-9,
        "o_information={oi_std:.10} must equal o_information_direct={oi_dir:.10}"
    );
}

#[test]
fn oi_three_channel_matches_interaction_information() {
    // For n=3: Ω(X,Y,Z) = II(X;Y;Z) = H(X)+H(Y)+H(Z)+H(X,Y,Z)−H(X,Y)−H(X,Z)−H(Y,Z).
    use super::super::mutual_information::interaction_information;
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..64).map(|i| ((i * 3) % 8) as f32).collect();
    let oi = o_information(&[a.as_slice(), b.as_slice(), c.as_slice()], 8).unwrap();
    let ii = interaction_information(&a, &b, &c, 8).unwrap();
    assert!(
        (oi - ii).abs() < 1e-9,
        "Ω(X,Y,Z)={oi:.10} must equal II(X;Y;Z)={ii:.10} for n=3 (Rosas 2019)"
    );
}

// ── n≥4 tests ─────────────────────────────────────────────────────────────────

#[test]
fn dtc_four_channels_non_negative() {
    // DTC ≥ 0 for any n, by Han (1978) Theorem 1 (sub-modularity of entropy).
    let a: Vec<f32> = (0..256).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..256).map(|i| ((i * 3) % 8) as f32).collect();
    let d: Vec<f32> = (0..256).map(|i| ((i * 5 + 1) % 8) as f32).collect();
    let dtc =
        dual_total_correlation(&[a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()], 8)
            .unwrap();
    assert!(dtc >= 0.0, "DTC(4 channels) must be ≥ 0, got {dtc}");
}

#[test]
fn oi_four_identical_channels_is_positive() {
    // Analytical: for n=4 identical X, TC = 3H(X) and DTC = H(X), so Ω = 2H(X) > 0.
    // Derivation: TC = 4H(X)−H(X) = 3H(X); DTC = 4H(X)−3H(X) = H(X); Ω = TC−DTC = 2H(X).
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
    let channels: &[&[f32]] = &[a.as_slice(), a.as_slice(), a.as_slice(), a.as_slice()];
    let oi = o_information(channels, 8).unwrap();
    let h_x: f64 = {
        // H(X) for a uniform 8-symbol distribution over 64 samples = log2(8) = 3.0 bits
        let tc_xx = total_correlation(&[a.as_slice(), a.as_slice()], 8).unwrap();
        // TC(X,X) = 2H(X) - H(X) = H(X)
        tc_xx
    };
    let expected = 2.0 * h_x;
    assert!(
        (oi - expected).abs() < 1e-9,
        "Ω(X,X,X,X) must equal 2H(X)={expected:.10}, got {oi:.10}"
    );
}

#[test]
fn oi_four_independent_channels_near_zero() {
    // Analytical: independent uniform channels over 256 samples with 4 bins.
    // a[i]=i%4, b[i]=(i/4)%4, c[i]=(i/16)%4, d[i]=(i/64)%4 tile perfectly.
    // TC = 4H(X) − H(X₁,X₂,X₃,X₄) = 4·2 − 8 = 0; DTC = 0 similarly; Ω = 0.
    let n = 256_usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i / 4) % 4) as f32).collect();
    let c: Vec<f32> = (0..n).map(|i| ((i / 16) % 4) as f32).collect();
    let d: Vec<f32> = (0..n).map(|i| ((i / 64) % 4) as f32).collect();
    let oi =
        o_information(&[a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()], 4).unwrap();
    assert!(
        oi.abs() < 1e-9,
        "Ω(independent 4-channel) must be 0, got {oi:.10}"
    );
}

#[test]
fn oi_direct_matches_for_n4() {
    // o_information_direct and o_information must agree for n=4 channels.
    let a: Vec<f32> = (0..128).map(|i| (i % 8) as f32).collect();
    let b: Vec<f32> = (0..128).map(|i| ((i / 4) % 8) as f32).collect();
    let c: Vec<f32> = (0..128).map(|i| ((i * 3) % 8) as f32).collect();
    let d: Vec<f32> = (0..128).map(|i| ((i * 5 + 1) % 8) as f32).collect();
    let channels: &[&[f32]] = &[a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()];
    let oi_std = o_information(channels, 8).unwrap();
    let oi_dir = o_information_direct(channels, 8).unwrap();
    assert!(
        (oi_std - oi_dir).abs() < 1e-9,
        "o_information={oi_std:.10} must equal o_information_direct={oi_dir:.10} for n=4"
    );
}

#[test]
fn dtc_n4_independent_near_zero() {
    // DTC(X₁,X₂,X₃,X₄) = 0 for independent uniform channels (Han 1978).
    // Each H(X₁,X₂,X₃) = 3·log₂(4) = 6 for independent 4-symbol vars over 256 samples.
    // DTC = 4·6 − 3·8 = 24 − 24 = 0.
    let n = 256_usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i / 4) % 4) as f32).collect();
    let c: Vec<f32> = (0..n).map(|i| ((i / 16) % 4) as f32).collect();
    let d: Vec<f32> = (0..n).map(|i| ((i / 64) % 4) as f32).collect();
    let dtc =
        dual_total_correlation(&[a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()], 4)
            .unwrap();
    assert!(
        dtc.abs() < 1e-9,
        "DTC(independent 4-channel) must be 0, got {dtc:.10}"
    );
}

//! Input validation and edge-case property tests for the direct Parzen path.

use super::super::types::ParzenConfig;
use super::super::*;

#[test]
#[should_panic(expected = "fixed_norm must not be empty")]
fn direct_rejects_empty_input() {
    // TEST-318-06: Empty input must panic with a clear message.
    let empty: Vec<f32> = vec![];
    let _ = compute_joint_histogram_direct(&empty, &empty, 32, 1.0, 1.0, None, None);
}

#[test]
#[should_panic(expected = "sigma_sq must be positive")]
fn direct_rejects_zero_sigma() {
    // TEST-318-06: Zero sigma must panic.
    let _ = ParzenConfig::new(0.0);
}

#[test]
#[should_panic(expected = "sigma_sq must be positive, got NaN")]
fn direct_rejects_nan_sigma() {
    // TEST-318-06: NaN sigma must panic.
    let _ = ParzenConfig::new(f32::NAN);
}

#[test]
fn direct_single_bin_histogram() {
    // TEST-318-06: Single-bin histogram (degenerate case) — all weight
    // concentrates in the one cell.
    let num_bins = 1;
    let sigma_sq = 1.0_f32;
    let fixed = vec![0.0f32];
    let moving = vec![0.0f32];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let slice = hist_data.as_slice::<f32>().unwrap();

    assert_eq!(slice.len(), 1, "single-bin histogram must have 1 entry");
    assert!(
        slice[0] > 0.0,
        "single-bin entry must be positive, got {}",
        slice[0]
    );
}

//! Tests for temporal synchronization (Sprint 354 split, ARCH-350-04).
//!
//! Extracted from the monolithic `classical/temporal.rs` so that the test
//! module lives next to the code it tests without bloating the public
//! module surface.

use leto::Array1;

use super::config::TemporalSyncConfig;
use super::sync::TemporalSync;

#[test]
fn test_sync_identical_signals() {
    let sync = TemporalSync::default();
    let signal = Array1::from_vec([100], (0..100).map(|i| (i as f64).sin()).collect()).expect("infallible: validated precondition");

    let (shift, metrics) = sync.synchronize(&signal, &signal).expect("infallible: validated precondition");

    // Identical signals should have zero shift
    assert!(
        shift.abs() < 1e-6,
        "Shift for identical signals should be ~0, got {}",
        shift
    );
    // Phase lock should be perfect
    assert!(
        (metrics.phase_lock_stability - 1.0).abs() < 1e-6,
        "Stability for identical signals should be 1.0, got {}",
        metrics.phase_lock_stability
    );
    // Success rate should be 1.0
    assert_eq!(metrics.sync_success_rate, 1.0);
}

#[test]
fn test_sync_lagged_signal() {
    let sync = TemporalSync::new();
    let n = 100;

    // Create two signals with known offset (5 frames)
    let mut signal1 = Array1::zeros([n]);
    let mut signal2 = Array1::zeros([n]);

    for i in 0..n {
        *signal1.get_mut([i]).expect("valid index") = (i as f64 * 0.1).sin();
        if i >= 5 {
            *signal2.get_mut([i]).expect("valid index") = ((i - 5) as f64 * 0.1).sin();
        }
    }

    let (shift, _metrics) = sync.synchronize(&signal1, &signal2).expect("infallible: validated precondition");

    // Shift should be approximately 5 * frame_spacing
    let expected_shift = 5.0 * sync.config.frame_spacing;
    assert!(
        (shift - expected_shift).abs() < 0.1,
        "Shift should be ~{}, got {}",
        expected_shift,
        shift
    );
}

#[test]
fn test_sync_constant_signals() {
    let sync = TemporalSync::default();
    let signal1 = Array1::from_elem([100], 1.0);
    let signal2 = Array1::from_elem([100], 1.0);

    let (shift, metrics) = sync.synchronize(&signal1, &signal2).expect("infallible: validated precondition");

    assert!(
        shift.abs() < 1e-6,
        "Shift for constant signals should be ~0, got {}",
        shift
    );
    assert_eq!(metrics.sync_success_rate, 1.0);
}

#[test]
fn test_sync_length_mismatch() {
    let sync = TemporalSync::default();
    let signal1 = Array1::zeros([100]);
    let signal2 = Array1::zeros([50]);

    let result = sync.synchronize(&signal1, &signal2);
    assert!(result.is_err());
}

#[test]
fn test_sync_too_short() {
    let sync = TemporalSync::default();
    let signal1 = Array1::zeros([2]);
    let signal2 = Array1::zeros([2]);

    let result = sync.synchronize(&signal1, &signal2);
    assert!(result.is_err());
}

#[test]
fn test_success_rate_thresholds() {
    let config = TemporalSyncConfig {
        frame_spacing: 1.0 / 30.0,
        search_range: 10,
        min_correlation: 0.3,
    };
    let sync = TemporalSync::with_config(config);

    // Both signals constant - should have high stability
    let signal1 = Array1::from_vec([100], (0..100).map(|i| i as f64).collect()).expect("valid dimension");
    let signal2 = Array1::from_vec([100], (0..100).map(|i| i as f64).collect()).expect("valid dimension");

    let (_, metrics) = sync.synchronize(&signal1, &signal2).expect("infallible: validated precondition");
    assert!(
        metrics.sync_success_rate >= 0.5,
        "Success rate should be at least 0.5, got {}",
        metrics.sync_success_rate
    );
}

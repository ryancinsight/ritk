use leto::Array1;

use super::quality::aligned_residual_metrics;
use super::{
    TemporalSignal, TemporalSync, TemporalSyncConfig, TemporalSyncError, TemporalSyncStatus,
};

const SAMPLE_COUNT: usize = 256;
const INTEGER_DELAY: f64 = 6.0;
const ROUNDING_BOUND: f64 = 128.0 * f64::EPSILON;

fn waveform(sample: f64) -> f64 {
    let normalized = sample / SAMPLE_COUNT as f64;
    (normalized * 31.0).sin()
        + 0.37 * (normalized * 79.0 + 0.2).cos()
        + 0.19 * (normalized * 173.0 - 0.4).sin()
}

fn signal_with_delay(delay_frames: f64) -> Array1<f64> {
    let values = (0..SAMPLE_COUNT)
        .map(|index| waveform(index as f64 - delay_frames))
        .collect::<Vec<_>>();
    Array1::from_vec([SAMPLE_COUNT], values).expect("test signal shape is valid")
}

fn configured_sync(minimum_correlation: f64) -> TemporalSync {
    let config = TemporalSyncConfig::try_new(0.02, 16, minimum_correlation)
        .expect("test configuration is valid");
    TemporalSync::with_config(config)
}

#[test]
fn identical_signal_has_zero_shift_and_zero_residual() {
    let signal = signal_with_delay(0.0);
    let result = configured_sync(0.9)
        .synchronize(&signal, &signal)
        .expect("the analytical signal is identifiable");

    assert!(result.shift_frames().abs() <= ROUNDING_BOUND);
    assert!(result.shift_seconds().abs() <= ROUNDING_BOUND);
    assert!((result.peak_correlation() - 1.0).abs() <= ROUNDING_BOUND);
    assert_eq!(result.overlap_samples(), SAMPLE_COUNT);
    assert!(result.residual_rms() <= ROUNDING_BOUND);
    assert!(result.residual_max_abs() <= ROUNDING_BOUND);
    assert_eq!(result.status(), TemporalSyncStatus::Accepted);
}

#[test]
fn integer_delay_uses_positive_moving_lag_convention() {
    let reference = signal_with_delay(0.0);
    let moving = signal_with_delay(INTEGER_DELAY);
    let result = configured_sync(0.9)
        .synchronize(&reference, &moving)
        .expect("the delayed signal is identifiable");

    // The exact integer match has r=1. Adjacent samples are unequal, so the
    // normalized correlation peak is unique and interpolation stays local.
    assert!((result.shift_frames() - INTEGER_DELAY).abs() < 0.08);
    assert!((result.shift_seconds() - INTEGER_DELAY * 0.02).abs() < 0.0016);
    assert!((result.peak_correlation() - 1.0).abs() <= ROUNDING_BOUND);
    assert!(result.residual_rms() < 0.015);
    assert_eq!(result.status(), TemporalSyncStatus::Accepted);
}

#[test]
fn swapping_signals_negates_the_estimated_shift() {
    let reference = signal_with_delay(0.0);
    let moving = signal_with_delay(4.25);
    let synchronizer = configured_sync(0.8);

    let forward = synchronizer
        .synchronize(&reference, &moving)
        .expect("the delayed signal is identifiable");
    let reverse = synchronizer
        .synchronize(&moving, &reference)
        .expect("the delayed signal is identifiable");

    assert!((forward.shift_frames() + reverse.shift_frames()).abs() < 0.05);
}

#[test]
fn fractional_refinement_beats_integer_quantization() {
    let reference = signal_with_delay(0.0);
    let moving = signal_with_delay(4.25);
    let result = configured_sync(0.8)
        .synchronize(&reference, &moving)
        .expect("the delayed signal is identifiable");

    // Rounding the known quarter-frame delay to the selected integer peak
    // incurs 0.25 frame error. The three-point refinement must reduce it.
    assert!((result.shift_frames() - 4.25).abs() < 0.25);
}

#[test]
fn positive_affine_intensity_change_preserves_shift_and_correlation() {
    let reference = signal_with_delay(0.0);
    let moving = signal_with_delay(3.5);
    let transformed_values = moving
        .iter()
        .map(|value| value.mul_add(2.75, 4.0))
        .collect::<Vec<_>>();
    let transformed = Array1::from_vec([SAMPLE_COUNT], transformed_values)
        .expect("transformed signal shape is valid");
    let synchronizer = configured_sync(0.8);

    let original = synchronizer
        .synchronize(&reference, &moving)
        .expect("the delayed signal is identifiable");
    let affine = synchronizer
        .synchronize(&reference, &transformed)
        .expect("the affine signal is identifiable");

    assert!((original.shift_frames() - affine.shift_frames()).abs() < 1.0e-12);
    assert!((original.peak_correlation() - affine.peak_correlation()).abs() < 1.0e-12);
}

#[test]
fn configured_threshold_classifies_without_discarding_estimate() {
    let reference = signal_with_delay(0.0);
    let moving_values = signal_with_delay(2.0)
        .iter()
        .enumerate()
        .map(|(index, value)| value + 0.15 * (index as f64 * 0.73).sin())
        .collect::<Vec<_>>();
    let moving =
        Array1::from_vec([SAMPLE_COUNT], moving_values).expect("noisy signal shape is valid");
    let result = configured_sync(1.0)
        .synchronize(&reference, &moving)
        .expect("the noisy signal remains identifiable");

    assert!(result.peak_correlation() < 1.0);
    assert_eq!(
        result.status(),
        TemporalSyncStatus::BelowMinimumCorrelation {
            minimum_correlation: 1.0
        }
    );
    assert!(result.shift_frames().is_finite());
}

#[test]
fn allocated_profile_and_streaming_peak_agree() {
    let reference = signal_with_delay(0.0);
    let moving = signal_with_delay(-5.0);
    let synchronizer = configured_sync(0.8);
    let profile = synchronizer
        .correlation_profile(&reference, &moving)
        .expect("the delayed signal is identifiable");
    let result = synchronizer
        .synchronize(&reference, &moving)
        .expect("the delayed signal is identifiable");

    let best = profile
        .iter()
        .filter_map(|sample| {
            sample
                .correlation()
                .map(|correlation| (sample.lag_frames(), correlation))
        })
        .max_by(|(lag_a, correlation_a), (lag_b, correlation_b)| {
            correlation_a
                .total_cmp(correlation_b)
                .then_with(|| lag_b.unsigned_abs().cmp(&lag_a.unsigned_abs()))
                .then_with(|| lag_b.cmp(lag_a))
        })
        .expect("at least one profile lag is identifiable");

    assert_eq!(best.1, result.peak_correlation());
    assert!((result.shift_frames() - best.0 as f64).abs() <= 1.0);
}

#[test]
fn residual_metrics_use_interpolated_overlap_denominator() {
    let values = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let signal = Array1::from_vec([values.len()], values).expect("test signal shape is valid");
    let metrics = aligned_residual_metrics(&signal, &signal, 0.5);

    // Linear interpolation at 0.5, 1.5, 2.5, and 3.5 yields residuals
    // -0.5, -1.5, -2.5, and -3.5. Their squared sum is 21.
    assert_eq!(metrics.overlap_samples, 4);
    assert!((metrics.rms - (21.0_f64 / 4.0).sqrt()).abs() <= ROUNDING_BOUND);
    assert!((metrics.max_abs - 3.5).abs() <= ROUNDING_BOUND);
}

#[test]
fn configuration_rejects_invalid_domain_values() {
    assert!(matches!(
        TemporalSyncConfig::try_new(0.0, 3, 0.5),
        Err(TemporalSyncError::InvalidFrameSpacing { value: 0.0 })
    ));
    assert!(matches!(
        TemporalSyncConfig::try_new(f64::INFINITY, 3, 0.5),
        Err(TemporalSyncError::InvalidFrameSpacing { .. })
    ));
    assert!(matches!(
        TemporalSyncConfig::try_new(0.1, 0, 0.5),
        Err(TemporalSyncError::EmptySearchRange)
    ));
    assert!(matches!(
        TemporalSyncConfig::try_new(0.1, 3, -0.1),
        Err(TemporalSyncError::InvalidMinimumCorrelation { value: -0.1 })
    ));
    assert!(matches!(
        TemporalSyncConfig::try_new(0.1, 3, f64::NAN),
        Err(TemporalSyncError::InvalidMinimumCorrelation { .. })
    ));
}

#[test]
fn input_contract_reports_typed_failures() {
    let synchronizer = TemporalSync::default();
    let short = Array1::from_vec([2], vec![0.0, 1.0]).expect("test shape is valid");
    assert_eq!(
        synchronizer.synchronize(&short, &short),
        Err(TemporalSyncError::InsufficientSamples { length: 2 })
    );

    let reference = signal_with_delay(0.0);
    let shorter = Array1::from_vec([3], vec![0.0, 1.0, 0.0]).expect("test signal shape is valid");
    assert_eq!(
        synchronizer.synchronize(&reference, &shorter),
        Err(TemporalSyncError::LengthMismatch {
            reference: SAMPLE_COUNT,
            moving: 3
        })
    );

    let constant = Array1::from_elem([SAMPLE_COUNT], 2.0);
    assert_eq!(
        synchronizer.synchronize(&constant, &reference),
        Err(TemporalSyncError::UnidentifiableSignal {
            signal: TemporalSignal::Reference
        })
    );
}

#[test]
fn non_finite_sample_reports_signal_and_index_without_panicking() {
    let reference = signal_with_delay(0.0);
    let mut moving = signal_with_delay(2.0);
    *moving.get_mut([17]).expect("test index is in bounds") = f64::NAN;

    let error = TemporalSync::default()
        .synchronize(&reference, &moving)
        .expect_err("NaN must be rejected");
    assert!(matches!(
        error,
        TemporalSyncError::NonFiniteSample {
            signal: TemporalSignal::Moving,
            index: 17,
            value
        } if value.is_nan()
    ));
}

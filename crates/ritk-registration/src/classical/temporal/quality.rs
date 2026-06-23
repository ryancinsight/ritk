//! Temporal quality-metric computations (Sprint 354 split, ARCH-350-04).
//!
//! Extracted from `TemporalSync` so that the RMS/max-deviation timing error
//! and the success-rate threshold logic live in their own bounded context,
//! independent of the core cross-correlation synchronization algorithm.

use leto::Array1;

use super::config::TemporalSyncConfig;

/// Stability threshold above which synchronization is considered excellent.
const STABILITY_EXCELLENT: f64 = 0.8;
/// Stability threshold above which synchronization is considered acceptable.
const STABILITY_ACCEPTABLE: f64 = 0.5;
/// Stability threshold above which synchronization is considered marginal.
const STABILITY_MARGINAL: f64 = 0.3;
/// Deviation threshold scaling factor for acceptable range (2× the base threshold).
const DEV_SCALE_ACCEPTABLE: f64 = 2.0;
/// Success rate for excellent stability: 100%.
const SUCCESS_RATE_EXCELLENT: f64 = 1.0;
/// Success rate for acceptable stability: 80%.
const SUCCESS_RATE_ACCEPTABLE: f64 = 0.8;
/// Success rate for marginal stability: 50%.
const SUCCESS_RATE_MARGINAL: f64 = 0.5;

/// Compute RMS and maximum timing deviations after applying estimated shift.
///
/// # Arguments
/// * `signal1` - Reference temporal signal
/// * `signal2` - Moving temporal signal (after estimated shift)
/// * `shift_frames` - The estimated shift in frames
///
/// # Returns
/// Tuple `(rms_error, max_deviation)` in signal units.
pub(crate) fn compute_timing_errors(
    signal1: &Array1<f64>,
    signal2: &Array1<f64>,
    shift_frames: f64,
) -> (f64, f64) {
    let n = signal1.size();
    let mut squared_errors = 0.0;
    let mut max_deviation = 0.0;

    for i in 0..n {
        // Expected index in signal2 after shift
        let expected_idx = i as f64 - shift_frames;
        let nearest_idx = expected_idx.round() as isize;

        if nearest_idx >= 0 && nearest_idx < n as isize {
            let s2_val = *signal2.get([nearest_idx as usize]).expect("bounds checked");
            let error = *signal1.get([i]).expect("bounds checked") - s2_val;
            squared_errors += error * error;
            let abs_error = error.abs();
            if abs_error > max_deviation {
                max_deviation = abs_error;
            }
        }
    }

    let rms_error = (squared_errors / n as f64).sqrt();
    (rms_error, max_deviation)
}

/// Compute synchronization success rate based on stability and deviation.
///
/// # Arguments
/// * `stability` - Phase-lock stability ∈ [0, 1]
/// * `max_deviation` - Maximum timing deviation in signal units
/// * `config` - Temporal sync configuration (for the deviation threshold)
///
/// # Returns
/// Success rate ∈ {0.0, 0.5, 0.8, 1.0}.
pub(crate) fn compute_success_rate(
    stability: f64,
    max_deviation: f64,
    config: &TemporalSyncConfig,
) -> f64 {
    let dev_threshold = 0.1 * config.frame_spacing;

    if stability > STABILITY_EXCELLENT && max_deviation < dev_threshold {
        SUCCESS_RATE_EXCELLENT
    } else if stability > STABILITY_ACCEPTABLE
        && max_deviation < DEV_SCALE_ACCEPTABLE * dev_threshold
    {
        SUCCESS_RATE_ACCEPTABLE
    } else if stability > STABILITY_MARGINAL {
        SUCCESS_RATE_MARGINAL
    } else {
        0.0
    }
}

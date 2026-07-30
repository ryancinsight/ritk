//! Pearson-normalized lag correlation and constant-memory peak selection.

use core::cmp::Ordering;
use core::ops::RangeInclusive;

use leto::Array1;

use super::error::{Result, TemporalSignal, TemporalSyncError};
use super::result::TemporalCorrelationSample;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CorrelationPeak {
    pub(crate) lag_frames: f64,
    pub(crate) correlation: f64,
}

#[derive(Debug, Clone, Copy)]
struct IntegerPeak {
    lag: isize,
    correlation: f64,
    left: Option<f64>,
    right: Option<f64>,
}

pub(crate) fn validate_signals(reference: &Array1<f64>, moving: &Array1<f64>) -> Result<()> {
    let reference_length = reference.size();
    let moving_length = moving.size();
    if reference_length != moving_length {
        return Err(TemporalSyncError::LengthMismatch {
            reference: reference_length,
            moving: moving_length,
        });
    }
    if reference_length < 3 {
        return Err(TemporalSyncError::InsufficientSamples {
            length: reference_length,
        });
    }

    validate_signal(reference, TemporalSignal::Reference)?;
    validate_signal(moving, TemporalSignal::Moving)
}

fn validate_signal(signal: &Array1<f64>, identity: TemporalSignal) -> Result<()> {
    let mut count = 0.0;
    let mut mean = 0.0;
    let mut squared_deviations = 0.0;

    for (index, &sample) in signal.iter().enumerate() {
        if !sample.is_finite() {
            return Err(TemporalSyncError::NonFiniteSample {
                signal: identity,
                index,
                value: sample,
            });
        }

        count += 1.0;
        let delta = sample - mean;
        mean += delta / count;
        squared_deviations += delta * (sample - mean);
    }

    if squared_deviations > 0.0 {
        Ok(())
    } else {
        Err(TemporalSyncError::UnidentifiableSignal { signal: identity })
    }
}

pub(crate) fn find_peak(
    reference: &Array1<f64>,
    moving: &Array1<f64>,
    configured_search: usize,
) -> Result<CorrelationPeak> {
    let mut best: Option<IntegerPeak> = None;
    let mut previous: Option<(isize, f64)> = None;

    for lag in lag_range(reference.size(), configured_search) {
        let Some(correlation) = normalized_correlation(reference, moving, lag) else {
            previous = None;
            continue;
        };

        match best.as_mut() {
            None => {
                best = Some(IntegerPeak {
                    lag,
                    correlation,
                    left: None,
                    right: None,
                });
            }
            Some(peak) if better_peak(lag, correlation, peak.lag, peak.correlation) => {
                let left = previous
                    .filter(|(previous_lag, _)| *previous_lag + 1 == lag)
                    .map(|(_, value)| value);
                *peak = IntegerPeak {
                    lag,
                    correlation,
                    left,
                    right: None,
                };
            }
            Some(peak) if peak.lag + 1 == lag => {
                peak.right = Some(correlation);
            }
            Some(_) => {}
        }

        previous = Some((lag, correlation));
    }

    let peak = best.ok_or(TemporalSyncError::NoIdentifiableLag)?;
    Ok(CorrelationPeak {
        lag_frames: refine_peak(peak),
        correlation: peak.correlation,
    })
}

pub(crate) fn correlation_profile(
    reference: &Array1<f64>,
    moving: &Array1<f64>,
    configured_search: usize,
) -> Box<[TemporalCorrelationSample]> {
    lag_range(reference.size(), configured_search)
        .map(|lag| {
            TemporalCorrelationSample::new(lag, normalized_correlation(reference, moving, lag))
        })
        .collect()
}

fn lag_range(sample_count: usize, configured_search: usize) -> RangeInclusive<isize> {
    let search = configured_search.min(sample_count.saturating_sub(2));
    let search = isize::try_from(search)
        .expect("invariant: an allocated signal length cannot exceed isize::MAX");
    -search..=search
}

fn normalized_correlation(
    reference: &Array1<f64>,
    moving: &Array1<f64>,
    lag: isize,
) -> Option<f64> {
    let offset = lag.unsigned_abs();
    let overlap = reference.size().checked_sub(offset)?;
    let (reference_start, moving_start) = if lag >= 0 { (0, offset) } else { (offset, 0) };

    let mut count = 0.0;
    let mut reference_mean = 0.0;
    let mut moving_mean = 0.0;
    let mut reference_m2 = 0.0;
    let mut moving_m2 = 0.0;
    let mut covariance = 0.0;

    for overlap_index in 0..overlap {
        let reference_value = *reference
            .get([reference_start + overlap_index])
            .expect("invariant: overlap is bounded by the reference signal");
        let moving_value = *moving
            .get([moving_start + overlap_index])
            .expect("invariant: overlap is bounded by the moving signal");

        count += 1.0;
        let reference_delta = reference_value - reference_mean;
        reference_mean += reference_delta / count;
        let moving_delta = moving_value - moving_mean;
        moving_mean += moving_delta / count;
        reference_m2 += reference_delta * (reference_value - reference_mean);
        moving_m2 += moving_delta * (moving_value - moving_mean);
        covariance += reference_delta * (moving_value - moving_mean);
    }

    let denominator = (reference_m2 * moving_m2).sqrt();
    if denominator > 0.0 {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    } else {
        None
    }
}

fn better_peak(
    candidate_lag: isize,
    candidate_correlation: f64,
    current_lag: isize,
    current_correlation: f64,
) -> bool {
    match candidate_correlation.total_cmp(&current_correlation) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            candidate_lag
                .unsigned_abs()
                .cmp(&current_lag.unsigned_abs())
                .then_with(|| candidate_lag.cmp(&current_lag))
                == Ordering::Less
        }
    }
}

fn refine_peak(peak: IntegerPeak) -> f64 {
    let (Some(left), Some(right)) = (peak.left, peak.right) else {
        return peak.lag as f64;
    };

    let denominator = 2.0 * (left - 2.0 * peak.correlation + right);
    if !denominator.is_finite() || denominator >= 0.0 {
        return peak.lag as f64;
    }

    let offset = ((left - right) / denominator).clamp(-1.0, 1.0);
    if offset.is_finite() {
        peak.lag as f64 + offset
    } else {
        peak.lag as f64
    }
}

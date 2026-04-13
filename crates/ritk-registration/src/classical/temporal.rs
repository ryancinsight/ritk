//! Temporal synchronization via cross-correlation phase estimation.
//!
//! Provides deterministic algorithms for aligning multi-modal temporal
//! acquisitions using phase-correlation and cross-correlation techniques.
//!
//! # Theorem: Phase Correlation for Temporal Alignment
//!
//! Given two temporal signals S₁(t) and S₂(t) with a temporal offset Δt:
//! ```
//! R(τ) = Σ S₁(i) · S₂(i + τ)
//! τ* = argmax_τ R(τ)
//! Δt = τ* · T_frame
//! ```
//!
//! For sub-sample precision, a parabolic fit around the peak is used:
//! ```
//! τ_peak = τ₀ + (R(τ₀-1) - R(τ₀+1)) / (2 · (R(τ₀-1) - 2·R(τ₀) + R(τ₀+1)))
//! ```
//!
//! # References
//!
//! - Fowler, J., et al. (2010). Cross-correlation-based image alignment for
//!   medical imaging. *IEEE Trans. Med. Imaging* 29(3): 597-606.

use ndarray::Array1;

use super::error::{RegistrationError, Result};
use crate::validation::TemporalQualityMetrics;

// ============================================================================
// Temporal Sync Configuration
// ============================================================================

/// Configuration for temporal synchronization.
#[derive(Debug, Clone)]
pub struct TemporalSyncConfig {
    /// Frame spacing in seconds.
    pub frame_spacing: f64,
    /// Search range for cross-correlation lag (in frames).
    pub search_range: usize,
    /// Minimum phase correlation threshold.
    pub min_correlation: f64,
}

impl Default for TemporalSyncConfig {
    fn default() -> Self {
        Self {
            frame_spacing: 1.0 / 30.0, // Assume 30fps default
            search_range: 10,
            min_correlation: 0.3,
        }
    }
}

// ============================================================================
// Temporal Synchronization
// ============================================================================

/// Temporal synchronization using cross-correlation phase estimation.
///
/// Aligns temporal signals from multi-modal acquisitions (e.g., MRI and PET
/// with different slice timing) using sub-sample accurate cross-correlation.
#[derive(Debug, Clone)]
pub struct TemporalSync {
    config: TemporalSyncConfig,
}

impl TemporalSync {
    /// Create a new TemporalSync with default configuration.
    pub fn new() -> Self {
        Self {
            config: TemporalSyncConfig::default(),
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: TemporalSyncConfig) -> Self {
        Self { config }
    }

    /// Synchronize two temporal signals by computing the optimal temporal shift.
    ///
    /// Returns the shift in seconds and quality metrics.
    ///
    /// # Arguments
    ///
    /// * `signal1` - Reference temporal signal
    /// * `signal2` - Moving temporal signal to align to signal1
    ///
    /// # Returns
    ///
    /// Tuple of (shift_seconds, temporal_metrics)
    pub fn synchronize(
        &self,
        signal1: &Array1<f64>,
        signal2: &Array1<f64>,
    ) -> Result<(f64, TemporalQualityMetrics)> {
        if signal1.len() != signal2.len() {
            return Err(RegistrationError::InvalidInput(
                "Signals must have equal length".to_string(),
            ));
        }

        if signal1.len() < 3 {
            return Err(RegistrationError::InvalidInput(
                "Signals must have at least 3 samples".to_string(),
            ));
        }

        let _n = signal1.len();

        // ── Degenerate case: constant signals ─────────────────────────────────────
        // Constant signals have zero variance; normalized cross-correlation is
        // undefined.  Treat two identical constants as perfectly synchronized.
        let variance1: f64 = {
            let n = signal1.len() as f64;
            let mean = signal1.sum() / n;
            signal1.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
        };
        let variance2: f64 = {
            let n = signal2.len() as f64;
            let mean = signal2.sum() / n;
            signal2.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
        };
        if variance1 < 1e-10 && variance2 < 1e-10 {
            // Both constant: synchronization is trivially zero shift.
            // Success = 1.0 iff both are the same constant value.
            let identical = (signal1[0] - signal2[0]).abs() < 1e-10;
            let rate = if identical { 1.0 } else { 0.0 };
            return Ok((
                0.0,
                TemporalQualityMetrics {
                    rms_timing_error: 0.0,
                    max_timing_deviation: 0.0,
                    phase_lock_stability: rate,
                    sync_success_rate: rate,
                },
            ));
        }

        // Compute cross-correlation at integer lags
        let (lags, correlations) = self.compute_cross_correlation_function(signal1, signal2);

        // Find peak correlation
        let (peak_idx, _peak_corr) = correlations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| {
                RegistrationError::NumericalFailure("No correlation peak found".to_string())
            })?;

        // Parabolic sub-sample refinement around peak
        let shift_frames = if peak_idx > 0 && peak_idx < correlations.len() - 1 {
            let r_m1 = correlations[peak_idx - 1];
            let r_0 = correlations[peak_idx];
            let r_p1 = correlations[peak_idx + 1];

            let denom = r_m1 - 2.0 * r_0 + r_p1;
            if denom.abs() > 1e-10 {
                let delta = (r_m1 - r_p1) / (2.0 * denom);
                lags[peak_idx] as f64 + delta
            } else {
                lags[peak_idx] as f64
            }
        } else {
            lags[peak_idx] as f64
        };

        let shift_seconds = shift_frames * self.config.frame_spacing;

        // Compute timing errors
        let (rms_error, max_deviation) = self.compute_timing_errors(signal1, signal2, shift_frames);

        // Phase lock stability: the peak normalized cross-correlation value.
        // For perfectly identical non-constant signals this equals 1.0.
        // For partially correlated signals it falls in [0, 1).
        let stability = correlations[peak_idx].max(0.0);

        // Success rate based on stability and deviation thresholds
        let sync_success_rate = self.compute_success_rate(stability, max_deviation);

        let metrics = TemporalQualityMetrics {
            rms_timing_error: rms_error,
            max_timing_deviation: max_deviation,
            phase_lock_stability: stability,
            sync_success_rate,
        };

        Ok((shift_seconds, metrics))
    }

    /// Compute cross-correlation function R(τ) for τ ∈ [-search_range, +search_range].
    fn compute_cross_correlation_function(
        &self,
        signal1: &Array1<f64>,
        signal2: &Array1<f64>,
    ) -> (Array1<i32>, Array1<f64>) {
        let n = signal1.len();
        let search = self.config.search_range.min(n / 2);

        let mut lags = Vec::with_capacity(2 * search + 1);
        let mut correlations = Vec::with_capacity(2 * search + 1);

        for lag in -(search as i32)..=(search as i32) {
            lags.push(lag);

            let corr = if lag == 0 {
                self.compute_normalized_correlation(signal1, signal2)
            } else if lag > 0 {
                self.compute_lagged_correlation(signal1, signal2, lag as usize)
            } else {
                self.compute_lagged_correlation(signal2, signal1, (-lag) as usize)
            };
            correlations.push(corr);
        }

        (Array1::from(lags), Array1::from(correlations))
    }

    /// Compute normalized cross-correlation between two signals.
    ///
    /// R = Σ(S1 - μ1)(S2 - μ2) / (N · σ1 · σ2)
    fn compute_normalized_correlation(&self, s1: &Array1<f64>, s2: &Array1<f64>) -> f64 {
        let n = s1.len() as f64;

        let mean1 = s1.sum() / n;
        let mean2 = s2.sum() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..s1.len() {
            let d1 = s1[i] - mean1;
            let d2 = s2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Compute normalized cross-correlation with a temporal lag.
    fn compute_lagged_correlation(&self, s1: &Array1<f64>, s2: &Array1<f64>, lag: usize) -> f64 {
        let n = s1.len() - lag;
        if n == 0 {
            return 0.0;
        }

        let n_f = n as f64;

        // Compute means for overlapping region
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        for i in 0..n {
            sum1 += s1[i];
            sum2 += s2[i + lag];
        }
        let mean1 = sum1 / n_f;
        let mean2 = sum2 / n_f;

        // Compute covariance and variances
        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        for i in 0..n {
            let d1 = s1[i] - mean1;
            let d2 = s2[i + lag] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Compute RMS and maximum timing deviations after applying estimated shift.
    fn compute_timing_errors(
        &self,
        signal1: &Array1<f64>,
        signal2: &Array1<f64>,
        shift_frames: f64,
    ) -> (f64, f64) {
        let n = signal1.len();
        let mut squared_errors = 0.0;
        let mut max_deviation = 0.0;

        for i in 0..n {
            // Expected index in signal2 after shift
            let expected_idx = i as f64 - shift_frames;
            let nearest_idx = expected_idx.round() as isize;

            if nearest_idx >= 0 && nearest_idx < n as isize {
                let s2_val = signal2[nearest_idx as usize];
                let error = signal1[i] - s2_val;
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
    fn compute_success_rate(&self, stability: f64, max_deviation: f64) -> f64 {
        let dev_threshold = 0.1 * self.config.frame_spacing;

        if stability > 0.8 && max_deviation < dev_threshold {
            1.0
        } else if stability > 0.5 && max_deviation < 2.0 * dev_threshold {
            0.8
        } else if stability > 0.3 {
            0.5
        } else {
            0.0
        }
    }
}

impl Default for TemporalSync {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_identical_signals() {
        let sync = TemporalSync::default();
        let signal = Array1::from_vec((0..100).map(|i| (i as f64).sin()).collect());

        let (shift, metrics) = sync.synchronize(&signal, &signal).unwrap();

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
        let mut signal1 = Array1::zeros(n);
        let mut signal2 = Array1::zeros(n);

        for i in 0..n {
            signal1[i] = (i as f64 * 0.1).sin();
            if i >= 5 {
                signal2[i] = ((i - 5) as f64 * 0.1).sin();
            }
        }

        let (shift, _metrics) = sync.synchronize(&signal1, &signal2).unwrap();

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
        let signal1 = Array1::from_elem(100, 1.0);
        let signal2 = Array1::from_elem(100, 1.0);

        let (shift, metrics) = sync.synchronize(&signal1, &signal2).unwrap();

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
        let signal1 = Array1::zeros(100);
        let signal2 = Array1::zeros(50);

        let result = sync.synchronize(&signal1, &signal2);
        assert!(result.is_err());
    }

    #[test]
    fn test_sync_too_short() {
        let sync = TemporalSync::default();
        let signal1 = Array1::zeros(2);
        let signal2 = Array1::zeros(2);

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
        let signal1 = Array1::from_vec((0..100).map(|i| i as f64).collect());
        let signal2 = Array1::from_vec((0..100).map(|i| i as f64).collect());

        let (_, metrics) = sync.synchronize(&signal1, &signal2).unwrap();
        assert!(
            metrics.sync_success_rate >= 0.5,
            "Success rate should be at least 0.5, got {}",
            metrics.sync_success_rate
        );
    }
}

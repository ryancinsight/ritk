//! Dimensionally explicit temporal synchronization outputs.

/// Whether the measured correlation satisfies the configured acceptance bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalSyncStatus {
    /// The peak correlation meets or exceeds the configured threshold.
    Accepted,
    /// The estimate is measurable but does not satisfy the configured threshold.
    BelowMinimumCorrelation {
        /// Threshold used for the classification.
        minimum_correlation: f64,
    },
}

/// A normalized-correlation diagnostic at one integer lag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalCorrelationSample {
    lag_frames: isize,
    correlation: Option<f64>,
}

impl TemporalCorrelationSample {
    pub(crate) const fn new(lag_frames: isize, correlation: Option<f64>) -> Self {
        Self {
            lag_frames,
            correlation,
        }
    }

    /// Integer lag in frames.
    #[must_use]
    pub const fn lag_frames(&self) -> isize {
        self.lag_frames
    }

    /// Pearson-normalized correlation, or `None` for a locally flat overlap.
    #[must_use]
    pub const fn correlation(&self) -> Option<f64> {
        self.correlation
    }
}

/// Measured temporal shift, acceptance status, and aligned residual diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalSyncResult {
    shift_frames: f64,
    shift_seconds: f64,
    peak_correlation: f64,
    overlap_samples: usize,
    residual_rms: f64,
    residual_max_abs: f64,
    status: TemporalSyncStatus,
}

impl TemporalSyncResult {
    pub(crate) const fn new(
        shift_frames: f64,
        shift_seconds: f64,
        peak_correlation: f64,
        overlap_samples: usize,
        residual_rms: f64,
        residual_max_abs: f64,
        status: TemporalSyncStatus,
    ) -> Self {
        Self {
            shift_frames,
            shift_seconds,
            peak_correlation,
            overlap_samples,
            residual_rms,
            residual_max_abs,
            status,
        }
    }

    /// Estimated lag in frames.
    ///
    /// A positive value means the moving signal is delayed and must be sampled
    /// at `reference_index + shift_frames` to align it to the reference.
    #[must_use]
    pub const fn shift_frames(&self) -> f64 {
        self.shift_frames
    }

    /// Estimated lag in seconds.
    #[must_use]
    pub const fn shift_seconds(&self) -> f64 {
        self.shift_seconds
    }

    /// Pearson-normalized correlation at the selected integer peak.
    #[must_use]
    pub const fn peak_correlation(&self) -> f64 {
        self.peak_correlation
    }

    /// Number of reference samples contributing to the residual metrics.
    #[must_use]
    pub const fn overlap_samples(&self) -> usize {
        self.overlap_samples
    }

    /// Root-mean-square aligned residual in signal units.
    #[must_use]
    pub const fn residual_rms(&self) -> f64 {
        self.residual_rms
    }

    /// Maximum absolute aligned residual in signal units.
    #[must_use]
    pub const fn residual_max_abs(&self) -> f64 {
        self.residual_max_abs
    }

    /// Acceptance classification under the configured minimum correlation.
    #[must_use]
    pub const fn status(&self) -> TemporalSyncStatus {
        self.status
    }
}

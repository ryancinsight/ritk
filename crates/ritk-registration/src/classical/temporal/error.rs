//! Typed failures for temporal synchronization.

use thiserror::Error;

/// Identifies an input signal in a temporal synchronization failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalSignal {
    /// The fixed/reference signal.
    Reference,
    /// The signal being shifted into alignment.
    Moving,
}

impl core::fmt::Display for TemporalSignal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reference => formatter.write_str("reference"),
            Self::Moving => formatter.write_str("moving"),
        }
    }
}

/// A temporal synchronization configuration or input failure.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum TemporalSyncError {
    /// Frame spacing must be finite and strictly positive.
    #[error("frame spacing must be finite and positive, got {value}")]
    InvalidFrameSpacing {
        /// Rejected spacing in seconds.
        value: f64,
    },
    /// At least one non-zero lag must be searched.
    #[error("temporal search range must be greater than zero")]
    EmptySearchRange,
    /// The acceptance threshold must be a finite normalized correlation.
    #[error("minimum correlation must be finite and within [0, 1], got {value}")]
    InvalidMinimumCorrelation {
        /// Rejected threshold.
        value: f64,
    },
    /// Reference and moving signals must describe the same sample count.
    #[error(
        "temporal signals must have equal lengths, got reference={reference} and moving={moving}"
    )]
    LengthMismatch {
        /// Reference signal length.
        reference: usize,
        /// Moving signal length.
        moving: usize,
    },
    /// Correlation requires enough samples to define a peak neighborhood.
    #[error("temporal signals require at least three samples, got {length}")]
    InsufficientSamples {
        /// Rejected sample count.
        length: usize,
    },
    /// Every sample must be finite.
    #[error("{signal} signal sample {index} must be finite, got {value}")]
    NonFiniteSample {
        /// Input containing the rejected sample.
        signal: TemporalSignal,
        /// Zero-based sample index.
        index: usize,
        /// Rejected sample value.
        value: f64,
    },
    /// A constant or numerically flat signal has no identifiable correlation lag.
    #[error("{signal} signal has zero representable variance")]
    UnidentifiableSignal {
        /// Input whose lag cannot be identified.
        signal: TemporalSignal,
    },
    /// No searched overlap contained enough variance to define correlation.
    #[error("no searched lag has an identifiable normalized correlation")]
    NoIdentifiableLag,
}

/// Internal result alias for temporal synchronization operations.
pub(crate) type Result<T> = core::result::Result<T, TemporalSyncError>;

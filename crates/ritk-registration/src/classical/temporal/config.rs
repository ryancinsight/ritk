//! Validated temporal synchronization configuration.

use core::num::NonZeroUsize;

use super::error::{Result, TemporalSyncError};

#[derive(Debug, Clone, Copy)]
struct FrameSpacing(f64);

impl FrameSpacing {
    fn try_new(value: f64) -> Result<Self> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(TemporalSyncError::InvalidFrameSpacing { value })
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MinimumCorrelation(f64);

impl MinimumCorrelation {
    fn try_new(value: f64) -> Result<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(TemporalSyncError::InvalidMinimumCorrelation { value })
        }
    }
}

/// Validated configuration for temporal synchronization.
#[derive(Debug, Clone, Copy)]
pub struct TemporalSyncConfig {
    frame_spacing: FrameSpacing,
    search_range: NonZeroUsize,
    minimum_correlation: MinimumCorrelation,
}

impl TemporalSyncConfig {
    /// Construct a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TemporalSyncError`] when frame spacing is not finite and
    /// positive, the search range is zero, or the correlation threshold is
    /// not finite and within `[0, 1]`.
    pub fn try_new(
        frame_spacing_seconds: f64,
        search_range_frames: usize,
        minimum_correlation: f64,
    ) -> Result<Self> {
        let frame_spacing = FrameSpacing::try_new(frame_spacing_seconds)?;
        let search_range =
            NonZeroUsize::new(search_range_frames).ok_or(TemporalSyncError::EmptySearchRange)?;
        let minimum_correlation = MinimumCorrelation::try_new(minimum_correlation)?;

        Ok(Self {
            frame_spacing,
            search_range,
            minimum_correlation,
        })
    }

    /// Time between adjacent samples, in seconds.
    #[must_use]
    pub const fn frame_spacing_seconds(&self) -> f64 {
        self.frame_spacing.0
    }

    /// Maximum integer lag searched in either direction, in frames.
    #[must_use]
    pub const fn search_range_frames(&self) -> usize {
        self.search_range.get()
    }

    /// Minimum normalized correlation accepted by the result classifier.
    #[must_use]
    pub const fn minimum_correlation(&self) -> f64 {
        self.minimum_correlation.0
    }
}

impl Default for TemporalSyncConfig {
    fn default() -> Self {
        Self {
            frame_spacing: FrameSpacing(1.0 / 30.0),
            search_range: NonZeroUsize::new(10)
                .expect("invariant: the default temporal search range is non-zero"),
            minimum_correlation: MinimumCorrelation(0.3),
        }
    }
}

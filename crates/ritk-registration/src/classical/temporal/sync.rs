//! Temporal synchronization by lagged Pearson cross-correlation.
//!
//! For each integer lag `k`, the algorithm evaluates Pearson-normalized
//! correlation over the valid signal overlap. Positive `k` means the moving
//! signal is delayed, so residual evaluation samples it at
//! `reference_index + k`. The selected integer peak is refined with a bounded
//! three-point parabola when both adjacent correlations are identifiable.
//!
//! The method and lag convention follow Xiao, Ding, and Hu, *Journal of
//! Imaging* 8(5), 120 (2022), Section 2.3, Equation 1, and Algorithm 1:
//! <https://pmc.ncbi.nlm.nih.gov/articles/PMC9145353/#sec2dot3-jimaging-08-00120>.
//! The sub-sample estimate is an interpolation, not an exact reconstruction;
//! its bias is characterized by Céspedes et al., *Ultrasonic Imaging* 17(2),
//! 142–171 (1995): <https://doi.org/10.1006/uimg.1995.1007>.

use leto::Array1;

use super::config::TemporalSyncConfig;
use super::correlation::{correlation_profile, find_peak, validate_signals};
use super::error::Result;
use super::quality::aligned_residual_metrics;
use super::result::{TemporalCorrelationSample, TemporalSyncResult, TemporalSyncStatus};

/// Temporal synchronization using normalized cross-correlation.
#[derive(Debug, Clone)]
pub struct TemporalSync {
    config: TemporalSyncConfig,
}

impl TemporalSync {
    /// Create a synchronizer with the validated default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TemporalSyncConfig::default(),
        }
    }

    /// Create a synchronizer with an explicit validated configuration.
    #[must_use]
    pub const fn with_config(config: TemporalSyncConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    #[must_use]
    pub const fn config(&self) -> &TemporalSyncConfig {
        &self.config
    }

    /// Estimate the moving signal's delay relative to the reference.
    ///
    /// Peak search uses constant scratch space. Use
    /// [`Self::correlation_profile`] only when an allocated diagnostic profile
    /// is required.
    ///
    /// # Errors
    ///
    /// Returns [`super::error::TemporalSyncError`] for mismatched or short inputs,
    /// non-finite samples, unidentifiable zero-variance signals, or when no
    /// searched overlap has a defined correlation.
    pub fn synchronize(
        &self,
        reference: &Array1<f64>,
        moving: &Array1<f64>,
    ) -> Result<TemporalSyncResult> {
        validate_signals(reference, moving)?;
        let peak = find_peak(reference, moving, self.config.search_range_frames())?;
        let residuals = aligned_residual_metrics(reference, moving, peak.lag_frames);
        let minimum_correlation = self.config.minimum_correlation();
        let status = if peak.correlation >= minimum_correlation {
            TemporalSyncStatus::Accepted
        } else {
            TemporalSyncStatus::BelowMinimumCorrelation {
                minimum_correlation,
            }
        };

        Ok(TemporalSyncResult::new(
            peak.lag_frames,
            peak.lag_frames * self.config.frame_spacing_seconds(),
            peak.correlation,
            residuals.overlap_samples,
            residuals.rms,
            residuals.max_abs,
            status,
        ))
    }

    /// Allocate the normalized correlation profile over the configured lags.
    ///
    /// A sample has no correlation when its local overlap is constant. The
    /// synchronizer ignores such samples during peak selection.
    ///
    /// # Errors
    ///
    /// Returns [`super::error::TemporalSyncError`] under the same input
    /// validation contract as [`Self::synchronize`].
    pub fn correlation_profile(
        &self,
        reference: &Array1<f64>,
        moving: &Array1<f64>,
    ) -> Result<Box<[TemporalCorrelationSample]>> {
        validate_signals(reference, moving)?;
        Ok(correlation_profile(
            reference,
            moving,
            self.config.search_range_frames(),
        ))
    }
}

impl Default for TemporalSync {
    fn default() -> Self {
        Self::new()
    }
}

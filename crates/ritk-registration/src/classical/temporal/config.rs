//! Temporal synchronization configuration (Sprint 354 split, ARCH-350-04).
//!
//! Extracted from the monolithic `classical/temporal.rs` so that the config
//! struct, the core [`TemporalSync`](super::sync::TemporalSync) algorithm,
//! and the quality-metric computations each live in their own bounded
//! context.

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

//! Shared sampling utility for similarity metrics (Sprint 354, DRY-354-01).
//!
//! Centralizes the `with_sampling` clamping logic and the
//! `sampling_percentage → Some/None` encoding that was previously duplicated
//! across `MutualInformation` and `CorrelationRatio`.
//!
//! # Usage
//!
//! ```ignore
//! use crate::metric::sampling::{SamplingConfig, SamplingMode, resolve_n_points};
//!
//! let cfg = SamplingConfig::new(0.20, SamplingMode::Uniform);
//! let n = resolve_n_points(cfg, 10_000); // → 2_000
//!
//! // Clamp to ≥ 1 to avoid degenerate empty batches:
//! assert!(n >= 1);
//! ```
//!
//! # Backward compatibility
//!
//! Existing call sites that use `with_sampling(0.20)` continue to work — the
//! `SamplingConfig::from_percentage(0.20)` constructor applies the same clamp
//! as the previous inline logic: `clamp(1e-4, 1.0)`, then `None` if `>= 1.0`.

/// Sampling mode for stochastic / mask-based sample selection.
///
/// * `Uniform` — randomly subsample `percentage * total` points from the
///   full grid (Mattes MI default).
/// * `Mask`   — use the caller-supplied foreground mask points directly;
///   `percentage` is ignored (or used as a per-mask subsample ratio).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplingMode {
    /// Uniform random subsample of the full grid.
    #[default]
    Uniform,
    /// Caller-supplied foreground mask points (e.g. `fixed_mask_points`).
    Mask,
}

/// Sampling configuration for similarity metrics.
///
/// Encapsulates the previously-duplicated `sampling_percentage: Option<f32>`
/// encoding used by `MutualInformation` and `CorrelationRatio`:
/// * `percentage >= 1.0`  → no sampling (full grid)
/// * `percentage ∈ (0, 1)` → stochastic subsample
/// * `percentage <= 0.0`  → clamped to `1e-4` (degenerate, but not zero)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Fraction of points to sample, ∈ (0, 1].  Stored as `Some(p)` when
    /// sampling is active, `None` when full-grid is used.
    pub percentage: Option<f32>,
    /// Sampling mode (uniform vs. mask).
    pub mode: SamplingMode,
}

impl SamplingConfig {
    /// Create a new `SamplingConfig` from a raw percentage (clamped).
    ///
    /// # Arguments
    /// * `percentage` - Sampling fraction ∈ (0, 1]. Values outside this range
    ///   are clamped: `<= 0.0` → `1e-4`, `> 1.0` → `1.0` (which disables
    ///   sampling — the field is stored as `None`).
    pub fn new(percentage: f32, mode: SamplingMode) -> Self {
        Self {
            percentage: Self::clamp_percentage(percentage),
            mode,
        }
    }

    /// Convenience: uniform sampling (the historical default).
    pub fn uniform(percentage: f32) -> Self {
        Self::new(percentage, SamplingMode::Uniform)
    }

    /// Mask mode — `percentage` is ignored.
    pub fn mask() -> Self {
        Self {
            percentage: None,
            mode: SamplingMode::Mask,
        }
    }

    /// Returns `true` if sampling is active (any percentage < 1.0).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.percentage.is_some()
    }

    /// Returns the raw percentage, or `None` if no sampling.
    #[inline]
    pub fn percentage(&self) -> Option<f32> {
        self.percentage
    }

    /// Builder-style setter (mirrors the old `with_sampling` pattern).
    pub fn with_percentage(mut self, percentage: f32) -> Self {
        self.percentage = Self::clamp_percentage(percentage);
        self
    }

    /// Builder-style setter for mode.
    pub fn with_mode(mut self, mode: SamplingMode) -> Self {
        self.mode = mode;
        self
    }

    /// The single source of truth for percentage clamping.
    ///
    /// Rules (matching the prior inline logic in `MutualInformation::with_sampling`):
    /// 1. Clamp to `[1e-4, 1.0]`.
    /// 2. If the clamped value is `>= 1.0`, return `None` (sampling disabled).
    /// 3. Otherwise, return `Some(clamped)`.
    #[inline]
    fn clamp_percentage(percentage: f32) -> Option<f32> {
        let clamped = percentage.clamp(1e-4, 1.0);
        if clamped >= 1.0 {
            None
        } else {
            Some(clamped)
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        // 20% uniform sampling matches the historical default for Mattes MI.
        Self::uniform(0.20)
    }
}

/// Resolve the number of sample points for a given total grid size.
///
/// # Arguments
/// * `config` - The sampling configuration
/// * `total`  - Total number of points in the full grid
///
/// # Returns
/// The number of points to sample. Guaranteed `>= 1` (even when the
/// percentage would round to zero on a small grid) so the caller never
/// has to handle an empty batch.
///
/// # Behavior
/// * `config.percentage = None` → `total` (no sampling)
/// * `config.percentage = Some(p)` → `max(1, (p * total) as usize)`
/// * `config.mode = Mask` → `total` (caller provides the points; the
///   `percentage` field is ignored in mask mode)
#[inline]
pub fn resolve_n_points(config: &SamplingConfig, total: usize) -> usize {
    if total == 0 {
        return 0;
    }
    match config.mode {
        SamplingMode::Mask => total,
        SamplingMode::Uniform => match config.percentage {
            None => total,
            Some(p) => {
                let n = (p * total as f32) as usize;
                n.max(1)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_disables_sampling_at_one() {
        // p = 1.0 disables sampling (stores None, matches prior behavior)
        let cfg = SamplingConfig::uniform(1.0);
        assert_eq!(cfg.percentage, None);
        assert!(!cfg.is_active());
    }

    #[test]
    fn clamp_clamps_negative_to_epsilon() {
        // p < 0 → clamped to 1e-4 (matches prior clamp(1e-4, 1.0))
        let cfg = SamplingConfig::uniform(-0.5);
        assert!(cfg.percentage.is_some());
        assert!((cfg.percentage.unwrap() - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn clamp_clamps_above_one() {
        // p > 1 → clamped to 1.0 → stored as None
        let cfg = SamplingConfig::uniform(1.5);
        assert_eq!(cfg.percentage, None);
    }

    #[test]
    fn resolve_n_points_full_grid_when_disabled() {
        let cfg = SamplingConfig::uniform(1.0);
        assert_eq!(resolve_n_points(&cfg, 10_000), 10_000);
    }

    #[test]
    fn resolve_n_points_uniform_sampling() {
        let cfg = SamplingConfig::uniform(0.20);
        assert_eq!(resolve_n_points(&cfg, 10_000), 2_000);
        assert_eq!(resolve_n_points(&cfg, 5_000), 1_000);
    }

    #[test]
    fn resolve_n_points_floor_at_one() {
        // 1e-4 * 5 = 0.0005 → would truncate to 0; must clamp to 1
        let cfg = SamplingConfig::uniform(1e-4);
        assert_eq!(resolve_n_points(&cfg, 5), 1);
    }

    #[test]
    fn resolve_n_points_mask_mode_ignores_percentage() {
        let cfg = SamplingConfig::mask();
        assert_eq!(resolve_n_points(&cfg, 1_000), 1_000);
    }

    #[test]
    fn resolve_n_points_zero_total_returns_zero() {
        let cfg = SamplingConfig::uniform(0.20);
        assert_eq!(resolve_n_points(&cfg, 0), 0);
    }
}

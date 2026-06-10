//! Temporal synchronization module (Sprint 354 split, ARCH-350-04).
//!
//! Deep vertical hierarchy:
//!
//! ```text
//! temporal/
//! ├── mod.rs       (this file — module decls + re-exports)
//! ├── config.rs    TemporalSyncConfig
//! ├── sync.rs      TemporalSync + cross-correlation + synchronize()
//! ├── quality.rs   compute_timing_errors + compute_success_rate
//! └── tests.rs     #[cfg(test)] tests
//! ```
//!
//! Public surface unchanged: `TemporalSync` and `TemporalSyncConfig` still
//! re-export from this module so external callers (e.g. `classical::mod.rs`)
//! continue to work without modification.

pub mod config;
pub mod quality;
pub mod sync;

#[cfg(test)]
mod tests;

// ── Public re-exports (preserve public API) ───────────────────────────────
pub use config::TemporalSyncConfig;
pub use sync::TemporalSync;

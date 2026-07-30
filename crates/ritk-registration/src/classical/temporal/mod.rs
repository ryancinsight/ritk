//! Temporal signal synchronization with dimensionally explicit diagnostics.

mod config;
mod correlation;
mod error;
mod quality;
mod result;
mod sync;

#[cfg(test)]
mod tests;

pub use config::TemporalSyncConfig;
pub use error::{TemporalSignal, TemporalSyncError};
pub use result::{TemporalCorrelationSample, TemporalSyncResult, TemporalSyncStatus};
pub use sync::TemporalSync;

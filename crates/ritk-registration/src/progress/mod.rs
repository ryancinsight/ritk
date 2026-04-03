//! Progress tracking and callbacks for registration workflows.
//!
//! This module provides progress tracking, callbacks, and monitoring
//! capabilities for registration operations.

pub mod console;
pub mod early_stopping;
pub mod history;
pub mod info;
pub mod tracker;

pub use console::ConsoleProgressCallback;
pub use early_stopping::EarlyStoppingCallback;
pub use history::HistoryCallback;
pub use info::ProgressInfo;
pub use tracker::ProgressTracker;

/// Progress callback trait for monitoring registration progress.
pub trait ProgressCallback: Send + Sync {
    /// Called at each iteration with progress information.
    fn on_progress(&self, info: &ProgressInfo);

    /// Called when registration starts.
    fn on_start(&self) {
        // Default: no-op
    }

    /// Called when registration completes successfully.
    fn on_complete(&self, _info: &ProgressInfo) {
        // Default: no-op
    }

    /// Called when registration fails.
    fn on_error(&self, _error: &str) {
        // Default: no-op
    }
}

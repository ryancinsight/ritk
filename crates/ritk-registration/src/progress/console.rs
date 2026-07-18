use crate::progress::{ProgressCallback, ProgressInfo};
use std::borrow::Cow;

/// Whether the console progress callback renders a progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressDisplay {
    /// Render a progress bar.
    #[default]
    WithBar,
    /// Log iteration info without a progress bar.
    Silent }

/// Console progress callback that logs to tracing.
#[derive(Debug, Clone)]
pub struct ConsoleProgressCallback {
    /// Log interval (iterations).
    pub log_interval: usize,
    /// Progress display mode.
    pub progress_display: ProgressDisplay }

impl Default for ConsoleProgressCallback {
    fn default() -> Self {
        Self {
            log_interval: 50,
            progress_display: ProgressDisplay::WithBar }
    }
}

impl ConsoleProgressCallback {
    /// Create a new console progress callback.
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            progress_display: ProgressDisplay::WithBar }
    }

    /// Disable progress bar.
    pub fn without_progress_bar(mut self) -> Self {
        self.progress_display = ProgressDisplay::Silent;
        self
    }
}

impl ProgressCallback for ConsoleProgressCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        if info.iteration.is_multiple_of(self.log_interval)
            || info.total_iterations == Some(info.iteration)
        {
            let progress = info.progress_percent().unwrap_or(0.0);
            let elapsed = format!("{:.2}s", info.elapsed.as_secs_f64());
            let remaining: Cow<str> = info
                .estimated_remaining
                .map(|d| Cow::Owned(format!("{:.2}s", d.as_secs_f64())))
                .unwrap_or(Cow::Borrowed("N/A"));
            let total_str: Cow<str> = info
                .total_iterations
                .map(|n| Cow::Owned(n.to_string()))
                .unwrap_or(Cow::Borrowed("?"));

            tracing::info!(
                "Iter {}/{} ({:.1}%) | Loss: {:.6} | LR: {:.2e} | Elapsed: {} | ETA: {}",
                info.iteration,
                total_str,
                progress,
                info.loss,
                info.learning_rate,
                elapsed,
                remaining
            );

            // Log additional metrics
            for (name, value) in &info.metrics {
                tracing::info!("  {}: {:.6}", name, value);
            }
        }
    }

    fn on_start(&self) {
        tracing::info!("Registration started");
    }

    fn on_complete(&self, info: &ProgressInfo) {
        tracing::info!(
            "Registration completed in {:.2}s with final loss: {:.6}",
            info.elapsed.as_secs_f64(),
            info.loss
        );
    }

    fn on_error(&self, error: &str) {
        tracing::error!("Registration failed: {}", error);
    }
}

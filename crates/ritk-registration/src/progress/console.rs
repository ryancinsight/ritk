use crate::progress::{ProgressCallback, ProgressInfo};

/// Console progress callback that logs to tracing.
#[derive(Debug, Clone)]
pub struct ConsoleProgressCallback {
    /// Log interval (iterations).
    pub log_interval: usize,
    /// Show progress bar.
    pub show_progress_bar: bool,
}

impl Default for ConsoleProgressCallback {
    fn default() -> Self {
        Self {
            log_interval: 50,
            show_progress_bar: true,
        }
    }
}

impl ConsoleProgressCallback {
    /// Create a new console progress callback.
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            show_progress_bar: true,
        }
    }

    /// Disable progress bar.
    pub fn without_progress_bar(mut self) -> Self {
        self.show_progress_bar = false;
        self
    }
}

impl ProgressCallback for ConsoleProgressCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        if info.iteration % self.log_interval == 0 || info.total_iterations == Some(info.iteration)
        {
            let progress = info.progress_percent().unwrap_or(0.0);
            let elapsed = format!("{:.2}s", info.elapsed.as_secs_f64());
            let remaining = info
                .estimated_remaining
                .map(|d| format!("{:.2}s", d.as_secs_f64()))
                .unwrap_or_else(|| "N/A".to_string());

            tracing::info!(
                "Iter {}/{} ({:.1}%) | Loss: {:.6} | LR: {:.2e} | Elapsed: {} | ETA: {}",
                info.iteration,
                info.total_iterations
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "?".to_string()),
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

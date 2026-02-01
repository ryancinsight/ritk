//! Progress tracking and callbacks for registration workflows.
//!
//! This module provides progress tracking, callbacks, and monitoring
//! capabilities for registration operations.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Progress information for registration iterations.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current iteration number.
    pub iteration: usize,
    /// Total number of iterations (if known).
    pub total_iterations: Option<usize>,
    /// Current loss value.
    pub loss: f64,
    /// Time elapsed since start.
    pub elapsed: Duration,
    /// Estimated remaining time.
    pub estimated_remaining: Option<Duration>,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Additional metrics.
    pub metrics: Vec<(String, f64)>,
}

impl ProgressInfo {
    /// Create new progress information.
    pub fn new(
        iteration: usize,
        total_iterations: Option<usize>,
        loss: f64,
        elapsed: Duration,
        learning_rate: f64,
    ) -> Self {
        Self {
            iteration,
            total_iterations,
            loss,
            elapsed,
            estimated_remaining: None,
            learning_rate,
            metrics: Vec::new(),
        }
    }

    /// Calculate progress percentage.
    pub fn progress_percent(&self) -> Option<f64> {
        self.total_iterations.map(|total| {
            (self.iteration as f64 / total as f64) * 100.0
        })
    }

    /// Calculate estimated remaining time.
    pub fn calculate_remaining(&mut self) {
        if let Some(total) = self.total_iterations {
            if self.iteration > 0 {
                let avg_time_per_iter = self.elapsed.as_secs_f64() / self.iteration as f64;
                let remaining_iters = total.saturating_sub(self.iteration);
                self.estimated_remaining = Some(Duration::from_secs_f64(
                    avg_time_per_iter * remaining_iters as f64
                ));
            }
        }
    }

    /// Add a custom metric.
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
}

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
        if info.iteration % self.log_interval == 0 || info.total_iterations == Some(info.iteration) {
            let progress = info.progress_percent().unwrap_or(0.0);
            let elapsed = format!("{:.2}s", info.elapsed.as_secs_f64());
            let remaining = info.estimated_remaining
                .map(|d| format!("{:.2}s", d.as_secs_f64()))
                .unwrap_or_else(|| "N/A".to_string());

            tracing::info!(
                "Iter {}/{} ({:.1}%) | Loss: {:.6} | LR: {:.2e} | Elapsed: {} | ETA: {}",
                info.iteration,
                info.total_iterations.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string()),
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

/// History callback that records all progress information.
#[derive(Debug, Clone)]
pub struct HistoryCallback {
    /// History of progress information.
    history: Arc<Mutex<Vec<ProgressInfo>>>,
}

impl HistoryCallback {
    /// Create a new history callback.
    pub fn new() -> Self {
        Self {
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the recorded history.
    pub fn get_history(&self) -> Vec<ProgressInfo> {
        self.history.lock().unwrap().clone()
    }

    /// Clear the history.
    pub fn clear(&self) {
        self.history.lock().unwrap().clear();
    }
}

impl Default for HistoryCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressCallback for HistoryCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        self.history.lock().unwrap().push(info.clone());
    }
}

/// Early stopping callback.
#[derive(Debug, Clone)]
pub struct EarlyStoppingCallback {
    /// Minimum improvement to continue.
    pub min_improvement: f64,
    /// Number of iterations to wait for improvement.
    pub patience: usize,
    /// Minimum loss threshold.
    pub min_loss: Option<f64>,
    /// Counter for iterations without improvement.
    counter: Arc<Mutex<usize>>,
    /// Best loss seen so far.
    best_loss: Arc<Mutex<f64>>,
    /// Whether to stop.
    should_stop: Arc<Mutex<bool>>,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(min_improvement: f64, patience: usize) -> Self {
        Self {
            min_improvement,
            patience,
            min_loss: None,
            counter: Arc::new(Mutex::new(0)),
            best_loss: Arc::new(Mutex::new(f64::INFINITY)),
            should_stop: Arc::new(Mutex::new(false)),
        }
    }

    /// Set minimum loss threshold.
    pub fn with_min_loss(mut self, min_loss: f64) -> Self {
        self.min_loss = Some(min_loss);
        self
    }

    /// Check if should stop.
    pub fn should_stop(&self) -> bool {
        *self.should_stop.lock().unwrap()
    }

    /// Reset early stopping state.
    pub fn reset(&self) {
        *self.counter.lock().unwrap() = 0;
        *self.best_loss.lock().unwrap() = f64::INFINITY;
        *self.should_stop.lock().unwrap() = false;
    }
}

impl ProgressCallback for EarlyStoppingCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        let mut best_loss = self.best_loss.lock().unwrap();
        let mut counter = self.counter.lock().unwrap();

        // Check minimum loss threshold
        if let Some(min_loss) = self.min_loss {
            if info.loss <= min_loss {
                *self.should_stop.lock().unwrap() = true;
                tracing::info!("Early stopping: loss {} reached minimum threshold {}", info.loss, min_loss);
                return;
            }
        }

        // Check for improvement
        let improvement = *best_loss - info.loss;
        if improvement > self.min_improvement {
            *best_loss = info.loss;
            *counter = 0;
        } else {
            *counter += 1;
        }

        // Check patience
        if *counter >= self.patience {
            *self.should_stop.lock().unwrap() = true;
            tracing::info!(
                "Early stopping: no improvement for {} iterations (best loss: {:.6}, current: {:.6})",
                self.patience,
                *best_loss,
                info.loss
            );
        }
    }
}

/// Progress tracker that manages multiple callbacks.
#[derive(Clone)]
pub struct ProgressTracker {
    /// Registered callbacks.
    callbacks: Vec<Arc<dyn ProgressCallback>>,
    /// Start time.
    start_time: Arc<Mutex<Option<Instant>>>,
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self {
            callbacks: Vec::new(),
            start_time: Arc::new(Mutex::new(None)),
        }
    }
}

impl ProgressTracker {
    /// Create a new progress tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a callback.
    pub fn add_callback(&mut self, callback: Arc<dyn ProgressCallback>) {
        self.callbacks.push(callback);
    }

    /// Start tracking.
    pub fn start(&self) {
        *self.start_time.lock().unwrap() = Some(Instant::now());
        for callback in &self.callbacks {
            callback.on_start();
        }
    }

    /// Update progress.
    pub fn update(&self, iteration: usize, total_iterations: Option<usize>, loss: f64, learning_rate: f64) {
        let start_time = *self.start_time.lock().unwrap();
        let elapsed = start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);

        let mut info = ProgressInfo::new(iteration, total_iterations, loss, elapsed, learning_rate);
        info.calculate_remaining();

        for callback in &self.callbacks {
            callback.on_progress(&info);
        }
    }

    /// Complete tracking.
    pub fn complete(&self, final_loss: f64, learning_rate: f64) {
        let start_time = *self.start_time.lock().unwrap();
        let elapsed = start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);

        let info = ProgressInfo::new(
            start_time.as_ref().map(|_| 0).unwrap_or(0),
            Some(0),
            final_loss,
            elapsed,
            learning_rate,
        );

        for callback in &self.callbacks {
            callback.on_complete(&info);
        }
    }

    /// Report error.
    pub fn error(&self, error: &str) {
        for callback in &self.callbacks {
            callback.on_error(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_info() {
        let info = ProgressInfo::new(10, Some(100), 0.5, Duration::from_secs(10), 0.01);
        assert_eq!(info.iteration, 10);
        assert_eq!(info.loss, 0.5);
        assert_eq!(info.progress_percent(), Some(10.0));
    }

    #[test]
    fn test_progress_info_remaining() {
        let mut info = ProgressInfo::new(10, Some(100), 0.5, Duration::from_secs(10), 0.01);
        info.calculate_remaining();
        assert!(info.estimated_remaining.is_some());
    }

    #[test]
    fn test_early_stopping() {
        let callback = EarlyStoppingCallback::new(0.01, 3);

        // Improving loss
        callback.on_progress(&ProgressInfo::new(1, Some(10), 1.0, Duration::ZERO, 0.01));
        callback.on_progress(&ProgressInfo::new(2, Some(10), 0.9, Duration::ZERO, 0.01));
        callback.on_progress(&ProgressInfo::new(3, Some(10), 0.8, Duration::ZERO, 0.01));
        assert!(!callback.should_stop());

        // No improvement
        callback.on_progress(&ProgressInfo::new(4, Some(10), 0.8, Duration::ZERO, 0.01));
        callback.on_progress(&ProgressInfo::new(5, Some(10), 0.8, Duration::ZERO, 0.01));
        callback.on_progress(&ProgressInfo::new(6, Some(10), 0.8, Duration::ZERO, 0.01));
        assert!(callback.should_stop());
    }

    #[test]
    fn test_history_callback() {
        let callback = HistoryCallback::new();
        callback.on_progress(&ProgressInfo::new(1, Some(10), 0.5, Duration::ZERO, 0.01));
        callback.on_progress(&ProgressInfo::new(2, Some(10), 0.4, Duration::ZERO, 0.01));

        let history = callback.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].iteration, 1);
        assert_eq!(history[1].iteration, 2);
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new();
        tracker.start();
        tracker.update(1, Some(10), 0.5, 0.01);
        tracker.update(2, Some(10), 0.4, 0.01);
        tracker.complete(0.3, 0.01);
    }
}

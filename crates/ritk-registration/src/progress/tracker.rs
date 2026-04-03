use crate::progress::{ProgressCallback, ProgressInfo};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    pub fn update(
        &self,
        iteration: usize,
        total_iterations: Option<usize>,
        loss: f64,
        learning_rate: f64,
    ) {
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
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new();
        tracker.start();
        tracker.update(1, Some(10), 0.5, 0.01);
        tracker.update(2, Some(10), 0.4, 0.01);
        tracker.complete(0.3, 0.01);
    }
}

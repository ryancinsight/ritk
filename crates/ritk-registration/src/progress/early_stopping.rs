use crate::progress::{ProgressCallback, ProgressInfo};
use std::sync::{Arc, Mutex};

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
                tracing::info!(
                    "Early stopping: loss {} reached minimum threshold {}",
                    info.loss,
                    min_loss
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
}

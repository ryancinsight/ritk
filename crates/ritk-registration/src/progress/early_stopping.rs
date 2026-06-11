use crate::progress::{ProgressCallback, ProgressInfo};
use std::sync::{Arc, Mutex};

/// Whether early stopping has been triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum EarlyStopSignal {
    /// Continue iterating.
    #[default]
    Continue,
    /// Early stopping criterion was met — halt iteration.
    Stop,
}

#[derive(Debug)]
struct EarlyStoppingState {
    counter: usize,
    best_loss: f64,
    stop_signal: EarlyStopSignal,
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
    state: Arc<Mutex<EarlyStoppingState>>,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(min_improvement: f64, patience: usize) -> Self {
        Self {
            min_improvement,
            patience,
            min_loss: None,
            state: Arc::new(Mutex::new(EarlyStoppingState {
                counter: 0,
                best_loss: f64::INFINITY,
                stop_signal: EarlyStopSignal::default(),
            })),
        }
    }

    /// Set minimum loss threshold.
    pub fn with_min_loss(mut self, min_loss: f64) -> Self {
        self.min_loss = Some(min_loss);
        self
    }

    /// Check if should stop.
    pub fn should_stop(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stop_signal
            == EarlyStopSignal::Stop
    }

    /// Reset early stopping state.
    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.counter = 0;
        s.best_loss = f64::INFINITY;
        s.stop_signal = EarlyStopSignal::default();
    }
}

impl ProgressCallback for EarlyStoppingCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        // Check minimum loss threshold (self.min_loss is Copy, no lock needed)
        if let Some(min_loss) = self.min_loss {
            if info.loss <= min_loss {
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .stop_signal = EarlyStopSignal::Stop;
                tracing::info!(
                    "Early stopping: loss {} reached minimum threshold {}",
                    info.loss,
                    min_loss
                );
                return;
            }
        }

        // Single lock covers the entire counter/best_loss/should_stop update atomically.
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let improvement = s.best_loss - info.loss;
        if improvement > self.min_improvement {
            s.best_loss = info.loss;
            s.counter = 0;
        } else {
            s.counter += 1;
        }
        if s.counter >= self.patience {
            s.stop_signal = EarlyStopSignal::Stop;
            tracing::info!(
                "Early stopping: no improvement for {} iterations (best loss: {:.6}, current: {:.6})",
                self.patience,
                s.best_loss,
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

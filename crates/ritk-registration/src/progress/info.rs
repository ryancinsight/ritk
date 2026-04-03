use std::time::Duration;

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
        self.total_iterations
            .map(|total| (self.iteration as f64 / total as f64) * 100.0)
    }

    /// Calculate estimated remaining time.
    pub fn calculate_remaining(&mut self) {
        if let Some(total) = self.total_iterations {
            if self.iteration > 0 {
                let avg_time_per_iter = self.elapsed.as_secs_f64() / self.iteration as f64;
                let remaining_iters = total.saturating_sub(self.iteration);
                self.estimated_remaining = Some(Duration::from_secs_f64(
                    avg_time_per_iter * remaining_iters as f64,
                ));
            }
        }
    }

    /// Add a custom metric.
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
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
}

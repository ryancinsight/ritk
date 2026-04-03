use crate::progress::{ProgressCallback, ProgressInfo};
use std::sync::{Arc, Mutex};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
}

//! Algorithmic bounds convergence state representations.

/// Check for mathematical convergence based upon empirical relative loss histories.
///
/// # Theorem: Global Convergence Constraint
/// The `ConvergenceChecker` algorithm assesses whether the optimization trajectory
/// over recent epochs has sufficiently stabilized relative to the lowest historically known state, evaluated explicitly over:
/// $$ \frac{L_{best} - L_{current}}{|L_{best}| + \epsilon} < \eta $$
/// Where $\eta$ denotes the target theoretical `min_improvement` patience differential constraint tolerance threshold.
#[derive(Debug, Clone)]
pub struct ConvergenceChecker {
    /// Minimum relative analytical boundary threshold mapping limit.
    pub min_improvement: f64,
    /// Absolute generation limit tolerance before enforcing bounds evaluation.
    pub patience: usize,
    pub min_loss: Option<f64>,
}

impl Default for ConvergenceChecker {
    fn default() -> Self {
        Self {
            min_improvement: 1e-6,
            patience: 50,
            min_loss: None,
        }
    }
}

impl ConvergenceChecker {
    pub fn new(min_improvement: f64, patience: usize) -> Self {
        Self {
            min_improvement,
            patience,
            min_loss: None,
        }
    }

    pub fn with_min_loss(mut self, min_loss: f64) -> Self {
        self.min_loss = Some(min_loss);
        self
    }

    /// Extrapolate internal states iterating across previous relative error domains mapped securely resolving bounds safely.
    pub fn check_convergence(&self, loss_history: &[f64]) -> bool {
        if loss_history.is_empty() {
            return false;
        }

        if let Some(min_loss) = self.min_loss {
            if loss_history.last().unwrap() < &min_loss {
                return true;
            }
        }

        if loss_history.len() < self.patience + 1 {
            return false;
        }

        let recent_start = loss_history.len() - self.patience - 1;
        let recent_losses = &loss_history[recent_start..];

        let best_loss = recent_losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let current_loss = *loss_history.last().unwrap();

        // Exact translation implementing analytic boundaries limits reliably.
        let relative_improvement = (best_loss - current_loss) / (best_loss.abs() + 1e-10);

        relative_improvement < self.min_improvement
    }
}

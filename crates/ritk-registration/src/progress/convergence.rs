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
            if loss_history
                .last()
                .expect("loss history must not be empty when checking convergence")
                < &min_loss
            {
                return true;
            }
        }

        if loss_history.len() < self.patience + 1 {
            return false;
        }

        let recent_start = loss_history.len() - self.patience - 1;
        let recent_losses = &loss_history[recent_start..];
        let previous_losses = &recent_losses[..self.patience];

        let best_loss = previous_losses
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let current_loss = *loss_history
            .last()
            .expect("loss history must not be empty when checking convergence");

        // Exact translation implementing analytic boundaries limits reliably.
        let relative_improvement = (best_loss - current_loss) / (best_loss.abs() + 1e-10);

        relative_improvement < self.min_improvement
    }
}

#[cfg(test)]
mod tests {
    use super::ConvergenceChecker;

    #[test]
    fn improving_best_loss_does_not_converge() {
        let checker = ConvergenceChecker::new(1.0e-3, 3);

        assert!(
            !checker.check_convergence(&[1.0, 0.9, 0.8, 0.7]),
            "current best loss must count as improvement, not convergence"
        );
    }

    #[test]
    fn small_window_improvement_converges() {
        let checker = ConvergenceChecker::new(1.0e-3, 3);

        assert!(
            checker.check_convergence(&[1.0, 0.9000, 0.8999, 0.8998]),
            "relative improvement across the patience window is below threshold"
        );
    }

    #[test]
    fn min_loss_converges_without_full_window() {
        let checker = ConvergenceChecker::new(1.0e-3, 10).with_min_loss(0.1);

        assert!(checker.check_convergence(&[0.09]));
    }
}

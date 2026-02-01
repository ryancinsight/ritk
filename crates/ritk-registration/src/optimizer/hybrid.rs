// Hybrid optimization implementation
// Combines multiple optimizers for improved performance and convergence

use crate::error::RegistrationError;
use crate::metric::Metric;
use crate::optimizer::{Optimizer, OptimizerState};
use crate::transform::Transform;
use std::sync::Arc;

/// Strategy for combining optimizers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// Sequential: run optimizers one after another
    Sequential,
    /// Adaptive: switch based on convergence criteria
    Adaptive,
    /// Ensemble: average gradients from multiple optimizers
    Ensemble,
}

/// Configuration for hybrid optimizer
#[derive(Debug, Clone)]
pub struct HybridOptimizerConfig {
    /// Strategy for combining optimizers
    pub strategy: HybridStrategy,
    /// Maximum iterations for each optimizer phase
    pub max_iterations_per_phase: Vec<usize>,
    /// Convergence threshold for switching phases
    pub switch_threshold: f64,
    /// Weight for ensemble averaging (only used with Ensemble strategy)
    pub ensemble_weights: Vec<f64>,
    /// Whether to use momentum across phases
    pub cross_phase_momentum: bool,
}

impl Default for HybridOptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: HybridStrategy::Sequential,
            max_iterations_per_phase: vec![100, 200],
            switch_threshold: 1e-3,
            ensemble_weights: vec![0.5, 0.5],
            cross_phase_momentum: false,
        }
    }
}

impl HybridOptimizerConfig {
    /// Create a new hybrid optimizer configuration
    pub fn new(strategy: HybridStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Set maximum iterations for each phase
    pub fn with_iterations(mut self, iterations: Vec<usize>) -> Self {
        self.max_iterations_per_phase = iterations;
        self
    }

    /// Set the threshold for switching between phases
    pub fn with_switch_threshold(mut self, threshold: f64) -> Self {
        self.switch_threshold = threshold;
        self
    }

    /// Set ensemble weights
    pub fn with_ensemble_weights(mut self, weights: Vec<f64>) -> Self {
        self.ensemble_weights = weights;
        self
    }

    /// Enable cross-phase momentum
    pub fn with_cross_phase_momentum(mut self, enabled: bool) -> Self {
        self.cross_phase_momentum = enabled;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), RegistrationError> {
        if self.max_iterations_per_phase.is_empty() {
            return Err(RegistrationError::InvalidConfiguration(
                "Hybrid optimizer requires at least one phase".to_string(),
            ));
        }
        if self.strategy == HybridStrategy::Ensemble {
            if self.ensemble_weights.is_empty() {
                return Err(RegistrationError::InvalidConfiguration(
                    "Ensemble strategy requires weights".to_string(),
                ));
            }
            let sum: f64 = self.ensemble_weights.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(RegistrationError::InvalidConfiguration(
                    format!("Ensemble weights must sum to 1.0, got {}", sum),
                ));
            }
        }
        Ok(())
    }
}

/// Hybrid optimizer that combines multiple optimizers
pub struct HybridOptimizer<T, M>
where
    T: Transform,
    M: Metric<T>,
{
    optimizers: Vec<Box<dyn Optimizer<T, M>>>,
    config: HybridOptimizerConfig,
    state: Option<HybridOptimizerState>,
}

/// State for hybrid optimizer
#[derive(Debug, Clone)]
pub struct HybridOptimizerState {
    /// Current phase index
    pub current_phase: usize,
    /// Iterations in current phase
    pub phase_iterations: usize,
    /// Best metric value seen so far
    pub best_metric: f64,
    /// Previous metric value (for convergence detection)
    pub previous_metric: f64,
    /// Momentum from previous phase (if enabled)
    pub momentum: Option<Vec<f64>>,
}

impl<T, M> HybridOptimizer<T, M>
where
    T: Transform,
    M: Metric<T>,
{
    /// Create a new hybrid optimizer
    pub fn new(
        optimizers: Vec<Box<dyn Optimizer<T, M>>>,
        config: HybridOptimizerConfig,
    ) -> Result<Self, RegistrationError> {
        config.validate()?;
        
        if optimizers.is_empty() {
            return Err(RegistrationError::InvalidConfiguration(
                "Hybrid optimizer requires at least one optimizer".to_string(),
            ));
        }

        Ok(Self {
            optimizers,
            config,
            state: None,
        })
    }

    /// Create a hybrid optimizer with default configuration
    pub fn with_defaults(optimizers: Vec<Box<dyn Optimizer<T, M>>>) -> Result<Self, RegistrationError> {
        Self::new(optimizers, HybridOptimizerConfig::default())
    }

    /// Get the current state
    pub fn state(&self) -> Option<&HybridOptimizerState> {
        self.state.as_ref()
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.state = None;
        for optimizer in &mut self.optimizers {
            optimizer.reset();
        }
    }
}

impl<T, M> Optimizer<T, M> for HybridOptimizer<T, M>
where
    T: Transform,
    M: Metric<T>,
{
    fn step(
        &mut self,
        transform: &mut T,
        metric: &M,
        fixed: &ritk_core::image::Image,
        moving: &ritk_core::image::Image,
    ) -> Result<OptimizerState, RegistrationError> {
        match self.config.strategy {
            HybridStrategy::Sequential => self.step_sequential(transform, metric, fixed, moving),
            HybridStrategy::Adaptive => self.step_adaptive(transform, metric, fixed, moving),
            HybridStrategy::Ensemble => self.step_ensemble(transform, metric, fixed, moving),
        }
    }

    fn reset(&mut self) {
        self.state = None;
        for optimizer in &mut self.optimizers {
            optimizer.reset();
        }
    }

    fn state(&self) -> Option<&OptimizerState> {
        self.state.as_ref().map(|s| &s.best_metric).map(|_| {
            // Create a basic OptimizerState from our hybrid state
            &OptimizerState {
                iterations: self.state.as_ref().map(|s| s.phase_iterations).unwrap_or(0),
                gradient_norm: 0.0,
                learning_rate: 0.0,
                metric_value: self.state.as_ref().map(|s| s.best_metric).unwrap_or(0.0),
                converged: false,
            }
        })
    }
}

impl<T, M> HybridOptimizer<T, M>
where
    T: Transform,
    M: Metric<T>,
{
    /// Sequential optimization strategy
    fn step_sequential(
        &mut self,
        transform: &mut T,
        metric: &M,
        fixed: &ritk_core::image::Image,
        moving: &ritk_core::image::Image,
    ) -> Result<OptimizerState, RegistrationError> {
        // Initialize state if needed
        if self.state.is_none() {
            let initial_metric = metric.compute(transform, fixed, moving)?;
            self.state = Some(HybridOptimizerState {
                current_phase: 0,
                phase_iterations: 0,
                best_metric: initial_metric,
                previous_metric: initial_metric,
                momentum: None,
            });
        }

        let state = self.state.as_mut().unwrap();

        // Check if we should move to next phase
        if state.phase_iterations >= self.config.max_iterations_per_phase[state.current_phase] {
            if state.current_phase + 1 < self.optimizers.len() {
                state.current_phase += 1;
                state.phase_iterations = 0;
                
                // Reset optimizer for new phase
                self.optimizers[state.current_phase].reset();
            }
        }

        // Execute step with current optimizer
        let optimizer_state = self.optimizers[state.current_phase].step(
            transform,
            metric,
            fixed,
            moving,
        )?;

        // Update state
        state.phase_iterations += 1;
        state.previous_metric = state.best_metric;
        state.best_metric = optimizer_state.metric_value;

        Ok(optimizer_state)
    }

    /// Adaptive optimization strategy
    fn step_adaptive(
        &mut self,
        transform: &mut T,
        metric: &M,
        fixed: &ritk_core::image::Image,
        moving: &ritk_core::image::Image,
    ) -> Result<OptimizerState, RegistrationError> {
        // Initialize state if needed
        if self.state.is_none() {
            let initial_metric = metric.compute(transform, fixed, moving)?;
            self.state = Some(HybridOptimizerState {
                current_phase: 0,
                phase_iterations: 0,
                best_metric: initial_metric,
                previous_metric: initial_metric,
                momentum: None,
            });
        }

        let state = self.state.as_mut().unwrap();

        // Check convergence for current phase
        let improvement = (state.previous_metric - state.best_metric).abs();
        let should_switch = improvement < self.config.switch_threshold 
            && state.phase_iterations > 10;

        if should_switch && state.current_phase + 1 < self.optimizers.len() {
            state.current_phase += 1;
            state.phase_iterations = 0;
            self.optimizers[state.current_phase].reset();
        }

        // Execute step with current optimizer
        let optimizer_state = self.optimizers[state.current_phase].step(
            transform,
            metric,
            fixed,
            moving,
        )?;

        // Update state
        state.phase_iterations += 1;
        state.previous_metric = state.best_metric;
        state.best_metric = optimizer_state.metric_value;

        Ok(optimizer_state)
    }

    /// Ensemble optimization strategy
    fn step_ensemble(
        &mut self,
        transform: &mut T,
        metric: &M,
        fixed: &ritk_core::image::Image,
        moving: &ritk_core::image::Image,
    ) -> Result<OptimizerState, RegistrationError> {
        // Initialize state if needed
        if self.state.is_none() {
            let initial_metric = metric.compute(transform, fixed, moving)?;
            self.state = Some(HybridOptimizerState {
                current_phase: 0,
                phase_iterations: 0,
                best_metric: initial_metric,
                previous_metric: initial_metric,
                momentum: None,
            });
        }

        let state = self.state.as_mut().unwrap();

        // Get gradients from all optimizers
        let mut weighted_gradients = Vec::new();
        let mut total_metric = 0.0;

        for (i, optimizer) in self.optimizers.iter_mut().enumerate() {
            // Clone transform for each optimizer
            let mut temp_transform = transform.clone();
            
            // Get optimizer state
            let optimizer_state = optimizer.step(
                &mut temp_transform,
                metric,
                fixed,
                moving,
            )?;

            // Accumulate weighted metric
            total_metric += self.config.ensemble_weights[i] * optimizer_state.metric_value;
        }

        // Apply weighted average of updates
        // Note: This is a simplified implementation
        // In practice, you'd need to properly combine the parameter updates
        
        state.phase_iterations += 1;
        state.previous_metric = state.best_metric;
        state.best_metric = total_metric;

        Ok(OptimizerState {
            iterations: state.phase_iterations,
            gradient_norm: 0.0,
            learning_rate: 0.0,
            metric_value: total_metric,
            converged: false,
        })
    }
}

/// Preset hybrid optimizer configurations
pub mod presets {
    use super::*;

    /// Create a fast hybrid optimizer (SGD + Momentum)
    pub fn fast<T, M>(
        sgd: Box<dyn Optimizer<T, M>>,
        momentum: Box<dyn Optimizer<T, M>>,
    ) -> Result<HybridOptimizer<T, M>, RegistrationError> {
        let config = HybridOptimizerConfig {
            strategy: HybridStrategy::Sequential,
            max_iterations_per_phase: vec![50, 100],
            switch_threshold: 1e-2,
            ensemble_weights: vec![0.3, 0.7],
            cross_phase_momentum: true,
        };

        HybridOptimizer::new(vec![sgd, momentum], config)
    }

    /// Create a robust hybrid optimizer (Adam + L-BFGS)
    pub fn robust<T, M>(
        adam: Box<dyn Optimizer<T, M>>,
        lbfgs: Box<dyn Optimizer<T, M>>,
    ) -> Result<HybridOptimizer<T, M>, RegistrationError> {
        let config = HybridOptimizerConfig {
            strategy: HybridStrategy::Adaptive,
            max_iterations_per_phase: vec![200, 300],
            switch_threshold: 1e-4,
            ensemble_weights: vec![0.5, 0.5],
            cross_phase_momentum: false,
        };

        HybridOptimizer::new(vec![adam, lbfgs], config)
    }

    /// Create a balanced hybrid optimizer (SGD + Adam + L-BFGS)
    pub fn balanced<T, M>(
        sgd: Box<dyn Optimizer<T, M>>,
        adam: Box<dyn Optimizer<T, M>>,
        lbfgs: Box<dyn Optimizer<T, M>>,
    ) -> Result<HybridOptimizer<T, M>, RegistrationError> {
        let config = HybridOptimizerConfig {
            strategy: HybridStrategy::Sequential,
            max_iterations_per_phase: vec![100, 200, 300],
            switch_threshold: 1e-3,
            ensemble_weights: vec![0.2, 0.3, 0.5],
            cross_phase_momentum: true,
        };

        HybridOptimizer::new(vec![sgd, adam, lbfgs], config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::MSE;
    use crate::optimizer::GradientDescent;
    use crate::transform::RigidTransform;
    use ritk_core::image::Image;

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridOptimizerConfig::default();
        assert_eq!(config.strategy, HybridStrategy::Sequential);
        assert_eq!(config.max_iterations_per_phase, vec![100, 200]);
    }

    #[test]
    fn test_hybrid_config_builder() {
        let config = HybridOptimizerConfig::new(HybridStrategy::Adaptive)
            .with_iterations(vec![50, 100, 150])
            .with_switch_threshold(1e-4)
            .with_cross_phase_momentum(true);

        assert_eq!(config.strategy, HybridStrategy::Adaptive);
        assert_eq!(config.max_iterations_per_phase.len(), 3);
        assert!(config.cross_phase_momentum);
    }

    #[test]
    fn test_hybrid_config_validation() {
        // Valid config
        let config = HybridOptimizerConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: empty phases
        let config = HybridOptimizerConfig {
            max_iterations_per_phase: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: ensemble weights don't sum to 1.0
        let config = HybridOptimizerConfig {
            strategy: HybridStrategy::Ensemble,
            ensemble_weights: vec![0.3, 0.5],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}

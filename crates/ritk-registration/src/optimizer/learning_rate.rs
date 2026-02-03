//! Learning rate scheduling strategies for registration optimization.
//!
//! This module provides various learning rate scheduling strategies to improve
//! convergence and optimization stability during registration.

/// Learning rate scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(f64),
    /// Step decay: multiply by gamma every step_size iterations
    StepDecay {
        initial_lr: f64,
        gamma: f64,
        step_size: usize,
    },
    /// Exponential decay: multiply by gamma every iteration
    ExponentialDecay {
        initial_lr: f64,
        gamma: f64,
    },
    /// Inverse time decay: lr = initial_lr / (1 + decay_rate * iteration)
    InverseTimeDecay {
        initial_lr: f64,
        decay_rate: f64,
    },
    /// Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * iteration / max_iterations))
    CosineAnnealing {
        initial_lr: f64,
        min_lr: f64,
        max_iterations: usize,
    },
    /// Polynomial decay: lr = initial_lr * (1 - iteration / max_iterations)^power
    PolynomialDecay {
        initial_lr: f64,
        end_lr: f64,
        max_iterations: usize,
        power: f64,
    },
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant(0.01)
    }
}

impl LearningRateSchedule {
    /// Create a constant learning rate schedule.
    pub fn constant(lr: f64) -> Self {
        Self::Constant(lr)
    }

    /// Create a step decay schedule.
    pub fn step_decay(initial_lr: f64, gamma: f64, step_size: usize) -> Self {
        Self::StepDecay {
            initial_lr,
            gamma,
            step_size,
        }
    }

    /// Create an exponential decay schedule.
    pub fn exponential_decay(initial_lr: f64, gamma: f64) -> Self {
        Self::ExponentialDecay {
            initial_lr,
            gamma,
        }
    }

    /// Create an inverse time decay schedule.
    pub fn inverse_time_decay(initial_lr: f64, decay_rate: f64) -> Self {
        Self::InverseTimeDecay {
            initial_lr,
            decay_rate,
        }
    }

    /// Create a cosine annealing schedule.
    pub fn cosine_annealing(initial_lr: f64, min_lr: f64, max_iterations: usize) -> Self {
        Self::CosineAnnealing {
            initial_lr,
            min_lr,
            max_iterations,
        }
    }

    /// Create a polynomial decay schedule.
    pub fn polynomial_decay(initial_lr: f64, end_lr: f64, max_iterations: usize, power: f64) -> Self {
        Self::PolynomialDecay {
            initial_lr,
            end_lr,
            max_iterations,
            power,
        }
    }

    /// Get the learning rate for a given iteration.
    pub fn get_learning_rate(&self, iteration: usize) -> f64 {
        match self {
            Self::Constant(lr) => *lr,
            Self::StepDecay {
                initial_lr,
                gamma,
                step_size,
            } => {
                let decay_steps = iteration / step_size;
                initial_lr * gamma.powi(decay_steps as i32)
            }
            Self::ExponentialDecay {
                initial_lr,
                gamma,
            } => initial_lr * gamma.powi(iteration as i32),
            Self::InverseTimeDecay {
                initial_lr,
                decay_rate,
            } => initial_lr / (1.0 + decay_rate * iteration as f64),
            Self::CosineAnnealing {
                initial_lr,
                min_lr,
                max_iterations,
            } => {
                let progress = (iteration as f64) / (*max_iterations as f64);
                let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                min_lr + (initial_lr - min_lr) * cosine
            }
            Self::PolynomialDecay {
                initial_lr,
                end_lr,
                max_iterations,
                power,
            } => {
                let progress = (iteration as f64) / (*max_iterations as f64);
                let decay = (1.0 - progress).powf(*power);
                end_lr + (initial_lr - end_lr) * decay
            }
        }
    }

    /// Get the initial learning rate.
    pub fn initial_lr(&self) -> f64 {
        match self {
            Self::Constant(lr) => *lr,
            Self::StepDecay { initial_lr, .. } => *initial_lr,
            Self::ExponentialDecay { initial_lr, .. } => *initial_lr,
            Self::InverseTimeDecay { initial_lr, .. } => *initial_lr,
            Self::CosineAnnealing { initial_lr, .. } => *initial_lr,
            Self::PolynomialDecay { initial_lr, .. } => *initial_lr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_schedule() {
        let schedule = LearningRateSchedule::constant(0.01);
        assert_eq!(schedule.get_learning_rate(0), 0.01);
        assert_eq!(schedule.get_learning_rate(100), 0.01);
    }

    #[test]
    fn test_step_decay() {
        let schedule = LearningRateSchedule::step_decay(0.1, 0.5, 10);
        assert_eq!(schedule.get_learning_rate(0), 0.1);
        assert_eq!(schedule.get_learning_rate(5), 0.1);
        assert_eq!(schedule.get_learning_rate(10), 0.05);
        assert_eq!(schedule.get_learning_rate(20), 0.025);
    }

    #[test]
    fn test_exponential_decay() {
        let schedule = LearningRateSchedule::exponential_decay(0.1, 0.99);
        assert_eq!(schedule.get_learning_rate(0), 0.1);
        assert!((schedule.get_learning_rate(100) - 0.0366).abs() < 0.001);
    }

    #[test]
    fn test_inverse_time_decay() {
        let schedule = LearningRateSchedule::inverse_time_decay(0.1, 0.01);
        assert_eq!(schedule.get_learning_rate(0), 0.1);
        // Formula: lr = initial_lr / (1.0 + decay_rate * iteration)
        // At iteration 100: 0.1 / (1.0 + 0.01 * 100) = 0.1 / 2.0 = 0.05
        assert!((schedule.get_learning_rate(100) - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_cosine_annealing() {
        let schedule = LearningRateSchedule::cosine_annealing(0.1, 0.001, 100);
        assert_eq!(schedule.get_learning_rate(0), 0.1);
        assert!((schedule.get_learning_rate(50) - 0.0505).abs() < 0.001);
        assert!((schedule.get_learning_rate(100) - 0.001).abs() < 0.001);
    }

    #[test]
    fn test_polynomial_decay() {
        let schedule = LearningRateSchedule::polynomial_decay(0.1, 0.001, 100, 1.0);
        assert_eq!(schedule.get_learning_rate(0), 0.1);
        assert!((schedule.get_learning_rate(50) - 0.0505).abs() < 0.001);
        assert!((schedule.get_learning_rate(100) - 0.001).abs() < 0.001);
    }
}

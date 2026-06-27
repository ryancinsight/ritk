use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::registration::{Registration, RegistrationConfig};
use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use ritk_core::image::Image;
use ritk_core::transform::{Resampleable, Transform};
use ritk_filter::pyramid::MultiResolutionPyramid;
use std::marker::PhantomData;

/// Configuration for multi-resolution registration.
///
/// P1-01: `shrink_factors` and `smoothing_sigmas` use stack-allocated `[T; D]`
/// arrays rather than `Vec<Vec<T>>`, eliminating one heap allocation per level
/// and one level of indirection on the hot path. The `[T; D]` representation is
/// also SIMD-friendly and enables the compiler to keep the per-dimension
/// factors in registers across the pyramid build.
pub struct RegistrationSchedule<const D: usize> {
    pub shrink_factors: Vec<[usize; D]>,
    pub smoothing_sigmas: Vec<[f64; D]>,
    pub iterations: Vec<usize>,
    pub learning_rates: Vec<f64>,
}

impl<const D: usize> RegistrationSchedule<D> {
    /// Create a default schedule with power-of-2 shrinking.
    pub fn default(levels: usize) -> Self {
        let mut shrink_factors = Vec::with_capacity(levels);
        let mut smoothing_sigmas = Vec::with_capacity(levels);
        let mut iterations = Vec::with_capacity(levels);
        let mut learning_rates = Vec::with_capacity(levels);

        for i in 0..levels {
            let exponent = (levels - 1 - i) as u32;
            let factor = 2usize.pow(exponent);
            let sigma = if factor > 1 {
                0.5 * (factor as f64)
            } else {
                0.0
            };

            // P1-01: stack array per level — no inner Vec allocation.
            shrink_factors.push([factor; D]);
            smoothing_sigmas.push([sigma; D]);
            iterations.push(100); // Default iterations
            learning_rates.push(1e-2); // Default LR
        }

        Self {
            shrink_factors,
            smoothing_sigmas,
            iterations,
            learning_rates,
        }
    }

    /// Build a schedule from parallel `Vec<[T; D]>` inputs. Use this when
    /// constructing a schedule from external configuration (e.g. user-supplied
    /// per-level factors). The arrays are stored directly — no inner Vec.
    pub fn from_per_level(
        shrink_factors: Vec<[usize; D]>,
        smoothing_sigmas: Vec<[f64; D]>,
    ) -> Self {
        assert_eq!(shrink_factors.len(), smoothing_sigmas.len());
        let n = shrink_factors.len();
        Self {
            shrink_factors,
            smoothing_sigmas,
            iterations: vec![100; n],
            learning_rates: vec![1e-2; n],
        }
    }

    pub fn with_iterations(mut self, iterations: Vec<usize>) -> Self {
        assert_eq!(iterations.len(), self.shrink_factors.len());
        self.iterations = iterations;
        self
    }

    pub fn with_learning_rates(mut self, learning_rates: Vec<f64>) -> Self {
        assert_eq!(learning_rates.len(), self.shrink_factors.len());
        self.learning_rates = learning_rates;
        self
    }
}

/// Multi-resolution registration framework.
///
/// Orchestrates the registration process across multiple resolution levels
/// (coarse-to-fine) to improve robustness and convergence range.
pub struct MultiResolutionRegistration<B, M, T, const D: usize> {
    metric: M,
    registration_config: RegistrationConfig,
    _phantom: PhantomData<(B, T)>,
}

impl<B, M, T, const D: usize> MultiResolutionRegistration<B, M, T, D>
where
    B: AutodiffBackend,
    M: Metric<B, D> + Clone,
    T: Transform<B, D> + AutodiffModule<B> + Resampleable<B, D>,
{
    /// Create a new multi-resolution registration framework.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            registration_config: RegistrationConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the per-level registration loop configuration.
    ///
    /// The same config is cloned into each resolution level so validation,
    /// progress, and convergence policies remain consistent across the
    /// coarse-to-fine schedule.
    #[must_use]
    pub fn with_registration_config(mut self, config: RegistrationConfig) -> Self {
        self.registration_config = config;
        self
    }

    /// Execute the multi-resolution registration.
    ///
    /// # Arguments
    /// * `fixed` - The fixed image
    /// * `moving` - The moving image
    /// * `transform` - The initial transform
    /// * `optimizer_factory` - A closure that creates an optimizer with a given learning rate
    /// * `schedule` - The registration schedule (levels, factors, iterations)
    pub fn execute<F, O>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        optimizer_factory: F,
        schedule: RegistrationSchedule<D>,
    ) -> T
    where
        F: Fn(f64) -> O,
        O: Optimizer<T, B>,
    {
        // 1. Create Pyramids
        // P1-01: schedule already stores `Vec<[usize; D]>` and `Vec<[f64; D]>`,
        // so the pyramid API consumes the arrays directly — no `Vec<Vec<T>>`
        // re-materialisation, no per-level inner allocation.
        let fixed_pyramid = MultiResolutionPyramid::new(
            fixed,
            &schedule.shrink_factors,
            &schedule.smoothing_sigmas,
        );
        let moving_pyramid = MultiResolutionPyramid::new(
            moving,
            &schedule.shrink_factors,
            &schedule.smoothing_sigmas,
        );

        let levels = schedule.shrink_factors.len();

        for i in 0..levels {
            let fixed_level = fixed_pyramid.get_level(i);
            let moving_level = moving_pyramid.get_level(i);

            // Resample transform to current level resolution
            transform = transform.resample(
                fixed_level.shape(),
                *fixed_level.origin(),
                *fixed_level.spacing(),
                *fixed_level.direction(),
            );

            let lr = schedule.learning_rates[i];
            let iters = schedule.iterations[i];

            // Create fresh optimizer for this level
            let optimizer = optimizer_factory(lr);
            let mut registration = Registration::with_config(
                optimizer,
                self.metric.clone(),
                self.registration_config.clone(),
            );

            tracing::info!(
                "Starting level {}/{} with lr={}, iters={}",
                i + 1,
                levels,
                lr,
                iters
            );
            tracing::info!("  Fixed size: {:?}", fixed_level.data().shape());
            tracing::info!("  Moving size: {:?}", moving_level.data().shape());

            transform = registration
                .execute(fixed_level, moving_level, transform, iters, lr)
                .expect("Registration failed at multiresolution level");
        }

        transform
    }
}

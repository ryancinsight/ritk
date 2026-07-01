//! Gradient-descent registration driver on the Coeus-native seams.
//!
//! Ties the verified primitives into one reusable entry point: given a moving
//! image, a fixed sampling grid, a [`CoeusMetric`], and a [`CoeusTransform`]
//! built from trainable parameter `Var`s, [`gradient_descent`] runs the
//! forward → backward → step loop ([`super::metric::evaluate`] →
//! `Var::backward` → [`super::optim::sgd_step_var`]) and returns the optimized
//! parameters. This is the SSOT for "run a Coeus registration"; the per-sprint
//! primitives (loss, sampling, transform, metric, optimizer step) were built
//! and verified individually — this composes them into a usable whole.
//!
//! The transform is rebuilt from the current parameters each iteration via a
//! caller-supplied closure, so the driver is generic over any transform without
//! needing a parameter-reflection trait: the caller owns the mapping from its
//! parameter leaves to its transform type.

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_ops::BackendOps;

use super::metric::evaluate;
use super::optim::sgd_step_var;
use super::traits::{CoeusMetric, CoeusTransform};

/// Configuration for [`gradient_descent`].
#[derive(Debug, Clone, Copy)]
pub struct GradientDescentConfig<T> {
    /// Maximum number of gradient-descent iterations.
    pub iterations: usize,
    /// Learning rate applied per parameter step.
    pub learning_rate: T,
}

/// Outcome of a [`gradient_descent`] run.
pub struct RegistrationOutcome<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
{
    /// Optimized parameter leaves (fresh `requires_grad` `Var`s), in the same
    /// order the caller supplied them.
    pub params: Vec<Var<T, B>>,
    /// Loss at the final parameter values.
    pub final_loss: T,
    /// Loss at the initial parameter values (before any step).
    pub initial_loss: T,
}

/// Run gradient descent to align `moving` to `fixed` under a differentiable
/// transform, minimizing `metric`.
///
/// - `params`: the trainable parameter leaves (each `requires_grad`); order is
///   preserved in the outcome.
/// - `make_transform`: builds the [`CoeusTransform`] from the current parameter
///   slice each iteration (e.g. `|p| Translation { t: p[0].clone() }`).
///
/// Each iteration rebuilds the transform from the current parameters, evaluates
/// the metric, backpropagates, and steps every parameter by
/// `−learning_rate · grad` (a fresh leaf, the tape-based-autograd idiom). The
/// returned `params` are the stepped leaves after the final iteration.
///
/// # Panics
///
/// Panics if `params` is empty, or on the composed primitives' shape invariants
/// — caller invariants.
pub fn gradient_descent<T, B, M, Tf, F>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid: &Var<T, B>,
    metric: &M,
    params: Vec<Var<T, B>>,
    make_transform: F,
    config: GradientDescentConfig<T>,
) -> RegistrationOutcome<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    M: CoeusMetric<T, B>,
    Tf: CoeusTransform<T, B>,
    F: Fn(&[Var<T, B>]) -> Tf,
{
    assert!(!params.is_empty(), "gradient_descent: params must be non-empty");

    let mut params = params;
    let mut initial_loss: Option<T> = None;

    for _ in 0..config.iterations {
        let transform = make_transform(&params);
        let loss = evaluate(moving_flat, dims, fixed, grid, metric, &transform);
        if initial_loss.is_none() {
            initial_loss = Some(loss.tensor.as_slice()[0]);
        }
        loss.backward();
        params = params.iter().map(|p| sgd_step_var(p, config.learning_rate)).collect();
    }

    // Re-evaluate at the returned (post-final-step) parameters so `final_loss`
    // corresponds to `params`, not to the pre-final-step values.
    let final_transform = make_transform(&params);
    let final_loss = evaluate(moving_flat, dims, fixed, grid, metric, &final_transform)
        .tensor
        .as_slice()[0];

    RegistrationOutcome {
        params,
        final_loss,
        initial_loss: initial_loss.unwrap_or(final_loss),
    }
}

#[cfg(test)]
#[path = "tests_driver.rs"]
mod tests;

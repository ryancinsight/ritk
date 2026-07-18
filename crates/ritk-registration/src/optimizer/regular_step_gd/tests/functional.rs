//! Functional step tests for RegularStepGradientDescent.
//!
//! Tests cover: quadratic minimization, gradient convergence,
//! step convergence via overshoot, maximum iterations, and step revert.

use super::super::{ConvergenceReason, RegularStepGdConfig, RegularStepGradientDescent};
use super::{Quadratic, TestBackend};
use crate::optimizer::Optimizer;
use ritk_image::burn::optim::GradientsParams;

// â”€â”€ Minimize f(x) = xáµ€x from xâ‚€ = [5, -3] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// RSGD must reduce loss by â‰¥ 50 % and converge.

#[test]
fn rsgd_minimizes_quadratic_function() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[5.0, -3.0], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 0.5,
        relaxation_factor: 0.5,
        minimum_step_length: 1e-10,
        maximum_step_length: 10.0,
        gradient_tolerance: 1e-8,
        maximum_iterations: 500,
        ..Default::default()
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    let initial_loss = module.loss_value();
    assert!(
        initial_loss > 1.0,
        "initial loss must be > 0; got {initial_loss}"
    );

    for _ in 0..1000 {
        if optimizer.converged() {
            break;
        }
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
    }

    let final_loss = module.loss_value();
    assert!(
        final_loss < initial_loss * 0.5,
        "RSGD must reduce loss by â‰¥ 50 %; initial={initial_loss:.6e}, \
         final={final_loss:.6e}, steps={}, converged={}, reason={:?}",
        optimizer.steps(),
        optimizer.converged(),
        optimizer.convergence_reason()
    );
    assert!(
        optimizer.converged(),
        "RSGD should have converged; steps={}, Î”={:.6e}",
        optimizer.steps(),
        optimizer.current_step_length()
    );
}

// â”€â”€ Gradient convergence near minimum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Starting at x â‰ˆ 1e-8, gradient norm â‰ˆ 2e-8 < gradient_tolerance = 1e-5.

#[test]
fn rsgd_detects_gradient_convergence() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[1e-8, -1e-8], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 0.1,
        relaxation_factor: 0.5,
        minimum_step_length: 1e-20,
        maximum_step_length: 10.0,
        gradient_tolerance: 1e-5,
        maximum_iterations: 100,
        ..Default::default()
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    for _ in 0..10 {
        if optimizer.converged() {
            break;
        }
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
    }

    assert!(
        optimizer.converged(),
        "RSGD should converge near the minimum"
    );
    assert_eq!(
        optimizer.convergence_reason(),
        Some(ConvergenceReason::GradientConvergence),
        "should report gradient convergence; got {:?}",
        optimizer.convergence_reason()
    );
}

// â”€â”€ Step convergence from overshooting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Very large initial step forces repeated shrinkage until Î” < minimum_step_length.

#[test]
fn rsgd_detects_step_convergence() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[1.0, -1.0], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 100.0,
        relaxation_factor: 0.5,
        minimum_step_length: 0.1,
        maximum_step_length: 1000.0,
        gradient_tolerance: 1e-20,
        maximum_iterations: 10000,
        ..Default::default()
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    for _ in 0..1000 {
        if optimizer.converged() {
            break;
        }
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
    }

    assert!(
        optimizer.converged(),
        "RSGD should converge; steps={}, Î”={:.6e}",
        optimizer.steps(),
        optimizer.current_step_length()
    );
    assert_eq!(
        optimizer.convergence_reason(),
        Some(ConvergenceReason::StepConvergence),
        "should report step convergence; got {:?}",
        optimizer.convergence_reason()
    );
}

// â”€â”€ Maximum iterations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn rsgd_detects_maximum_iterations() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[10.0, -10.0], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 0.01,
        relaxation_factor: 0.9,
        minimum_step_length: 1e-30,
        maximum_step_length: 10.0,
        gradient_tolerance: 1e-30,
        maximum_iterations: 5,
        ..Default::default()
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    for _ in 0..100 {
        if optimizer.converged() {
            break;
        }
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
    }

    assert!(optimizer.converged(), "RSGD should converge");
    assert_eq!(
        optimizer.convergence_reason(),
        Some(ConvergenceReason::MaximumIterations),
        "should report maximum iterations; got {:?}",
        optimizer.convergence_reason()
    );
    assert_eq!(optimizer.steps(), 5);
}

// â”€â”€ Robbins-Monro decay reduces step length over accepted iterations â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// With `learning_rate_decay = 0.1` and `relaxation_factor = 1.0` (no
// rejection-shrink), the step length after k accepted steps must be
// Î”â‚– = Î”â‚€ / (1 + Î»_decay Â· k).  After 20 accepted steps on the simple
// quadratic f(x) = xÂ², Î” must have dropped well below Î”â‚€.

#[test]
fn rsgd_robbins_monro_decay_reduces_step_length() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[50.0, -30.0], &device);

    // Starting from [50, -30] keeps the optimizer far from the minimum for
    // the first 10+ accepted steps, ensuring no overshoot-driven rejections
    // corrupt the Robbins-Monro Î” sequence under test.

    let config = RegularStepGdConfig {
        initial_step_length: 2.0,
        relaxation_factor: 0.5, // all steps accepted on convex quadratic; decay dominates
        minimum_step_length: 1e-4,
        maximum_step_length: 10.0,
        gradient_tolerance: 1e-12, // keep the loop alive
        maximum_iterations: 200,
        learning_rate_decay: 0.1 };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    // Collect step lengths after each accepted step.
    let mut deltas: Vec<f64> = Vec::with_capacity(30);

    for _ in 0..500 {
        if optimizer.converged() {
            break;
        }
        let steps_before = optimizer.steps();
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
        // Record Î” only when an acceptance occurred (steps incremented).
        if optimizer.steps() > steps_before {
            deltas.push(optimizer.current_step_length());
        }
    }

    // Sanity: at least 5 accepted steps
    assert!(
        deltas.len() >= 5,
        "need at least 5 accepted steps to test decay; got {}",
        deltas.len()
    );

    // Î” after k=1: 2.0 / (1 + 0.1Â·1) = 2.0/1.1 â‰ˆ 1.818
    let delta_1 = deltas[0];
    let expected_1 = 2.0 / (1.0 + 0.1 * 1.0);
    assert!(
        (delta_1 - expected_1).abs() < 1e-10,
        "Î” after 1st accepted step must be 2.0/1.1 = {expected_1:.6}; got {delta_1:.6}"
    );

    // Î” after k=5: 2.0 / (1 + 0.1Â·5) = 2.0/1.5 â‰ˆ 1.333
    let delta_5 = deltas[4];
    let expected_5 = 2.0 / (1.0 + 0.1 * 5.0);
    assert!(
        (delta_5 - expected_5).abs() < 1e-10,
        "Î” after 5th accepted step must be 2.0/1.5 = {expected_5:.6}; got {delta_5:.6}"
    );

    // Î” after k=10: 2.0 / (1 + 0.1Â·10) = 2.0/2.0 = 1.0
    let delta_10 = deltas[9];
    let expected_10 = 2.0 / (1.0 + 0.1 * 10.0);
    assert!(
        (delta_10 - expected_10).abs() < 1e-10,
        "Î” after 10th accepted step must be 2.0/2.0 = {expected_10:.6}; got {delta_10:.6}"
    );

    // Monotonic: Î” must be non-increasing across accepted steps
    for window in deltas.windows(2) {
        assert!(
            window[0] >= window[1],
            "Î” must be non-increasing across accepted steps; got {} â†’ {}",
            window[0],
            window[1]
        );
    }
}

// â”€â”€ Robbins-Monro decay is disabled when learning_rate_decay = 0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn rsgd_zero_decay_preserves_step_length() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[5.0, -3.0], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 2.0,
        relaxation_factor: 0.5,
        minimum_step_length: 1e-4,
        maximum_step_length: 10.0,
        gradient_tolerance: 1e-12,
        maximum_iterations: 200,
        learning_rate_decay: 0.0, // explicitly disabled
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    // Verify that accepted steps never change Î” when decay=0.  We compare
    // Î” before and after each accepted step individually (not across
    // consecutive accepted steps, because rejections may shrink Î” between
    // them via relaxation_factor).
    let mut accept_count = 0usize;
    let mut first_accepted_delta: Option<f64> = None;

    for _ in 0..500 {
        if optimizer.converged() {
            break;
        }
        let steps_before = optimizer.steps();
        let delta_before = optimizer.current_step_length();
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);
        let delta_after = optimizer.current_step_length();

        if optimizer.steps() > steps_before {
            // Accepted: decay=0 means Î” must not change from before to after.
            assert_eq!(
                delta_before,
                delta_after,
                "accepted step {} with decay=0 must not change Î”; {} â†’ {}",
                accept_count + 1,
                delta_before,
                delta_after
            );
            if first_accepted_delta.is_none() {
                first_accepted_delta = Some(delta_after);
            }
            accept_count += 1;
        }
    }

    assert!(
        accept_count >= 3,
        "need at least 3 accepted steps; got {accept_count}"
    );

    // First accepted step starts from Î”â‚€.
    assert!(
        (first_accepted_delta.unwrap() - 2.0).abs() < 1e-10,
        "Î” after 1st accepted step must be 2.0 with decay=0; got {}",
        first_accepted_delta.unwrap()
    );
}

// â”€â”€ Revert on loss increase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// xâ‚€ = [1.0], Î”â‚€ = 3.0:
//   effective_lr = 3.0/2.0 = 1.5, xâ‚ = 1.0 âˆ’ 1.5Â·2 = âˆ’2.0 (accepted, first step)
//   L(Î¸â‚) = 4.0 > L(Î¸â‚€) = 1.0 â†’ reject, revert to x = âˆ’2.0, Î”â‚ = 1.5

#[test]
fn rsgd_reverts_on_loss_increase() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[1.0], &device);

    let config = RegularStepGdConfig {
        initial_step_length: 3.0,
        relaxation_factor: 0.5,
        minimum_step_length: 1e-20,
        maximum_step_length: 100.0,
        gradient_tolerance: 1e-30,
        maximum_iterations: 100,
        ..Default::default()
    };

    let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
        RegularStepGradientDescent::new(config);

    // Step 1: L(Î¸â‚€) = 1.0, always accepted
    let loss = module.forward();
    optimizer.set_loss(module.loss_value());
    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &module);
    module = optimizer.step(module, grads_params);

    let first_step_x = module.param_value();
    assert!(
        (first_step_x - (-2.0)).abs() < 0.01,
        "first step should move x to â‰ˆ âˆ’2.0; got {first_step_x:.4}"
    );

    // Step 2: L(Î¸â‚) = 4.0 > prev=1.0 â†’ reject
    let loss = module.forward();
    optimizer.set_loss(module.loss_value());
    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &module);
    module = optimizer.step(module, grads_params);

    let reverted_x = module.param_value();
    assert!(
        (reverted_x - first_step_x).abs() < 0.01,
        "after rejection, x should revert to {first_step_x:.4}; got {reverted_x:.4}"
    );
    assert!(
        (optimizer.current_step_length() - 1.5).abs() < 1e-10,
        "step length should be 1.5; got {}",
        optimizer.current_step_length()
    );
}

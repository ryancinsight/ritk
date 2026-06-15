//! Structural invariant tests for RegularStepGradientDescent.

use super::super::{RegularStepGdConfig, RegularStepGradientDescent};
use super::{Quadratic, TestBackend};
use burn::nn::Linear;

type TestModule = Linear<TestBackend>;

// ── Structural invariants ─────────────────────────────────────────────────────

#[test]
fn rsgd_default_initial_state() {
    let rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
        RegularStepGradientDescent::new(RegularStepGdConfig::default());

    assert!(!rsgd.converged());
    assert_eq!(rsgd.convergence_reason(), None);
    assert_eq!(rsgd.steps(), 0);
    assert!((rsgd.current_step_length() - 1.0).abs() < 1e-12);
    assert!((rsgd.learning_rate() - 1.0).abs() < 1e-12);
}

#[test]
fn rsgd_set_learning_rate_updates_step_length() {
    let mut rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
        RegularStepGradientDescent::new(RegularStepGdConfig::default());

    rsgd.set_learning_rate(0.5);
    assert!((rsgd.learning_rate() - 0.5).abs() < 1e-12);
    assert!((rsgd.current_step_length() - 0.5).abs() < 1e-12);
}

use crate::optimizer::OptimizerAlgorithm;

#[test]
fn rsgd_telemetry_reports_algorithm_name() {
    let rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
        RegularStepGradientDescent::new(RegularStepGdConfig::default());

    let telemetry = rsgd.telemetry();
    assert_eq!(
        telemetry.algorithm,
        OptimizerAlgorithm::RegularStepGradientDescent
    );
    assert_eq!(telemetry.steps, 0);
    assert!(telemetry.learning_rate.is_some());
}

#[test]
fn rsgd_set_loss_stores_current_loss() {
    let mut rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
        RegularStepGradientDescent::new(RegularStepGdConfig::default());

    rsgd.set_loss(1.5);
    assert!(!rsgd.converged());
}

use crate::optimizer::Optimizer;
use burn::optim::GradientsParams;

#[test]
fn rsgd_returns_module_unchanged_after_convergence() {
    let device = Default::default();
    let mut module = Quadratic::<TestBackend>::new(&[1e-8], &device);

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

    for _ in 0..20 {
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

    assert!(optimizer.converged());

    let pre_step_x = module.x.val().to_data();
    let pre_slice = pre_step_x.as_slice::<f32>().unwrap().to_vec();

    let loss = module.forward();
    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &module);

    optimizer.set_loss(1.0);
    module = optimizer.step(module, grads_params);

    let post_step_x = module.x.val().to_data();
    let post_slice = post_step_x.as_slice::<f32>().unwrap();

    for (i, (pre, post)) in pre_slice.iter().zip(post_slice.iter()).enumerate() {
        assert!(
            (pre - post).abs() < 1e-10,
            "Parameter [{i}] unchanged after convergence: pre={pre}, post={post}"
        );
    }
}

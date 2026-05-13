//! Config validation and ConvergenceReason equality tests.

use super::super::{ConvergenceReason, RegularStepGdConfig};

// ── Config validation ─────────────────────────────────────────────────────────

#[test]
fn default_config_validates() {
    assert!(RegularStepGdConfig::default().validate().is_ok());
}

#[test]
fn config_rejects_zero_initial_step() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.initial_step_length = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_negative_initial_step() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.initial_step_length = -1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_relaxation_zero() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.relaxation_factor = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_relaxation_one() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.relaxation_factor = 1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_min_step_ge_initial() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.minimum_step_length = 1.0;
    assert!(cfg.validate().is_err());

    cfg.minimum_step_length = 2.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_min_step_ge_max() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.minimum_step_length = 10.0;
    cfg.maximum_step_length = 10.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_zero_gradient_tolerance() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.gradient_tolerance = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_zero_max_iterations() {
    let mut cfg = RegularStepGdConfig::default();
    cfg.maximum_iterations = 0;
    assert!(cfg.validate().is_err());
}

// ── ConvergenceReason equality ────────────────────────────────────────────────

#[test]
fn convergence_reason_equality() {
    assert_eq!(
        ConvergenceReason::GradientConvergence,
        ConvergenceReason::GradientConvergence
    );
    assert_ne!(
        ConvergenceReason::GradientConvergence,
        ConvergenceReason::StepConvergence
    );
    assert_ne!(
        ConvergenceReason::StepConvergence,
        ConvergenceReason::MaximumIterations
    );
}

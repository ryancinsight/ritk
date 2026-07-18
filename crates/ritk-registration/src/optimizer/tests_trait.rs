use super::*;

/// Tolerance for exact scheduler arithmetic assertions (f64 precision).
const SCHEDULER_TOL: f64 = 1e-12;

// â”€â”€ StepDecay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// At step=0 the exponent is 0, so get_lr returns initial_lr * gamma^0 = initial_lr.
#[test]
fn step_decay_initial_step_returns_initial_lr() {
    let sched = StepDecay::new(10, 0.1);
    let lr = sched.get_lr(0, 1.0);
    assert!(
        (lr - 1.0).abs() < SCHEDULER_TOL,
        "step=0 must return initial_lr unchanged; got {lr}"
    );
}

/// After exactly `step_size` steps, one decay is applied: lr = initial * gamma.
///
/// # Derivation
/// step=10, step_size=10 â†’ exponent=1 â†’ lr = 0.5 * 0.5^1 = 0.25.
#[test]
fn step_decay_applies_one_decay_at_first_boundary() {
    let sched = StepDecay::new(10, 0.5);
    let lr = sched.get_lr(10, 0.5);
    let expected = 0.5 * 0.5_f64.powi(1);
    assert!(
        (lr - expected).abs() < SCHEDULER_TOL,
        "step=10 with step_size=10, initial=0.5, gamma=0.5: expected {expected}, got {lr}"
    );
}

/// Learning rate is monotone non-increasing with step count.
#[test]
fn step_decay_monotone_non_increasing() {
    let sched = StepDecay::new(5, 0.5);
    let initial_lr = 1.0;
    let lrs: Vec<f64> = (0..=25).map(|s| sched.get_lr(s, initial_lr)).collect();
    for window in lrs.windows(2) {
        assert!(
            window[0] >= window[1],
            "LR must be non-increasing: {:.6} >= {:.6}",
            window[0],
            window[1]
        );
    }
}

/// With gamma=1.0, the learning rate never decays.
#[test]
fn step_decay_gamma_one_constant_lr() {
    let sched = StepDecay::new(1, 1.0);
    let initial_lr = 0.01;
    for step in 0..100 {
        let lr = sched.get_lr(step, initial_lr);
        assert!(
            (lr - initial_lr).abs() < SCHEDULER_TOL,
            "gamma=1.0 must produce constant LR; step={step}, got {lr}"
        );
    }
}

/// Two boundary steps apart: exponent increments by 1 each time.
///
/// # Derivation
/// step_size=3, gamma=0.5:
///   step=0  â†’ exp=0 â†’ lr=initial
///   step=3  â†’ exp=1 â†’ lr=initial*0.5
///   step=6  â†’ exp=2 â†’ lr=initial*0.25
#[test]
fn step_decay_multiple_boundaries_correct() {
    let sched = StepDecay::new(3, 0.5);
    let lr0 = sched.get_lr(0, 1.0);
    let lr3 = sched.get_lr(3, 1.0);
    let lr6 = sched.get_lr(6, 1.0);
    assert!((lr0 - 1.0).abs() < SCHEDULER_TOL, "step=0 â†’ 1.0; got {lr0}");
    assert!((lr3 - 0.5).abs() < SCHEDULER_TOL, "step=3 â†’ 0.5; got {lr3}");
    assert!(
        (lr6 - 0.25).abs() < SCHEDULER_TOL,
        "step=6 â†’ 0.25; got {lr6}"
    );
}

// â”€â”€ OptimizerTelemetry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn optimizer_telemetry_debug_and_eq() {
    let t1 = OptimizerTelemetry {
        algorithm: OptimizerAlgorithm::GradientDescent,
        steps: 42,
        learning_rate: Some(1e-3) };
    let t2 = t1.clone();
    assert_eq!(t1, t2);
    assert!(format!("{t1:?}").contains("GradientDescent"));
}

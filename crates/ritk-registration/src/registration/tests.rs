use super::*;
use crate::optimizer::OptimizerTelemetry;
use crate::registration::summary::StopReason;

#[test]
fn test_registration_config_default() {
    let config = RegistrationConfig::default();
    assert_eq!(config.early_stopping, EarlyStoppingPolicy::Disabled);
    assert_eq!(config.log_interval, 50);
}

#[test]
fn test_registration_config_builder() {
    let config = RegistrationConfig::new()
        .with_early_stopping(10, 1e-5)
        .with_log_interval(25);

    assert_eq!(
        config.early_stopping,
        EarlyStoppingPolicy::Enabled {
            patience: 10,
            min_improvement: 1e-5
        }
    );
    assert_eq!(config.log_interval, 25);
}

#[test]
fn registration_summary_holds_execution_diagnostics() {
    let summary = RegistrationSummary {
        transform: 3_u32,
        loss_history: vec![2.0, 1.0],
        optimizer_telemetry: OptimizerTelemetry {
            algorithm: "Test",
            steps: 2,
            learning_rate: Some(0.1),
        },
        iterations_completed: 2,
        final_loss: 1.0,
        stop_reason: StopReason::Completed,
    };

    assert_eq!(summary.transform, 3);
    assert_eq!(summary.optimizer_telemetry.steps, 2);
    assert_eq!(summary.final_loss, 1.0);
}

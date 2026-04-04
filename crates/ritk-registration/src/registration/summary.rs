use crate::optimizer::OptimizerTelemetry;

/// Summary returned by registration workflows that need execution diagnostics.
#[derive(Debug)]
pub struct RegistrationSummary<T> {
    pub transform: T,
    pub loss_history: Vec<f64>,
    pub optimizer_telemetry: OptimizerTelemetry,
    pub iterations_completed: usize,
    pub final_loss: f64,
    pub stopped_early: bool,
}

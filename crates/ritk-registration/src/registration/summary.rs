use crate::optimizer::OptimizerTelemetry;

/// Why a registration loop terminated before exhausting all configured iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// All configured iterations were executed without early termination.
    Completed,
    /// The loop exited early due to the early-stopping or convergence policy.
    EarlyStopping,
}

/// Summary returned by registration workflows that need execution diagnostics.
#[derive(Debug)]
pub struct RegistrationSummary<T> {
    pub transform: T,
    pub loss_history: Vec<f64>,
    pub optimizer_telemetry: OptimizerTelemetry,
    pub iterations_completed: usize,
    pub final_loss: f64,
    pub stop_reason: StopReason,
}

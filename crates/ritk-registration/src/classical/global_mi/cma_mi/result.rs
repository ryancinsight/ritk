// ─── Result ───────────────────────────────────────────────────────────────────

use crate::optimizer::CmaEsStopReason;
use crate::types::AffineTransform;

/// Result produced by [`CmaMiRegistration::register_rigid`](super::CmaMiRegistration::register_rigid).
#[derive(Debug, Clone)]
pub struct CmaMiResult {
    /// 4×4 homogeneous matrix of the final transform (row-major, f64).
    pub matrix: AffineTransform,

    /// Final MI value (positive; negated from the CMA-ES loss).
    /// Reflects the CMA-ES coarse-level MI, not the full-resolution value
    /// even when RSGD refinement is applied.
    pub final_mi: f64,

    /// Number of CMA-ES generations executed (last level for cascade mode).
    pub cma_generations: usize,

    /// Reason the CMA-ES loop terminated (last level for cascade mode).
    pub cma_stop_reason: CmaEsStopReason,

    /// CMA-ES final step-size σ (last level for cascade mode).
    pub cma_final_sigma: f64,

    /// Total RSGD iterations across all resolution levels (0 if no refinement).
    pub rsgd_iterations: usize,

    /// Per-iteration loss history from RSGD refinement (empty if no refinement).
    pub rsgd_loss_history: Vec<f64>,

    /// Normalised CMA-ES best parameter vector `[α_n, β_n, γ_n, tz_n, ty_n, tx_n]`.
    /// Each component is in `[−1, 1]`; multiply by `rotation_range_rad` (first 3) or
    /// `translation_range_mm` (last 3) to recover physical units.
    /// Populated from the last cascade level in multi-scale mode.
    pub cma_best_params: Vec<f64>,
}

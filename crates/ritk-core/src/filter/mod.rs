// ── Thin shims re-exporting to extracted crates ────────────────────────
//
// All filter implementations moved to `ritk-filter` (Sprint 361).
// `ops` was extracted to `ritk-tensor-ops` (Sprint 361, Phase 3).
// `kernel_utils` (gaussian_kernel) was extracted to `ritk-tensor-ops`
// (Sprint 361, Phase 5). This module is a compatibility shim; callers
// should prefer importing directly from `ritk_tensor_ops`.
pub mod ops;

pub use ritk_tensor_ops::gaussian_kernel;

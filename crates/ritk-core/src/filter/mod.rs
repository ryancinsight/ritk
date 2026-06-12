// ── ops module: thin shim re-exporting to `ritk_tensor_ops` ────────────────
//
// All filter implementations moved to `ritk-filter` (Sprint 361).
// `ops` was extracted to `ritk-tensor-ops` (Sprint 361, Phase 3) to break
// the circular dependency (`ritk-filter` → `ritk-core` ← `ritk-filter`).
// This module is a compatibility shim; callers should prefer importing from
// `ritk_tensor_ops` directly.
// `kernel_utils` owns `gaussian_kernel` (the only remaining substantive code
// in ritk-core's filter module).
pub mod kernel_utils;
pub mod ops;

pub use kernel_utils::gaussian_kernel;

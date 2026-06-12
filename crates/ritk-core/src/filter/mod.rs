// ── ops module: shared pixel-buffer I/O helpers retained in ritk-core ───────
//
// All other filter modules moved to `ritk-filter` (Sprint 361).
// `ops` stays here because `ritk-core::statistics` also uses `extract_vec` /
// `extract_vec_infallible` / `rebuild` and moving them would create a circular
// dependency (`ritk-filter` → `ritk-core` ← `ritk-filter`).
// `kernel_utils` owns `gaussian_kernel` separately so ops stays focused on
// tensor extract/rebuild helpers.
pub mod kernel_utils;
pub mod ops;

pub use kernel_utils::gaussian_kernel;

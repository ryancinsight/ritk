// ── ops module: shared pixel-buffer I/O helpers retained in ritk-core ───────
//
// All other filter modules moved to `ritk-filter` (Sprint 361).
// `ops` stays here because `ritk-core::statistics` also uses `extract_vec` /
// `extract_vec_infallible` / `rebuild` / `gaussian_kernel_1d` and moving them
// would create a circular dependency (`ritk-filter` → `ritk-core` ← `ritk-filter`).
pub mod ops;

pub use ops::gaussian_kernel_1d;

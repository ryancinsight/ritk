// SSOT: keep in sync with `ritk_wgpu_compat::WGPU_CHUNK_SIZE` (which `ritk-registration`
// depends on directly). This constant is reserved as the future single source of truth
// once callers are migrated away from `ritk_wgpu_compat::` direct references.
// [arch] ExecutionPolicy::max_batch_size() will replace this pattern.
#[allow(dead_code)]
pub(crate) const WGPU_CHUNK_SIZE: usize = 32_768;

#![cfg(feature = "direct-parzen")]

//! Atlas-typed sibling for the Parzen-window cache surface (sub-batch #3.b of
//! atlas-meta Batch #3, per `atlas/docs/adr/0012-ritk-burn-trait-rebind.md`).
//!
//! # Summary
//!
//! Provides Atlas-friendly wrapper functions around the leaf direct-path CPU
//! algorithms in `super::direct`. The leaf algorithms operate on host-side
//! `&[f32]` slices — but their return types still carry burn-tensor
//! representation (`TensorData`, `SparseWFixedT`). The wrappers here unpack
//! those burn-tensor return types into plain host `Vec<f32>` /
//! `Vec<(Vec<(u16, f32)>, f32)>` (the latter being the flattened
//! sparse-cache form) so callers can use the cache surface without any
//! burn-tensor indirection.
//!
//! # Why wrappers, not `pub use` aliases
//!
//! The leaf direct-path CPU algorithms are already backend-agnostic at the
//! input level (operate on `&[f32]` host slices). Their return types still
//! hold burn-tensor representation (`TensorData`, `SparseWFixedT`). `pub use`
//! aliases would propagate those burn types to the caller, defeating the
//! purpose of an Atlas-side sibling. The wrappers here do the
//! `TensorData.as_slice::<f32>().to_vec()` and the
//! `Vec<(SparseSampleCache, f32)>` unpack at one site, then return the
//! flattened host form. Binary erosion now exposes its Coeus-native function
//! directly instead of duplicating the operation behind a second state type.
//!
//! # Cargo.toml
//!
//! No manifest mutation. The dependency graph on `coeus-tensor` was already
//! in place on `repos/ritk/crates/ritk-registration/Cargo.toml` (from
//! sub-batch #2 readiness); the leaf-level re-exports need no new dep.
//! `xtask/burn_surface.allowlist` contracts by exactly 1 source-row on the
//! post-rewrite `tests/cache_property_tests.rs` (sub-batch #3.b invariant).
//!
//! # Atomic-boundary invariant (per ADR 0012 §Decision §1 + §atomic-boundary discipline §2)
//!
//! **Strictly additive on production surface**:
//! - `atlas_parzen_cache.rs` is the new module; no legacy Burn-keyed
//!   surface is touched.
//! - No `#[deprecated]` attribute on any Burn-keyed item (would emit a
//!   compile-warning cascade per sub-batch #2 carryover invariant).
//! - No `Cargo.toml` mutation (this sub-batch's manifest stance).
//! - The Type symbol `ParzenConfig` (`pub(crate)` in `direct::ParzenConfig`)
//!   is consumed by the test through the crate-local path
//!   `crate::metric::histogram::parzen::direct::ParzenConfig` (Rust rejects
//!   visibility-elevation of `pub(crate)` items via `pub use ... as Alias`).
//!
//! # Atlas-side host-slice normalisation helper
//!
//! [`atlas_normalize_intensities`] mirrors the legacy
//! `dispatch::normalize_and_extract` algorithm shape (multiplicative
//! scale + offset + clamp to `[0, num_bins - 1]`) but operates on plain
//! `&[f32]` host slices — no burn-tensor indirection. The legacy
//! `dispatch::normalize_and_extract` is `pub(in crate::metric::histogram)`
//! and gets the `Tensor<B, 1>` → `Cow<'static, [f32]>` shape from
//! `burn::Backend`; the Atlas twin is endpoint-and-shape compatible but
//! takes plain `&[f32]` because Atlas-side tests do not need the
//! autodiff-tape preservation contract (Atlas substrate has its own
//! autograd path through `coeus_autograd`).

use super::direct::{
    build_sparse_w_fixed_transposed as legacy_build,
    compute_joint_histogram_direct as legacy_compute, HistogramPool, SparseWFixedT,
};

/// Atlas-side flattened sparse-cache entry: `bin` index in
/// `[0, num_bins - 1]` (a `u16`) paired with the `weight` (a `f32` Parzen
/// weight for the sample at that bin). Mirrors `direct::SparseWFixedEntry`
/// but is host-side (no burn-tensor representation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasSparseEntry {
    pub bin: u16,
    pub weight: f32,
}

/// Atlas-side `compute_joint_histogram_direct` wrapper that returns a host
/// `Vec<f32>` instead of `burn::prelude::TensorData`.
///
/// Internally calls the leaf direct-path CPU algorithm in
/// `super::direct::compute_joint_histogram_direct`, which returns
/// `TensorData`. The wrapper extracts the underlying `&[f32]` to a host
/// `Vec<f32>` so the caller does not need any burn-tensor indirection.
#[inline]
pub fn compute_atlas_joint_histogram_direct(
    fixed_norm: &[f32],
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> Vec<f32> {
    let data = legacy_compute(
        fixed_norm,
        moving_norm,
        num_bins,
        sigma_sq_fix,
        sigma_sq_mov,
        oob_mask,
        pool,
    );
    data.as_slice::<f32>()
        .expect("compute_joint_histogram_direct returned non-f32 TensorData")
        .to_vec()
}

/// Atlas-side `build_sparse_w_fixed_transposed` wrapper that returns a host
/// `Vec<(Vec<AtlasSparseEntry>, f32)>` (flat sparse-cache form) instead of
/// the burn-representation `SparseWFixedT = Vec<(SparseSampleCache, f32)>`.
///
/// Internally calls the leaf direct-path CPU algorithm in
/// `super::direct::build_sparse_w_fixed_transposed`, which returns
/// `SparseWFixedT` (a `Vec` of `(SparseSampleCache, f32)` tuples). The
/// wrapper unpacks each `SparseSampleCache` (which `Deref`s to
/// `[SparseWFixedEntry]`) into `Vec<AtlasSparseEntry>`, paired with the
/// `inv_sum_f: f32` factor, so the caller does not need any burn-tensor
/// indirection. Each `AtlasSparseEntry` carries named fields
/// (`bin: u16`, `weight: f32`) preserving semantics from
/// `direct::SparseWFixedEntry`.
#[inline]
pub fn build_atlas_sparse_w_fixed_transposed(
    fixed_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    oob_mask: Option<&[f32]>,
) -> Vec<(Vec<AtlasSparseEntry>, f32)> {
    let sparse: SparseWFixedT = legacy_build(fixed_norm, num_bins, sigma_sq_fix, oob_mask);
    sparse
        .into_iter()
        .map(|(cache, inv_sum_f)| {
            let entries = cache
                .iter()
                .map(|e| AtlasSparseEntry {
                    bin: e.bin,
                    weight: e.weight,
                })
                .collect();
            (entries, inv_sum_f)
        })
        .collect()
}

/// Atlas-side normalisation: input intensity range → `[0, num_bins - 1]`
/// host slice, clamped.
///
/// Mirrors the legacy `dispatch::normalize_and_extract` algorithm shape
/// (multiplicative scale + offset + clamp) without going through
/// `burn::Tensor<B, 1>`.
#[inline]
pub fn atlas_normalize_intensities(
    values: &[f32],
    min_intensity: f32,
    max_intensity: f32,
    num_bins: usize,
) -> Vec<f32> {
    let num_bins_f = (num_bins - 1) as f32;
    let scale = num_bins_f / (max_intensity - min_intensity);
    let offset = -min_intensity * scale;
    values
        .iter()
        .map(|&v| (v * scale + offset).clamp(0.0, num_bins_f))
        .collect()
}

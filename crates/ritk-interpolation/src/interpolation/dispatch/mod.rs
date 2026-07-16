//! Dimension dispatch for interpolators.
//!
//! Routes interpolation calls to the correct dimension-specific implementation
//! based on `const D: usize`. Only `D ∈ {1, 2, 3, 4}` is supported; other
//! values panic at runtime. Because the dispatch is a simple `match` over a
//! `const` parameter, the compiler monomorphizes each branch and dead-code
//! eliminates unreachable arms — achieving the same zero-cost dispatch as the
//! previous sealed-trait approach, but without requiring a `where` bound on
//! callers.
//!
//! # Per-shape typed specialization (audit §8 351-01)
//!
//! For `D = 3`, [`dispatch_linear`] delegates to the sealed [`DispatchByShape`] trait,
//! which matches on the runtime shape and routes common volumes
//! (64³, 128³, 256³, 512³) to the **const-generic** typed instantiations
//! in `kernel::linear::dim3::interpolate_3d_typed::<B, D0, D1, D2>`.
//! This gives automatic speedup for known shapes without requiring callers
//! to change their API.
//!
//! Trade-off: each monomorphized typed function adds a small amount of
//! compile-time code. We currently specialise 4 common 3-D cube shapes
//! (64³, 128³, 256³, 512³) — the most common registration volumes.
//! Uncommon shapes (e.g. 200×300×400 clinical scans) fall through to the
//! generic `interpolate_3d` path with no overhead.
//!
//! # Trait-based shape dispatch (Sprint 357)
//!
//! The shape-routing logic is encapsulated in the sealed [`DispatchByShape`]
//! trait, implemented **only** for `Tensor<B, 3>`. The trait method
//! [`DispatchByShape::dispatch_by_shape`] performs the runtime shape match
//! and routes to the appropriate typed instantiation.
//!
//! To bridge the `&Tensor<B, D>` (generic D) → `&Tensor<B, 3>` gap in
//! [`dispatch_linear`], we use a second trait [`Dispatch3DTyped`] that is
//! implemented for any `Tensor<B, D>`. When `D == 3` (a compile-time const
//! check, constant-folded away by the compiler), it routes through the
//! sealed trait method. This is **safer than `unsafe` transmute** — the
//! narrowing is encapsulated in a named, documented trait method — and
//! **more complex** (two traits, sealed module, `match D` dispatch), but
//! preserves the zero-cost abstraction goal: the `if D == 3` branch is
//! dead-code eliminated for non-3-D callers.

use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

use super::shared::OutOfBoundsMode;

pub mod linear;
pub mod nearest;

// ── Re-exports: preserve `interpolation::dispatch::*` API ───────────────
pub use linear::{
    Dispatch1DTyped, Dispatch2DTyped, Dispatch3DTyped, Dispatch4DTyped, DispatchByShape,
};
pub use nearest::{DispatchNearest3DTyped, DispatchNearestByShape};

// ════════════════════════════════════════════════════════════════════════
// Sealed trait for shape-based dispatch (Sprint 357)
// ════════════════════════════════════════════════════════════════════════
//
// The `DispatchByShape` trait is sealed: only `Tensor<B, 3>` can implement
// it. This prevents external code from adding implementations that might
// violate the invariants of the shape-based routing (e.g. calling the typed
// functions with the wrong shape). The `Sealed` supertrait is in a private
// `sealed` module, so external code cannot name it in an impl.

pub(crate) mod sealed {
    //! Sealed module — prevents external implementations of [`super::DispatchByShape`].
    //!
    /// `Sealed` is implemented for `Tensor<B, 1>`, `Tensor<B, 2>`,
    /// `Tensor<B, 3>`, and `Tensor<B, 4>` — the only dimensions the
    /// per-shape dispatchers support. External code cannot add
    /// implementations because the `Sealed` supertrait is in a
    /// private module.
    use ritk_image::tensor::Backend;
    use ritk_image::tensor::Tensor;
    pub trait Sealed {}
    impl<B: Backend> Sealed for Tensor<B, 1> {}
    impl<B: Backend> Sealed for Tensor<B, 2> {}
    impl<B: Backend> Sealed for Tensor<B, 3> {}
    impl<B: Backend> Sealed for Tensor<B, 4> {}
}

/// Linear interpolation dispatch.
///
/// Delegates to the dimension-specific implementation for `D`.
/// For `D ∈ {1, 2, 3, 4}`, routes through the corresponding
/// `DispatchNDTyped` type-narrowing wrapper
/// ([`Dispatch1DTyped`], [`Dispatch2DTyped`], [`Dispatch3DTyped`],
/// [`Dispatch4DTyped`]) which uses the sealed [`DispatchByShape`]
/// trait method for per-shape const-generic specialization
/// (audit §8 351-01, 351-01-ND-TYPED).
/// Panics if `D ∉ {1, 2, 3, 4}`.
#[inline]
pub fn dispatch_linear<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    match D {
        1 => data.dispatch_1d_typed(indices, mode),
        2 => data.dispatch_2d_typed(indices, mode),
        3 => data.dispatch_3d_typed(indices, mode),
        4 => data.dispatch_4d_typed(indices, mode),
        _ => panic!("Linear interpolation only supports D ∈ {{1, 2, 3, 4}}, got D = {D}"),
    }
}

/// Nearest-neighbor interpolation dispatch.
///
/// Delegates to the dimension-specific implementation for `D`.
/// For `D = 3`, routes through [`DispatchNearest3DTyped::dispatch_nearest_3d_typed`]
/// (which uses the sealed [`DispatchNearestByShape`] trait method) for
/// per-shape const-generic specialization. Currently the trait method
/// falls through to the generic `nearest::interpolate_3d` for all
/// shapes (no typed nearest-neighbor instantiations yet) — the trait
/// abstraction is in place so typed variants can be added without
/// changing the public API.
/// Panics if `D ∉ {1, 2, 3, 4}`.
#[inline]
pub fn dispatch_nearest<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    match D {
        1 => super::kernel::nearest::interpolate_1d(data, indices, mode),
        2 => super::kernel::nearest::interpolate_2d(data, indices, mode),
        3 => data.dispatch_nearest_3d_typed(indices, mode),
        4 => super::kernel::nearest::interpolate_4d(data, indices, mode),
        _ => panic!("Nearest-neighbor interpolation only supports D ∈ {{1, 2, 3, 4}}, got D = {D}"),
    }
}

/// Per-shape runtime dispatcher for linear interpolation.
///
/// Alias for [`dispatch_linear`] — the canonical const-generic entry point.
/// Routes D-dimensional tensors to the correct const-generic typed
/// instantiation based on `D` and the runtime shape.
///
/// # Panics
/// Panics if `D ∉ {1, 2, 3, 4}`.
pub use dispatch_linear as dispatch_for_shape;

/// Per-shape runtime dispatcher for nearest-neighbor interpolation.
///
/// Alias for [`dispatch_nearest`] — the canonical const-generic entry point.
/// For `D = 3`, routes through the sealed [`DispatchNearestByShape`] trait
/// method; for other supported dimensions, falls through to the generic path.
///
/// # Panics
/// Panics if `D ∉ {1, 2, 3, 4}`.
pub use dispatch_nearest as dispatch_nearest_for_shape;

#[cfg(test)]
#[path = "../tests_dispatch/mod.rs"]
mod tests_dispatch;

/// Cross-dimension routing smoke tests (distinct from [`tests_dispatch`],
/// which covers per-kernel dispatch correctness).
#[cfg(test)]
#[path = "tests_dispatch.rs"]
mod tests_dispatch_routing;

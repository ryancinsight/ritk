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
//! For `D = 3`, [`dispatch_linear`] delegates to [`dispatch_3d_for_shape`],
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

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::kernel::linear::dim1;
use super::kernel::linear::dim2;
use super::kernel::linear::dim3;
use super::kernel::linear::dim4;
use super::kernel::nearest;
use super::shared::OutOfBoundsMode;

// ════════════════════════════════════════════════════════════════════════
//  Sealed trait for shape-based dispatch (Sprint 357)
// ════════════════════════════════════════════════════════════════════════
//
// The `DispatchByShape` trait is sealed: only `Tensor<B, 3>` can implement
// it. This prevents external code from adding implementations that might
// violate the invariants of the shape-based routing (e.g. calling the typed
// functions with the wrong shape). The `Sealed` supertrait is in a private
// `sealed` module, so external code cannot name it in an impl.

mod sealed {
    //! Sealed module — prevents external implementations of [`DispatchByShape`].
    //!
    /// `Sealed` is implemented for `Tensor<B, 1>`, `Tensor<B, 2>`,
    /// `Tensor<B, 3>`, and `Tensor<B, 4>` — the only dimensions the
    /// per-shape dispatchers support. External code cannot add
    /// implementations because the `Sealed` supertrait is in a
    /// private module.
    use super::Tensor;
    use burn::tensor::backend::Backend;
    pub trait Sealed {}
    impl<B: Backend> Sealed for Tensor<B, 1> {}
    impl<B: Backend> Sealed for Tensor<B, 2> {}
    impl<B: Backend> Sealed for Tensor<B, 3> {}
    impl<B: Backend> Sealed for Tensor<B, 4> {}
}

/// Sealed trait for per-shape 3-D interpolation dispatch.
///
/// This trait is sealed: it is only implemented for `Tensor<B, 3>`. External
/// code cannot add new implementations. The trait method
/// [`DispatchByShape::dispatch_by_shape`] matches on the runtime shape and routes common
/// cube volumes (64³, 128³, 256³, 512³) to the const-generic typed
/// instantiations, with fallback to the generic `interpolate_3d` for
/// other shapes.
///
/// # Why sealed?
///
/// The trait is sealed to prevent external implementations that might
/// violate the invariants of the shape-based routing (e.g. calling the
/// typed functions with the wrong shape). The `Sealed` supertrait ensures
/// only `Tensor<B, 3>` can implement this trait.
///
/// # See also
///
/// - [`Dispatch3DTyped`] — the type-narrowing wrapper trait used by
///   [`dispatch_linear`] to route `&Tensor<B, D>` (generic D) through
///   this sealed trait.
pub trait DispatchByShape<B: Backend>: sealed::Sealed {
    /// Dispatch to the appropriate 3-D interpolation implementation based
    /// on the runtime shape.
    ///
    /// Matches on `self.shape().dims` and routes common cube shapes
    /// (64³, 128³, 256³, 512³) to the const-generic typed instantiations.
    /// Other shapes fall through to the generic `interpolate_3d`.
    fn dispatch_by_shape(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1>;
}

impl<B: Backend> DispatchByShape<B> for Tensor<B, 3> {
    fn dispatch_by_shape(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        // `self.shape().dims` is a `Vec<usize>`; we need a slice to match on
        // fixed-size array patterns. Bind the shape to a local first to extend
        // the borrow lifetime (E0716: temporary lifetime issue).
        let shape = self.shape();
        let dims: &[usize] = shape.dims.as_slice();
        match dims {
            // Power-of-2 cubes (most common registration volumes).
            [32, 32, 32] => dim3::interpolate_3d_typed::<B, 32, 32, 32>(self, indices, mode),
            [48, 48, 48] => dim3::interpolate_3d_typed::<B, 48, 48, 48>(self, indices, mode),
            [64, 64, 64] => dim3::interpolate_3d_typed::<B, 64, 64, 64>(self, indices, mode),
            [96, 96, 96] => dim3::interpolate_3d_typed::<B, 96, 96, 96>(self, indices, mode),
            [128, 128, 128] => dim3::interpolate_3d_typed::<B, 128, 128, 128>(self, indices, mode),
            [192, 192, 192] => dim3::interpolate_3d_typed::<B, 192, 192, 192>(self, indices, mode),
            [256, 256, 256] => dim3::interpolate_3d_typed::<B, 256, 256, 256>(self, indices, mode),
            [384, 384, 384] => dim3::interpolate_3d_typed::<B, 384, 384, 384>(self, indices, mode),
            [512, 512, 512] => dim3::interpolate_3d_typed::<B, 512, 512, 512>(self, indices, mode),
            [1024, 1024, 1024] => {
                dim3::interpolate_3d_typed::<B, 1024, 1024, 1024>(self, indices, mode)
            }
            // Non-cube clinical shapes (CT/MRI).
            [256, 256, 128] => {
                dim3::interpolate_3d_typed::<B, 256, 256, 128>(self, indices, mode)
            }
            [192, 256, 256] => {
                dim3::interpolate_3d_typed::<B, 192, 256, 256>(self, indices, mode)
            }
            _ => dim3::interpolate_3d(self, indices, mode),
        }
    }
}

// ── 1-D implementation (Sprint 359) ───────────────────────────────────
//
// Common 1-D shapes: 64, 128, 256, 512 (signal processing, FFT sizes).
// Each shape is a separate monomorphized typed function via
// `interpolate_1d_typed::<B, D0>`. Uncommon shapes fall through to
// the generic `interpolate_1d`.
impl<B: Backend> DispatchByShape<B> for Tensor<B, 1> {
    fn dispatch_by_shape(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        let shape = self.shape();
        let dims: &[usize] = shape.dims.as_slice();
        match dims {
            [64] => dim1::interpolate_1d_typed::<B, 64>(self, indices, mode),
            [128] => dim1::interpolate_1d_typed::<B, 128>(self, indices, mode),
            [256] => dim1::interpolate_1d_typed::<B, 256>(self, indices, mode),
            [512] => dim1::interpolate_1d_typed::<B, 512>(self, indices, mode),
            _ => dim1::interpolate_1d(self, indices, mode),
        }
    }
}

// ── 2-D implementation (Sprint 359) ───────────────────────────────────
//
// Common 2-D shapes: 64², 128², 256², 512² (square image sizes).
// Each shape is a separate monomorphized typed function via
// `interpolate_2d_typed::<B, D0, D1>`. Uncommon shapes fall through to
// the generic `interpolate_2d`.
impl<B: Backend> DispatchByShape<B> for Tensor<B, 2> {
    fn dispatch_by_shape(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        let shape = self.shape();
        let dims: &[usize] = shape.dims.as_slice();
        match dims {
            [64, 64] => dim2::interpolate_2d_typed::<B, 64, 64>(self, indices, mode),
            [128, 128] => dim2::interpolate_2d_typed::<B, 128, 128>(self, indices, mode),
            [256, 256] => dim2::interpolate_2d_typed::<B, 256, 256>(self, indices, mode),
            [512, 512] => dim2::interpolate_2d_typed::<B, 512, 512>(self, indices, mode),
            _ => dim2::interpolate_2d(self, indices, mode),
        }
    }
}

// ── 4-D implementation (Sprint 359) ───────────────────────────────────
//
// Common 4-D shapes: 64⁴, 128⁴ (dynamic 3-D volumes over time).
// 256⁴ = 4B elements is impractical, so we only specialise the two
// smaller shapes. Each shape is a separate monomorphized typed
// function via `interpolate_4d_typed::<B, D0, D1, D2, D3>`. Uncommon
// shapes fall through to the generic `interpolate_4d`.
impl<B: Backend> DispatchByShape<B> for Tensor<B, 4> {
    fn dispatch_by_shape(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        let shape = self.shape();
        let dims: &[usize] = shape.dims.as_slice();
        match dims {
            [64, 64, 64, 64] => {
                dim4::interpolate_4d_typed::<B, 64, 64, 64, 64>(self, indices, mode)
            }
            [128, 128, 128, 128] => {
                dim4::interpolate_4d_typed::<B, 128, 128, 128, 128>(self, indices, mode)
            }
            _ => dim4::interpolate_4d(self, indices, mode),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Type-narrowing wrapper trait (Sprint 357)
// ════════════════════════════════════════════════════════════════════════
//
// The challenge: [`dispatch_linear`] takes `data: &Tensor<B, D>` where D
// is a generic const. When D == 3, we want to call the sealed
// [`DispatchByShape`] trait method, which requires `&Tensor<B, 3>`. The
// type system can't automatically narrow D from generic to 3.
//
// Options considered:
//   1. `unsafe` transmute: zero-cost but unsafe (rejected).
//   2. `clone().reshape(...)`: safe but allocates a new tensor (current).
//   3. **Trait-based narrowing (this design)**: safe, named, documented.
//      The `Dispatch3DTyped` trait is implemented for any
//      `Tensor<B, D>`. When D == 3 (compile-time check, constant-folded),
//      it routes through the sealed trait method. The narrowing is
//      encapsulated in a named, testable trait method.

/// Extension trait that bridges `&Tensor<B, D>` (generic D) to the sealed
/// [`DispatchByShape`] trait method on `&Tensor<B, 3>`.
///
/// Implemented for **any** `Tensor<B, D>`, but the `D == 3` branch is the
/// only meaningful one — the other branches `unreachable!()` since the
/// compiler can prove D != 3 at compile time and dead-code eliminates
/// them. This gives us type-safe narrowing without `unsafe` transmute.
///
/// # When to use
///
/// Call [`Dispatch3DTyped::dispatch_3d_typed`] from a `match D` arm where D is known to be
/// 3 (e.g. inside `dispatch_linear`). The trait method handles the
/// narrowing internally and routes through [`DispatchByShape`] for the
/// actual shape-based dispatch.
pub trait Dispatch3DTyped<B: Backend, const D: usize> {
    /// Narrow `self` to `&Tensor<B, 3>` and dispatch through the sealed
    /// [`DispatchByShape`] trait method.
    ///
    /// This method is only meaningful when `D == 3`. For other D values,
    /// it `unreachable!()`s — the caller is expected to gate the call
    /// with a `match D` arm.
    ///
    /// The return type is always `Tensor<B, 1>` (the shape-routing always
    /// produces a 1-D tensor of interpolated values), regardless of the
    /// const generic D. This allows the trait method to be called from
    /// [`dispatch_linear`] (which returns `Tensor<B, 1>`) without a
    /// type mismatch.
    fn dispatch_3d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1>;
}

impl<B: Backend, const D: usize> Dispatch3DTyped<B, D> for Tensor<B, D> {
    fn dispatch_3d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        // D is a const generic — the `if D == 3` branch is a compile-time
        // check that the compiler constant-folds. For non-3-D monomorphizations,
        // this entire branch is dead-code eliminated.
        if D == 3 {
            // Narrowing: when D == 3, `self` is `&Tensor<B, 3>`. The sealed
            // `DispatchByShape` trait method requires `&Tensor<B, 3>`, so we
            // need to "narrow" the type. We do this via `clone().reshape(...)`,
            // which is safe and produces a `Tensor<B, 3>` that the trait
            // method can accept.
            //
            // Trade-off vs `unsafe` transmute: the `reshape` allocates a new
            // tensor (heap allocation) and does a memory copy. The `unsafe`
            // alternative would be zero-cost but unsafe. We chose safety
            // here because the allocation is small relative to the
            // interpolation work that follows.
            let shape = self.shape();
            let dims = shape.dims.as_slice();
            let data_3d = self.clone().reshape([dims[0], dims[1], dims[2]]);
            data_3d.dispatch_by_shape(indices, mode)
        } else {
            // Caller is expected to gate with a `match D` arm. Reaching
            // this branch means the caller invoked `dispatch_3d_typed` on
            // a non-3-D tensor, which is a logic error.
            unreachable!("dispatch_3d_typed called with D = {}, expected D = 3", D)
        }
    }
}

/// Per-shape runtime dispatcher for 3-D linear interpolation.
///
/// This is a convenience wrapper around the sealed [`DispatchByShape`]
/// trait method. It matches on `data.shape().dims` at runtime and routes
/// common cube shapes (64³, 128³, 256³, 512³) to the const-generic typed
/// instantiations, with fallback to the generic `dim3::interpolate_3d`.
///
/// See [`DispatchByShape`] for the trait-based implementation.
#[inline]
pub fn dispatch_3d_for_shape<B: Backend>(
    data: &Tensor<B, 3>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    data.dispatch_by_shape(indices, mode)
}

// ══════════════════════════════════════════════════════════════════════
//  N-D type-narrowing wrapper traits (Sprint 359)
// ══════════════════════════════════════════════════════════════════════
//
// Parallel to [`Dispatch3DTyped`], but for the other supported
// dimensions (D=1, D=2, D=4). Each trait bridges `&Tensor<B, D>`
// (generic D) to the sealed [`DispatchByShape`] trait method on
// `&Tensor<B, K>` where K ∈ {1, 2, 4}.

/// Extension trait that bridges `&Tensor<B, D>` (generic D) to the
/// sealed [`DispatchByShape`] trait method on `&Tensor<B, 1>`.
///
/// Implemented for **any** `Tensor<B, D>`, but the `D == 1` branch is
/// the only meaningful one — the other branches `unreachable!()` since
/// the compiler can prove D != 1 at compile time and dead-code
/// eliminates them. This gives us type-safe narrowing without `unsafe`
/// transmute.
pub trait Dispatch1DTyped<B: Backend, const D: usize> {
    /// Narrow `self` to `&Tensor<B, 1>` and dispatch through the sealed
    /// [`DispatchByShape`] trait method.
    fn dispatch_1d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1>;
}

impl<B: Backend, const D: usize> Dispatch1DTyped<B, D> for Tensor<B, D> {
    fn dispatch_1d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        if D == 1 {
            // Safe narrowing via `clone().reshape([dims[0]])` — parallel to
            // [`Dispatch3DTyped`]. See that trait for the full rationale
            // (safer than `unsafe` transmute, allocates one tensor).
            let shape = self.shape();
            let dims = shape.dims.as_slice();
            let data_1d = self.clone().reshape([dims[0]]);
            data_1d.dispatch_by_shape(indices, mode)
        } else {
            unreachable!("dispatch_1d_typed called with D = {}, expected D = 1", D)
        }
    }
}

/// Extension trait that bridges `&Tensor<B, D>` (generic D) to the
/// sealed [`DispatchByShape`] trait method on `&Tensor<B, 2>`. See
/// [`Dispatch1DTyped`] for the full design rationale — this trait is
/// the D=2 equivalent.
pub trait Dispatch2DTyped<B: Backend, const D: usize> {
    /// Narrow `self` to `&Tensor<B, 2>` and dispatch through the sealed
    /// [`DispatchByShape`] trait method.
    fn dispatch_2d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1>;
}

impl<B: Backend, const D: usize> Dispatch2DTyped<B, D> for Tensor<B, D> {
    fn dispatch_2d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        if D == 2 {
            let shape = self.shape();
            let dims = shape.dims.as_slice();
            let data_2d = self.clone().reshape([dims[0], dims[1]]);
            data_2d.dispatch_by_shape(indices, mode)
        } else {
            unreachable!("dispatch_2d_typed called with D = {}, expected D = 2", D)
        }
    }
}

/// Extension trait that bridges `&Tensor<B, D>` (generic D) to the
/// sealed [`DispatchByShape`] trait method on `&Tensor<B, 4>`. See
/// [`Dispatch1DTyped`] for the full design rationale — this trait is
/// the D=4 equivalent.
pub trait Dispatch4DTyped<B: Backend, const D: usize> {
    /// Narrow `self` to `&Tensor<B, 4>` and dispatch through the sealed
    /// [`DispatchByShape`] trait method.
    fn dispatch_4d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1>;
}

impl<B: Backend, const D: usize> Dispatch4DTyped<B, D> for Tensor<B, D> {
    fn dispatch_4d_typed(&self, indices: Tensor<B, 2>, mode: OutOfBoundsMode) -> Tensor<B, 1> {
        if D == 4 {
            let shape = self.shape();
            let dims = shape.dims.as_slice();
            let data_4d = self.clone().reshape([dims[0], dims[1], dims[2], dims[3]]);
            data_4d.dispatch_by_shape(indices, mode)
        } else {
            unreachable!("dispatch_4d_typed called with D = {}, expected D = 4", D)
        }
    }
}

/// Per-shape runtime dispatcher for 1-D linear interpolation.
///
/// Convenience wrapper around the sealed [`DispatchByShape`] trait
/// method for `Tensor<B, 1>`. Matches on `data.shape().dims` and routes
/// common 1-D shapes (64, 128, 256, 512) to the const-generic typed
/// instantiations, with fallback to the generic `dim1::interpolate_1d`.
#[inline]
pub fn dispatch_1d_for_shape<B: Backend>(
    data: &Tensor<B, 1>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    data.dispatch_by_shape(indices, mode)
}

/// Per-shape runtime dispatcher for 2-D linear interpolation.
///
/// Convenience wrapper around the sealed [`DispatchByShape`] trait
/// method for `Tensor<B, 2>`. Matches on `data.shape().dims` and routes
/// common 2-D shapes (64², 128², 256², 512²) to the const-generic typed
/// instantiations, with fallback to the generic `dim2::interpolate_2d`.
#[inline]
pub fn dispatch_2d_for_shape<B: Backend>(
    data: &Tensor<B, 2>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    data.dispatch_by_shape(indices, mode)
}

/// Per-shape runtime dispatcher for 4-D linear interpolation.
///
/// Convenience wrapper around the sealed [`DispatchByShape`] trait
/// method for `Tensor<B, 4>`. Matches on `data.shape().dims` and routes
/// common 4-D shapes (64⁴, 128⁴) to the const-generic typed
/// instantiations, with fallback to the generic `dim4::interpolate_4d`.
#[inline]
pub fn dispatch_4d_for_shape<B: Backend>(
    data: &Tensor<B, 4>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    data.dispatch_by_shape(indices, mode)
}

// ══════════════════════════════════════════════════════════════════════
//  Sealed trait for nearest-neighbor shape-based dispatch (Sprint 358)
// ══════════════════════════════════════════════════════════════════════
//
// Parallel to the linear-dispatch sealed trait [`DispatchByShape`], but
// for nearest-neighbor interpolation. Currently no typed nearest-neighbor
// instantiations exist, so the trait method falls through to the generic
// [`nearest::interpolate_3d`] for all shapes. The trait abstraction is in
// place so that typed nearest-neighbor instantiations (parallel to
// `interpolate_3d_typed`) can be added without changing the public API.
//
// # Why a separate trait (not the same `DispatchByShape`)?
//
// The existing `DispatchByShape` trait is bound to the linear interpolation
// impl block (it calls `dim3::interpolate_3d_typed`). A single trait can't
// dispatch to two different function families from the same method, so we
// use a parallel sealed trait `DispatchNearestByShape` for the nearest-
// neighbor case. The sealing mechanism is identical: a private `Sealed`
// supertrait restricts implementations to `Tensor<B, 3>`.

/// Sealed trait for per-shape 3-D nearest-neighbor interpolation dispatch.
///
/// This trait is sealed: it is only implemented for `Tensor<B, 3>`. External
/// code cannot add new implementations. The trait method
/// [`DispatchNearestByShape::dispatch_nearest_by_shape`] currently falls through to the generic
/// `nearest::interpolate_3d` for all shapes — typed nearest-neighbor
/// instantiations are not yet available. The trait abstraction is in
/// place so they can be added without changing the public API.
///
/// # See also
///
/// - [`DispatchByShape`] — the linear-dispatch equivalent
/// - [`DispatchNearest3DTyped`] — the type-narrowing wrapper trait used
///   by [`dispatch_nearest`] to route `&Tensor<B, D>` through this trait.
pub trait DispatchNearestByShape<B: Backend>: sealed::Sealed {
    /// Dispatch to the appropriate 3-D nearest-neighbor interpolation
    /// implementation based on the runtime shape.
    ///
    /// Currently falls through to the generic `nearest::interpolate_3d`
    /// for all shapes. When typed nearest-neighbor instantiations are
    /// added (parallel to `interpolate_3d_typed::<B, D0, D1, D2>`), this
    /// method will route common cube shapes to the typed paths.
    fn dispatch_nearest_by_shape(
        &self,
        indices: Tensor<B, 2>,
        mode: OutOfBoundsMode,
    ) -> Tensor<B, 1>;
}

impl<B: Backend> DispatchNearestByShape<B> for Tensor<B, 3> {
    fn dispatch_nearest_by_shape(
        &self,
        indices: Tensor<B, 2>,
        mode: OutOfBoundsMode,
    ) -> Tensor<B, 1> {
        // Match the same shapes as the linear-dispatch [`DispatchByShape`]
        // impl so the routing is symmetric. Power-of-2 cubes route to
        // the const-generic typed nearest-neighbor instantiations
        // (Sprint 361 — 351-01-NN-TYPED); other shapes fall through to
        // the generic `nearest::interpolate_3d`.
        let shape = self.shape();
        let dims: &[usize] = shape.dims.as_slice();
        match dims {
            // Power-of-2 cubes (most common registration volumes).
            [64, 64, 64] => nearest::interpolate_nearest_3d_typed::<B, 64, 64, 64>(
                self, indices, mode,
            ),
            [128, 128, 128] => nearest::interpolate_nearest_3d_typed::<B, 128, 128, 128>(
                self, indices, mode,
            ),
            [256, 256, 256] => nearest::interpolate_nearest_3d_typed::<B, 256, 256, 256>(
                self, indices, mode,
            ),
            [512, 512, 512] => nearest::interpolate_nearest_3d_typed::<B, 512, 512, 512>(
                self, indices, mode,
            ),
            // Other common shapes (Sprint 360 — 351-01-SHAPE-LIST) fall
            // through to the generic path until typed variants are added.
            [32, 32, 32]
            | [48, 48, 48]
            | [96, 96, 96]
            | [192, 192, 192]
            | [384, 384, 384]
            | [1024, 1024, 1024]
            | [256, 256, 128]
            | [192, 256, 256] => nearest::interpolate_3d(self, indices, mode),
            _ => nearest::interpolate_3d(self, indices, mode),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Type-narrowing wrapper trait for nearest-neighbor (Sprint 358)
// ══════════════════════════════════════════════════════════════════════
//
// Parallel to [`Dispatch3DTyped`]. Bridges `&Tensor<B, D>` (generic D) to
// the sealed [`DispatchNearestByShape`] trait method on `&Tensor<B, 3>`.

/// Extension trait that bridges `&Tensor<B, D>` (generic D) to the sealed
/// [`DispatchNearestByShape`] trait method on `&Tensor<B, 3>`.
///
/// Implemented for **any** `Tensor<B, D>`, but the `D == 3` branch is the
/// only meaningful one — the other branches `unreachable!()` since the
/// compiler can prove D != 3 at compile time and dead-code eliminates
/// them. This gives us type-safe narrowing without `unsafe` transmute.
///
/// # When to use
///
/// Call [`DispatchNearest3DTyped::dispatch_nearest_3d_typed`] from a `match D` arm where D is
/// known to be 3 (e.g. inside `dispatch_nearest`). The trait method
/// handles the narrowing internally and routes through
/// [`DispatchNearestByShape`] for the actual shape-based dispatch.
pub trait DispatchNearest3DTyped<B: Backend, const D: usize> {
    /// Narrow `self` to `&Tensor<B, 3>` and dispatch through the sealed
    /// [`DispatchNearestByShape`] trait method.
    ///
    /// This method is only meaningful when `D == 3`. For other D values,
    /// it `unreachable!()`s — the caller is expected to gate the call
    /// with a `match D` arm.
    fn dispatch_nearest_3d_typed(
        &self,
        indices: Tensor<B, 2>,
        mode: OutOfBoundsMode,
    ) -> Tensor<B, 1>;
}

impl<B: Backend, const D: usize> DispatchNearest3DTyped<B, D> for Tensor<B, D> {
    fn dispatch_nearest_3d_typed(
        &self,
        indices: Tensor<B, 2>,
        mode: OutOfBoundsMode,
    ) -> Tensor<B, 1> {
        // D is a const generic — the `if D == 3` branch is a compile-time
        // check that the compiler constant-folds. For non-3-D monomorphizations,
        // this entire branch is dead-code eliminated.
        if D == 3 {
            // Safe narrowing via `clone().reshape(...)` — parallel to the
            // linear-dispatch wrapper. See [`Dispatch3DTyped`] for the
            // full rationale.
            let shape = self.shape();
            let dims = shape.dims.as_slice();
            let data_3d = self.clone().reshape([dims[0], dims[1], dims[2]]);
            data_3d.dispatch_nearest_by_shape(indices, mode)
        } else {
            unreachable!(
                "dispatch_nearest_3d_typed called with D = {}, expected D = 3",
                D
            )
        }
    }
}

/// Per-shape runtime dispatcher for 3-D nearest-neighbor interpolation.
///
/// This is a convenience wrapper around the sealed [`DispatchNearestByShape`]
/// trait method. It currently falls through to the generic
/// `nearest::interpolate_3d` for all shapes — typed nearest-neighbor
/// instantiations are not yet available. The wrapper exists so that the
/// public API matches the linear-dispatch path, and so that typed
/// nearest-neighbor instantiations can be added without changing callers.
///
/// See [`DispatchNearestByShape`] for the trait-based implementation.
#[inline]
pub fn dispatch_nearest_3d_for_shape<B: Backend>(
    data: &Tensor<B, 3>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    data.dispatch_nearest_by_shape(indices, mode)
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

#[cfg(test)]
#[path = "tests_dispatch.rs"]
mod tests_dispatch;

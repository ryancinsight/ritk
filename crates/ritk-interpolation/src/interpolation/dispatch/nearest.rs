//! Nearest-neighbor shape dispatch and type-narrowing wrapper traits.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::sealed;
use crate::interpolation::kernel::nearest;
use crate::interpolation::shared::OutOfBoundsMode;

// ════════════════════════════════════════════════════════════════════════
// Sealed trait for nearest-neighbor shape-based dispatch (Sprint 358)
// ════════════════════════════════════════════════════════════════════════
//
// Parallel to the linear-dispatch sealed trait [`DispatchByShape`](super::DispatchByShape), but
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
/// - [`DispatchByShape`](super::DispatchByShape) — the linear-dispatch equivalent
/// - [`DispatchNearest3DTyped`] — the type-narrowing wrapper trait used
///   by [`dispatch_nearest`](super::dispatch_nearest) to route `&Tensor<B, D>` (generic D)
///   through this sealed trait.
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
            [64, 64, 64] => {
                nearest::interpolate_nearest_3d_typed::<B, 64, 64, 64>(self, indices, mode)
            }
            [128, 128, 128] => {
                nearest::interpolate_nearest_3d_typed::<B, 128, 128, 128>(self, indices, mode)
            }
            [256, 256, 256] => {
                nearest::interpolate_nearest_3d_typed::<B, 256, 256, 256>(self, indices, mode)
            }
            [512, 512, 512] => {
                nearest::interpolate_nearest_3d_typed::<B, 512, 512, 512>(self, indices, mode)
            }
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

// ════════════════════════════════════════════════════════════════════════
// Type-narrowing wrapper trait for nearest-neighbor (Sprint 358)
// ════════════════════════════════════════════════════════════════════════
//
// Parallel to [`Dispatch3DTyped`](super::Dispatch3DTyped). Bridges `&Tensor<B, D>` (generic D) to
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
            // linear-dispatch wrapper. See [`Dispatch3DTyped`](super::Dispatch3DTyped) for the
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

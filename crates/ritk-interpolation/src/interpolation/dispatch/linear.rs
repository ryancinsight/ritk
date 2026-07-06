//! Linear-interpolation shape dispatch and type-narrowing wrapper traits.

use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

use super::sealed;
use crate::interpolation::kernel::linear::dim1;
use crate::interpolation::kernel::linear::dim2;
use crate::interpolation::kernel::linear::dim3;
use crate::interpolation::kernel::linear::dim4;
use crate::interpolation::shared::OutOfBoundsMode;

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
///   [`dispatch_linear`](super::dispatch_linear) to route `&Tensor<B, D>` (generic D) through
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
            [256, 256, 128] => dim3::interpolate_3d_typed::<B, 256, 256, 128>(self, indices, mode),
            [192, 256, 256] => dim3::interpolate_3d_typed::<B, 192, 256, 256>(self, indices, mode),
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
// Type-narrowing wrapper trait (Sprint 357)
// ════════════════════════════════════════════════════════════════════════
//
// The challenge: [`dispatch_linear`](super::dispatch_linear) takes `data: &Tensor<B, D>` where D
// is a generic const. When D == 3, we want to call the sealed
// [`DispatchByShape`] trait method, which requires `&Tensor<B, 3>`. The
// type system can't automatically narrow D from generic to 3.
//
// Options considered:
// 1. `unsafe` transmute: zero-cost but unsafe (rejected).
// 2. `clone().reshape(...)`: safe but allocates a new tensor (current).
// 3. **Trait-based narrowing (this design)**: safe, named, documented.
// The `Dispatch3DTyped` trait is implemented for any
// `Tensor<B, D>`. When D == 3 (compile-time check, constant-folded),
// it routes through the sealed trait method. The narrowing is
// encapsulated in a named, testable trait method.

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
    /// [`dispatch_linear`](super::dispatch_linear) (which returns `Tensor<B, 1>`) without a
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

// ════════════════════════════════════════════════════════════════════════
// N-D type-narrowing wrapper traits (Sprint 359)
// ════════════════════════════════════════════════════════════════════════
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

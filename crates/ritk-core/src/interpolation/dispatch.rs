//! Dimension dispatch for interpolators.
//!
//! Routes interpolation calls to the correct dimension-specific implementation
//! based on `const D: usize`. Only `D ∈ {1, 2, 3, 4}` is supported; other
//! values panic at runtime. Because the dispatch is a simple `match` over a
//! `const` parameter, the compiler monomorphizes each branch and dead-code
//! eliminates unreachable arms — achieving the same zero-cost dispatch as the
//! previous sealed-trait approach, but without requiring a `where` bound on
//! callers.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::shared::OutOfBoundsMode;

/// Linear interpolation dispatch.
///
/// Delegates to the dimension-specific implementation for `D`.
/// Panics if `D ∉ {1, 2, 3, 4}`.
#[inline]
pub fn dispatch_linear<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    match D {
        1 => super::kernel::linear::dim1::interpolate_1d(data, indices, mode),
        2 => super::kernel::linear::dim2::interpolate_2d(data, indices, mode),
        3 => super::kernel::linear::dim3::interpolate_3d(data, indices, mode),
        4 => super::kernel::linear::dim4::interpolate_4d(data, indices, mode),
        _ => panic!("Linear interpolation only supports D ∈ {{1, 2, 3, 4}}, got D = {D}"),
    }
}

/// Nearest-neighbor interpolation dispatch.
///
/// Delegates to the dimension-specific implementation for `D`.
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
        3 => super::kernel::nearest::interpolate_3d(data, indices, mode),
        4 => super::kernel::nearest::interpolate_4d(data, indices, mode),
        _ => panic!("Nearest-neighbor interpolation only supports D ∈ {{1, 2, 3, 4}}, got D = {D}"),
    }
}

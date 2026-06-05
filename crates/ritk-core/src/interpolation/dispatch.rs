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

/// Linear interpolation dispatch.
///
/// Delegates to the dimension-specific implementation for `D`.
/// Panics if `D ∉ {1, 2, 3, 4}`.
#[inline]
pub fn dispatch_linear<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    zero_pad: bool,
) -> Tensor<B, 1> {
    match D {
        1 => super::linear::dim1::interpolate_1d(data, indices, zero_pad),
        2 => super::linear::dim2::interpolate_2d(data, indices, zero_pad),
        3 => super::linear::dim3::interpolate_3d(data, indices, zero_pad),
        4 => super::linear::dim4::interpolate_4d(data, indices, zero_pad),
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
    zero_pad: bool,
) -> Tensor<B, 1> {
    match D {
        1 => super::nearest::interpolate_1d(data, indices, zero_pad),
        2 => super::nearest::interpolate_2d(data, indices, zero_pad),
        3 => super::nearest::interpolate_3d(data, indices, zero_pad),
        4 => super::nearest::interpolate_4d(data, indices, zero_pad),
        _ => panic!("Nearest-neighbor interpolation only supports D ∈ {{1, 2, 3, 4}}, got D = {D}"),
    }
}

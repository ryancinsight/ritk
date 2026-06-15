pub mod basis;
pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::transform::Transform;

// Four per-D impl blocks. The const-assert / const-eval where-bound tricks
// for producing a compile-time human-readable diagnostic for D ∉ {1, 2, 3, 4}
// are blocked in this Rust toolchain:
//   * `const _: () = assert!(matches!(D, 1..=4), "...{D}...");` requires
//     unstable `const_panic_fmt` (formatted messages in const panic).
//   * `[(); (D >= 1 && D <= 4) as usize - 1]:` where-bound fails with
//     "cannot perform const operation using 'D'" (const-eval of the
//     comparison + cast is not stabilized in this toolchain).
// The per-D impl blocks below produce a clean trait-coherence error
// ("the trait `Transform<B, 5>` is not implemented for `BSplineTransform<B, 5>`")
// which is already a human-readable compile-time diagnostic. A future
// const-assert companion can be added once the Rust toolchain stabilizes
// the required features. See `backlog.md` §Blocked/Deferred (VAR-375-01b).

impl<B: Backend> Transform<B, 1> for BSplineTransform<B, 1> {
    #[inline]
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        dim1::transform_1d(self, points)
    }
}

impl<B: Backend> Transform<B, 2> for BSplineTransform<B, 2> {
    #[inline]
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        dim2::transform_2d(self, points)
    }
}

impl<B: Backend> Transform<B, 3> for BSplineTransform<B, 3> {
    #[inline]
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        dim3::transform_3d(self, points)
    }
}

impl<B: Backend> Transform<B, 4> for BSplineTransform<B, 4> {
    #[inline]
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        dim4::transform_4d(self, points)
    }
}

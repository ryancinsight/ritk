pub mod basis;
pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::transform::Transform;

// Inherent const assert lives in its own impl block because trait impl
// blocks can only contain members of the trait (the `Transform` trait in
// `ritk-core` does not declare `_SUPPORTED_DIM`). The const is a
// compile-time guard for `D ∈ {1, 2, 3, 4}` — it fires a human-readable
// `assertion failed: ...` diagnostic at the use site for any other `D`.
//
// The const assert uses a static string (no `{D}` formatting) so it works
// in const context under stable Rust — formatting const generic values
// in a const panic is not yet stabilized. The diagnostic for D=5 reads:
//
//     assertion `left == right` failed: BSplineTransform only supports D ∈ {1, 2, 3, 4}
//
// The const is evaluated per-monomorphization (one bool comparison per
// supported D) — negligible cost. The trait impl body forces evaluation
// at every use site via `let _: () = Self::_SUPPORTED_DIM;`.
//
// The match-D body still requires a `_ => unreachable!()` arm because
// Rust's stable exhaustiveness checker does not consult associated
// consts to prove match exhaustiveness on a const parameter. The arm is
// statically unreachable in monomorphized code.
impl<B: Backend, const D: usize> BSplineTransform<B, D> {
    // Compile-time guard: `D ∈ {1, 2, 3, 4}`. Fires a human-readable
    // `assertion failed: ...` diagnostic at the use site for any other D.
    const _SUPPORTED_DIM: () = assert!(
        matches!(D, 1..=4),
        "BSplineTransform only supports D ∈ {{1, 2, 3, 4}}"
    );
}

impl<B: Backend, const D: usize> Transform<B, D> for BSplineTransform<B, D> {
    #[inline]
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // Force monomorphization of the const assert at every use site.
        let _: () = Self::_SUPPORTED_DIM;
        match D {
            1 => dim1::transform_1d(self, points),
            2 => dim2::transform_2d(self, points),
            3 => dim3::transform_3d(self, points),
            4 => dim4::transform_4d(self, points),
            _ => unreachable!("BSplineTransform: D ∈ {{1, 2, 3, 4}}, got D = {D}"),
        }
    }
}

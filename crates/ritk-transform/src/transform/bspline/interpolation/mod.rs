pub mod basis;
pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::BSplineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::transform::Transform;

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

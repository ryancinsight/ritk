pub mod basis;
pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::BSplineTransform;
use crate::transform::Transform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

impl<B: Backend, const D: usize> Transform<B, D> for BSplineTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        match D {
            4 => self.transform_4d(points),
            3 => self.transform_3d(points),
            2 => self.transform_2d(points),
            1 => self.transform_1d(points),
            _ => panic!("BSplineTransform only supports 1D, 2D, 3D and 4D"),
        }
    }
}

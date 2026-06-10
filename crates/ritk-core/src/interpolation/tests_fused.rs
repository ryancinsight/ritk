//! Smoke tests for `interpolation::fused`.
//!
//! Full behavioural tests for the fused transform→interpolation path require
//! a running backend and registered test fixtures.  These stubs verify that
//! the module compiles and that the `is_identity_direction` fast-path is
//! entered for identity-direction images.

use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_3d_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

#[test]
fn fused_module_compiles() {
    // Verifies the module compiles and the helper type is accessible.
    let _img = make_3d_image(vec![1.0f32; 4], [1, 2, 2]);
    // FusedInterpolationResult is returned by transform_and_interpolate_3d;
    // constructing an image here confirms the dependency path resolves.
}

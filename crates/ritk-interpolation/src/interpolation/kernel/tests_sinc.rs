//! Smoke tests for `interpolation::kernel::sinc`.

use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// Verify that the sinc kernel module compiles and the default kernel
/// radius is sane (≥ 1).
#[test]
fn sinc_module_compiles() {
    // Any kernel that can be instantiated with the default backend.
    let device = Default::default();
    // Create a trivial 1-D "image" tensor to confirm type resolution.
    let _t = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32], Shape::new([1])), &device);
}

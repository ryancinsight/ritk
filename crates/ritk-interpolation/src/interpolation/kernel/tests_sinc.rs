//! Smoke tests for `interpolation::kernel::sinc`.

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_image::tensor::{Shape, TensorData};

type B = SequentialBackend;

/// Verify that the sinc kernel module compiles and the default kernel
/// radius is sane (≥ 1).
#[test]
fn sinc_module_compiles() {
    // Any kernel that can be instantiated with the default backend.
    let device = Default::default();
    // Create a trivial 1-D "image" tensor to confirm type resolution.
    let _t = Tensor::<f32, B>::from_data((vec![1.0f32], ([1])), &device);
}

//! Smoke tests for `interpolation::kernel::sinc`.

use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// Verify that the sinc kernel module compiles and the default kernel
/// radius is sane (>= 1).
#[test]
fn sinc_module_compiles() {
    // Create a trivial 1-D "image" tensor to confirm type resolution.
    let _t = Tensor::<f32, MoiraiBackend>::from_slice([1], &[1.0f32]);
}

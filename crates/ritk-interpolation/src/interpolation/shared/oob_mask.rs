//! Out-of-bounds mask computation for multi-dimensional voxel indices.

use coeus_core::Backend;
use coeus_tensor::Tensor;

pub fn compute_oob_mask<B>(indices: &Tensor<f32, B>, shape: &[usize]) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    assert!(!shape.is_empty(), "image dimensionality must be non-zero");
    let idx_shape = indices.shape();
    assert_eq!(idx_shape.len(), 2, "indices must be rank 2");

    let backend = B::default();
    let rank = shape.len();
    let n = idx_shape[0];
    let raw = indices.as_slice();
    let mut out = vec![1.0f32; n];

    for point in 0..n {
        for axis in 0..rank {
            let coord = raw[point * rank + axis];
            if coord.floor() < 0.0 || coord.floor() > (shape[axis].saturating_sub(1)) as f32 {
                out[point] = 0.0;
                break;
            }
        }
    }

    Tensor::<f32, B>::from_slice_on([n], &out, &backend)
}

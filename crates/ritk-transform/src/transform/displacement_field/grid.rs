use super::core::DisplacementField;
use coeus_autograd::{matmul, sub, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

impl<B: Backend + BackendOps<f32>, const D: usize> DisplacementField<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Map physical `[N, D]` points to continuous field indices.
    #[must_use]
    pub fn world_to_index_tensor(&self, points: &Var<f32, B>) -> Var<f32, B> {
        matmul(
            &sub(points, &self.origin_tensor),
            &self.world_to_index_matrix,
        )
    }
}

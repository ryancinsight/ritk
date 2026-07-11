//! Scaling-and-squaring velocity integration.

use coeus_autograd::{add, scalar_mul, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

use super::spatial_transform::SpatialTransformer;
use crate::ModelError;

/// Integrates a stationary velocity field into a displacement field.
#[derive(Debug, Clone)]
pub struct VecInt<B> {
    transformer: SpatialTransformer<B>,
    steps: usize,
}

impl<B> VecInt<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct an integrator with `steps` squaring stages.
    #[must_use]
    pub const fn new(steps: usize) -> Self {
        Self {
            transformer: SpatialTransformer::new(),
            steps,
        }
    }

    /// Integrate a `[batch, 3, depth, height, width]` velocity field.
    pub fn forward(&self, flow: &Var<f32, B>) -> Result<Var<f32, B>, ModelError> {
        let scale = 1.0 / (2.0_f32).powi(self.steps as i32);
        let mut integrated = scalar_mul(flow, scale);
        for _ in 0..self.steps {
            let warped = self.transformer.forward(&integrated, &integrated)?;
            integrated = add(&integrated, &warped);
        }
        Ok(integrated)
    }
}

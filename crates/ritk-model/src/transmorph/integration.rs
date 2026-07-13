//! Stationary-velocity-field integration (scaling and squaring), Coeus-native.
//!
//! Integrates a stationary velocity field into a diffeomorphic displacement
//! field via scaling and squaring: scale the field by `1/2^N`, then compose it
//! with itself `N` times. Composition is a self-warp through
//! [`SpatialTransformer`] (differentiable [`coeus_autograd::grid_sample_3d`]), so
//! the whole integration is differentiable.
//!
//! `φ = exp(v)`, realized as `v ← v + v ∘ (Id + v)` iterated `N` times.

use super::spatial_transform::SpatialTransformer;
use coeus_autograd::{add, scalar_mul, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

/// Velocity-field integrator (scaling and squaring).
#[derive(Debug, Clone, Copy)]
pub struct VecInt {
    stn: SpatialTransformer,
    nsteps: usize,
}

impl VecInt {
    /// Create an integrator with `nsteps` squaring steps (`2^nsteps` sub-steps).
    #[must_use]
    pub fn new(nsteps: usize) -> Self {
        Self {
            stn: SpatialTransformer::new(),
            nsteps,
        }
    }

    /// Integrate velocity field `flow` `[B, 3, D, H, W]` to a displacement field.
    pub fn forward<B>(&self, flow: &Var<f32, B>) -> Var<f32, B>
    where
        B: Backend + BackendOps<f32> + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let scale = 1.0 / 2.0f32.powi(self.nsteps as i32);
        let mut flow = scalar_mul(flow, scale);
        for _ in 0..self.nsteps {
            let warped = self.stn.forward(&flow, &flow);
            flow = add(&flow, &warped);
        }
        flow
    }
}

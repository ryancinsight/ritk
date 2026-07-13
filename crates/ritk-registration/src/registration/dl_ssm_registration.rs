//! Native Coeus SSMMorph registration boundary.

use coeus_autograd::Var;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use ritk_image::native::Image;
use ritk_model::ssmmorph::{SSMMorph, SSMMorphConfig};
use ritk_transform::{StaticDisplacementField, StaticDisplacementFieldTransform};

/// SSMMorph inference over native Coeus images.
pub struct SSMMorphIntegration<B>
where
    B: Backend + BackendOps<f32>,
{
    network: SSMMorph<B>,
}

impl<B> SSMMorphIntegration<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize an SSMMorph inference boundary.
    #[must_use]
    pub fn new(config: &SSMMorphConfig) -> Self {
        Self {
            network: SSMMorph::new(config),
        }
    }

    /// Infer a static displacement transforming `moving` toward `fixed`.
    pub fn register(
        &self,
        fixed: &Image<f32, B, 3>,
        moving: &Image<f32, B, 3>,
    ) -> anyhow::Result<StaticDisplacementFieldTransform<B, 3>> {
        if fixed.shape() != moving.shape() {
            anyhow::bail!(
                "SSMMorph image dimensions differ: fixed={:?}, moving={:?}",
                fixed.shape(),
                moving.shape()
            );
        }
        if !fixed.data().is_contiguous() || !moving.data().is_contiguous() {
            anyhow::bail!("SSMMorph native image tensors must be contiguous for zero-copy reshape");
        }
        let [depth, height, width] = fixed.shape();
        let fixed_var = Var::new(fixed.data().reshape([1, 1, depth, height, width]), false);
        let moving_var = Var::new(moving.data().reshape([1, 1, depth, height, width]), false);
        let output = self.network.forward(&fixed_var, &moving_var)?;
        let components = (0..3)
            .map(|component| {
                output
                    .displacement
                    .tensor
                    .slice(&[
                        (0, 1),
                        (component, component + 1),
                        (0, depth),
                        (0, height),
                        (0, width),
                    ])
                    .reshape([depth, height, width])
            })
            .collect();
        let field = StaticDisplacementField::new(
            components,
            *fixed.origin(),
            *fixed.spacing(),
            *fixed.direction(),
        )?;
        Ok(StaticDisplacementFieldTransform::new(field))
    }
}

/// Diffeomorphic SSMMorph inference boundary.
pub struct DiffeomorphicSSMMorph<B>
where
    B: Backend + BackendOps<f32>,
{
    integration: SSMMorphIntegration<B>,
}

impl<B> DiffeomorphicSSMMorph<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Initialize diffeomorphic SSMMorph inference.
    #[must_use]
    pub fn new(config: &SSMMorphConfig) -> Self {
        Self {
            integration: SSMMorphIntegration::new(config),
        }
    }

    /// Infer a topology-preserving displacement transform.
    pub fn register_diffeomorphic(
        &self,
        fixed: &Image<f32, B, 3>,
        moving: &Image<f32, B, 3>,
    ) -> anyhow::Result<StaticDisplacementFieldTransform<B, 3>> {
        self.integration.register(fixed, moving)
    }
}

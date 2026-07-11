//! SSMMorph registration boundary for RITK images.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor as CoeusTensor;
use ritk_image::tensor::{Backend, Tensor, TensorData};
use ritk_image::Image;
use ritk_interpolation::LinearInterpolator;
use ritk_model::ssmmorph::{SSMMorph, SSMMorphConfig};
use ritk_transform::{StaticDisplacementField, StaticDisplacementFieldTransform};

/// Coeus SSMMorph inference exposed at the current RITK image boundary.
pub struct SSMMorphIntegration<B: Backend> {
    network: SSMMorph<MoiraiBackend>,
    device: B::Device,
}

impl<B: Backend> SSMMorphIntegration<B> {
    /// Initialize an SSMMorph inference boundary.
    #[must_use]
    pub fn new(config: &SSMMorphConfig, device: &B::Device) -> Self {
        Self {
            network: SSMMorph::new(config),
            device: device.clone(),
        }
    }

    /// Infer a displacement transforming `moving` toward `fixed`.
    pub fn register(
        &self,
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
    ) -> anyhow::Result<StaticDisplacementFieldTransform<B, 3>> {
        if fixed.shape() != moving.shape() {
            anyhow::bail!(
                "SSMMorph image dimensions differ: fixed={:?}, moving={:?}",
                fixed.shape(),
                moving.shape()
            );
        }
        let [depth, height, width] = fixed.shape();
        let fixed_values = fixed
            .data()
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("fixed image extraction failed: {error:?}"))?;
        let moving_values = moving
            .data()
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("moving image extraction failed: {error:?}"))?;
        let backend = MoiraiBackend::new();
        let fixed_var = Var::new(
            CoeusTensor::from_slice_on([1, 1, depth, height, width], &fixed_values, &backend),
            false,
        );
        let moving_var = Var::new(
            CoeusTensor::from_slice_on([1, 1, depth, height, width], &moving_values, &backend),
            false,
        );
        let output = self.network.forward(&fixed_var, &moving_var)?;
        let values = output.displacement.tensor.as_slice();
        let voxels = depth * height * width;
        let components = (0..3)
            .map(|component| {
                let start = component * voxels;
                Tensor::<B, 3>::from_data(
                    TensorData::new(
                        values[start..start + voxels].to_vec(),
                        [depth, height, width],
                    ),
                    &self.device,
                )
            })
            .collect();
        let field = StaticDisplacementField::new(
            components,
            *fixed.origin(),
            *fixed.spacing(),
            *fixed.direction(),
        );
        Ok(StaticDisplacementFieldTransform::new(
            field,
            LinearInterpolator::new(),
        ))
    }
}

/// Diffeomorphic SSMMorph inference boundary.
pub struct DiffeomorphicSSMMorph<B: Backend> {
    integration: SSMMorphIntegration<B>,
}

impl<B: Backend> DiffeomorphicSSMMorph<B> {
    /// Initialize diffeomorphic SSMMorph inference.
    #[must_use]
    pub fn new(config: &SSMMorphConfig, device: &B::Device) -> Self {
        Self {
            integration: SSMMorphIntegration::new(config, device),
        }
    }

    /// Infer a topology-preserving displacement transform.
    pub fn register_diffeomorphic(
        &self,
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
    ) -> anyhow::Result<StaticDisplacementFieldTransform<B, 3>> {
        self.integration.register(fixed, moving)
    }
}

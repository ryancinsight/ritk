use coeus_autograd::{add, cat, mean, mul, scalar_div, slice, sub, Parameter, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use coeus_optim::{Adam, Optimizer};
use coeus_tensor::Tensor;
use ritk_model::{
    affine::{AffineNetwork, AffineNetworkConfig, AffineTransform},
    transmorph::{
        spatial_transform::SpatialTransformer, TransMorph, TransMorphConfig, TransformIntegration,
    },
    ModelError,
};

type RegistrationOutput<B> = (Var<f32, B>, Var<f32, B>, Var<f32, B>);
use std::time::Instant;

struct CombinedModel<B>
where
    B: Backend + BackendOps<f32>,
{
    affine: AffineNetwork<B>,
    transmorph: TransMorph<B>,
    affine_transform: AffineTransform<B>,
    spatial_transform: SpatialTransformer<B>,
}

impl<B> CombinedModel<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn new() -> Self {
        Self {
            affine: AffineNetworkConfig {
                channels: [4, 8, 16, 32, 64],
            }
            .init(),
            transmorph: TransMorphConfig {
                in_channels: 2,
                embed_dim: 12,
                out_channels: 3,
                window_size: 4,
                integration: TransformIntegration::Integrated,
                integration_steps: 5,
            }
            .init(),
            affine_transform: AffineTransform::new(),
            spatial_transform: SpatialTransformer::new(),
        }
    }

    fn forward(
        &self,
        moving: &Var<f32, B>,
        fixed: &Var<f32, B>,
    ) -> Result<RegistrationOutput<B>, ModelError> {
        let affine_input = cat(&[moving, fixed], 1);
        let theta = self.affine.forward(&affine_input);
        let affine_moving = self.affine_transform.forward(moving, &theta)?;
        let deformable_input = cat(&[&affine_moving, fixed], 1);
        let flow = self.transmorph.forward(&deformable_input)?.flow;
        let warped = self.spatial_transform.forward(&affine_moving, &flow)?;
        Ok((warped, flow, theta))
    }

    fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut parameters = self
            .affine
            .named_parameters()
            .into_iter()
            .map(|parameter| parameter.with_prefix("affine"))
            .collect::<Vec<_>>();
        parameters.extend(
            self.transmorph
                .named_parameters()
                .into_iter()
                .map(|parameter| parameter.with_prefix("transmorph")),
        );
        parameters
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let affine_count = self.affine.parameters().len();
        self.affine.load_parameters(&parameters[..affine_count]);
        self.transmorph.load_parameters(&parameters[affine_count..]);
    }
}

fn main() {
    let mut model = CombinedModel::<MoiraiBackend>::new();
    let mut optimizer = Adam::new(model.named_parameters(), 1e-4, 0.9, 0.999, 1e-8);
    let shape = [1, 1, 32, 32, 32];

    for epoch in 0..5 {
        let start = Instant::now();
        let fixed = volume(shape, 0);
        let moving = volume(shape, 1);
        let (warped, flow, _) = model
            .forward(&moving, &fixed)
            .expect("example inputs satisfy the combined model contract");
        let difference = sub(&fixed, &warped);
        let similarity = mean(&mul(&difference, &difference));
        let regularization = gradient_loss(&flow);
        let loss = add(&similarity, &regularization);

        loss.backward();
        optimizer.step();
        model.load_parameters(
            &optimizer
                .params
                .iter()
                .map(|parameter| parameter.var.clone())
                .collect::<Vec<_>>(),
        );
        optimizer.zero_grad();

        println!(
            "Epoch {epoch} | Loss: {:.6} (similarity: {:.6}, regularization: {:.6}) | {:?}",
            loss.tensor.as_slice()[0],
            similarity.tensor.as_slice()[0],
            regularization.tensor.as_slice()[0],
            start.elapsed()
        );
    }
}

fn gradient_loss<B>(flow: &Var<f32, B>) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let shape = flow.tensor.shape();
    let (batch, channels, depth, height, width) =
        (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let depth_loss = squared_difference(
        flow,
        &[
            (0, batch),
            (0, channels),
            (1, depth),
            (0, height),
            (0, width),
        ],
        &[
            (0, batch),
            (0, channels),
            (0, depth - 1),
            (0, height),
            (0, width),
        ],
    );
    let height_loss = squared_difference(
        flow,
        &[
            (0, batch),
            (0, channels),
            (0, depth),
            (1, height),
            (0, width),
        ],
        &[
            (0, batch),
            (0, channels),
            (0, depth),
            (0, height - 1),
            (0, width),
        ],
    );
    let width_loss = squared_difference(
        flow,
        &[
            (0, batch),
            (0, channels),
            (0, depth),
            (0, height),
            (1, width),
        ],
        &[
            (0, batch),
            (0, channels),
            (0, depth),
            (0, height),
            (0, width - 1),
        ],
    );
    scalar_div(&add(&add(&depth_loss, &height_loss), &width_loss), 3.0)
}

fn squared_difference<B>(
    flow: &Var<f32, B>,
    left: &[(usize, usize)],
    right: &[(usize, usize)],
) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    let delta = sub(&slice(flow, left), &slice(flow, right));
    mean(&mul(&delta, &delta))
}

fn volume(shape: [usize; 5], offset: usize) -> Var<f32, MoiraiBackend> {
    let [_batch, _channels, depth, height, width] = shape;
    let values = (0..depth)
        .flat_map(|z| {
            (0..height).flat_map(move |y| {
                (0..width).map(move |x| {
                    let dx = x.abs_diff(width / 2 + offset) as f32;
                    let dy = y.abs_diff(height / 2) as f32;
                    let dz = z.abs_diff(depth / 2) as f32;
                    (-(dx * dx + dy * dy + dz * dz) / 64.0).exp()
                })
            })
        })
        .collect::<Vec<_>>();
    Var::new(
        Tensor::from_slice_on(shape, &values, &MoiraiBackend::new()),
        false,
    )
}

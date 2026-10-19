#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]
use coeus_autograd::{cat, mean, pow, sub, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, SequentialBackend};
use coeus_nn::Module;
use coeus_ops::{BackendOps, CpuBackend, OptimizerOps};
use coeus_optim::{Adam, Optimizer};
use coeus_tensor::Tensor;
use ritk_model::affine::{AffineNetwork, AffineNetworkConfig, AffineTransform};
use std::time::Instant;

fn main() {
    println!("Starting Affine Registration Training Example...");
    run_training::<SequentialBackend>();
}

fn run_training<B>()
where
    B: Backend + BackendOps<f32> + CpuBackend + OptimizerOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let config = AffineNetworkConfig::default();
    let mut model: AffineNetwork<f32, B> = config.init::<f32, B>();
    let stn = AffineTransform::new();

    let num_epochs = 5;
    let in_shape = [1, 1, 32, 32, 32];
    let n: usize = in_shape.iter().product();

    let fixed_data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) / 17.0).collect();
    let moving_data: Vec<f32> = (0..n).map(|i| ((i * 3 + 7) % 31) as f32 / 31.0).collect();

    let fixed_tensor = Tensor::from_slice_on(in_shape, &fixed_data, &B::default())
        .expect("example fixed tensor build failed");
    let fixed = Var::new(fixed_tensor, false).expect("example fixed Var build failed");
    let moving_tensor = Tensor::from_slice_on(in_shape, &moving_data, &B::default())
        .expect("example moving tensor build failed");
    let moving = Var::new(moving_tensor, false).expect("example moving Var build failed");

    let params = model.named_parameters();
    let mut opt =
        Adam::new(params, 1e-3, 0.9, 0.999, 1e-8).expect("Adam builds over model parameters");

    println!("Starting training loop ({} epochs)...", num_epochs);

    for epoch in 0..num_epochs {
        let start = Instant::now();
        let input = cat(&[&moving, &fixed], 1).expect("example concat failed");
        let theta = model.forward(&input).expect("affine network forward");
        let warped = stn.forward(&moving, &theta).expect("affine warp forward");

        let diff = sub(&warped, &fixed).expect("example subtract failed");
        let sq = pow(&diff, 2.0).expect("example power failed");
        let loss = mean(&sq).expect("example mean failed");

        let loss_val = loss.tensor.as_slice()[0];

        loss.backward()
            .expect("backward succeeds over the test loss");
        opt.step()
            .expect("Adam step succeeds for finite registration gradients");

        let updated: Vec<Var<f32, B>> = opt.params.iter().map(|p| p.var.clone()).collect();
        model.load_parameters(&updated);

        opt.zero_grad().expect("zero_grad succeeds");

        let duration = start.elapsed();
        println!(
            "Epoch {} | Loss: {:.6} | Time: {:?}",
            epoch, loss_val, duration
        );
    }
}

use coeus_autograd::{cat, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, SequentialBackend};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use ritk_model::transmorph::{TransMorph, TransMorphConfig, TransformIntegration};
use std::time::Instant;

fn main() {
    println!("Initializing TransMorph Registration Example...");
    run_registration::<SequentialBackend>();
}

fn run_registration<B>()
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let config = TransMorphConfig::new(2, 48, 3)
        .with_window_size(2)
        .with_integration(TransformIntegration::Direct);
    let model: TransMorph<B> = config.init();

    let shape = [1, 1, 64, 64, 64];
    println!("Creating dummy volumes with shape: {:?}", shape);

    let n: usize = shape.iter().product();
    let fixed_data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) / 17.0).collect();
    let moving_data: Vec<f32> = (0..n).map(|i| ((i * 3 + 7) % 31) as f32 / 31.0).collect();

    let fixed = Var::new(
        Tensor::from_slice_on(shape, &fixed_data, &B::default()),
        false,
    );
    let moving = Var::new(
        Tensor::from_slice_on(shape, &moving_data, &B::default()),
        false,
    );

    println!("Running registration...");
    let start = Instant::now();
    let input = cat(&[&moving, &fixed], 1);
    let output = model.forward(&input);
    let flow = output.flow;
    let warped = output.warped;

    let duration = start.elapsed();
    println!("Registration took: {:?}", duration);
    println!("Flow shape: {:?}", flow.tensor.shape());
    println!("Warped shape: {:?}", warped.tensor.shape());
    println!("Example finished successfully!");
}

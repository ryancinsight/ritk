use coeus_autograd::{cat, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, SequentialBackend};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use ritk_model::transmorph::{
    spatial_transform::SpatialTransformer, TransMorph, TransMorphConfig, TransformIntegration,
};
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
    let config_reg = TransMorphConfig::new(2, 48, 3)
        .with_window_size(4)
        .with_integration(TransformIntegration::Direct);

    println!("Initializing model...");
    let model_reg: TransMorph<B> = config_reg.init::<B>();
    let st = SpatialTransformer::new();

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

    let input = cat(&vec![&moving, &fixed], 1);
    let output = model_reg.forward(&input);
    let flow = output.flow;

    let duration = start.elapsed();
    println!("Registration took: {:?}", duration);
    println!("Flow shape: {:?}", flow.tensor.shape());

    let dims_low = [1, 1, 16, 16, 16];
    let n_low: usize = dims_low.iter().product();
    let moving_low_data: Vec<f32> = (0..n_low)
        .map(|i| ((i * 3 + 7) % 31) as f32 / 31.0)
        .collect();
    let moving_low = Var::new(
        Tensor::from_slice_on(dims_low, &moving_low_data, &B::default()),
        false,
    );

    println!("Warping at 1/4 resolution (16x16x16)...");
    let warped = st.forward(&moving_low, &flow);

    println!("Warped shape: {:?}", warped.tensor.shape());
    println!("Example finished successfully!");
}

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::{Tensor, Distribution},
};
use ritk_model::transmorph::{TransMorphConfig, TransMorph, SpatialTransformer};
use std::time::Instant;

fn main() {
    // Select backend (Wgpu if available, else NdArray)
    // For this example, we use Wgpu for performance if possible, but fallback logic isn't built-in easily
    // without dynamic dispatch or cfg. We'll use Wgpu as it is the target.
    
    println!("Initializing TransMorph Registration Example...");
    run_registration_wgpu();
}

fn run_registration_wgpu() {
    type Backend = Wgpu;
    let device = WgpuDevice::default();
    
    println!("Using device: {:?}", device);

    // 1. Configuration
    let config_reg = TransMorphConfig {
        in_channels: 2,
        embed_dim: 48, // Standard TransMorph dim
        out_channels: 3,
        window_size: 4, // Small window for example
    };

    println!("Initializing model...");
    let model_reg: TransMorph<Backend> = config_reg.init(&device);
    let st = SpatialTransformer::<Backend>::new();

    // 2. Create Dummy Data (Fixed and Moving Images)
    // Shape: [Batch, Channels, D, H, W]
    // Using small volume for demo: 64x64x64
    let shape = [1, 1, 64, 64, 64];
    println!("Creating dummy volumes with shape: {:?}", shape);
    
    let fixed = Tensor::<Backend, 5>::random(shape, Distribution::Normal(0.5, 0.1), &device);
    let moving = Tensor::<Backend, 5>::random(shape, Distribution::Normal(0.5, 0.1), &device);

    // 3. Forward Pass (Registration)
    println!("Running registration...");
    let start = Instant::now();
    
    // TransMorph takes concatenated input [B, 2, D, H, W]
    let input = Tensor::cat(vec![moving.clone(), fixed.clone()], 1); // [1, 2, 64, 64, 64]
    
    let flow = model_reg.forward(input); // [1, 3, 16, 16, 16] (1/4 resolution)
    
    let duration = start.elapsed();
    println!("Registration took: {:?}", duration);
    println!("Flow shape: {:?}", flow.shape());

    // 4. Warp Moving Image
    // Since flow is 1/4 resolution (16^3), we downsample images to match for this demo.
    // In production, you would upsample the flow to 64^3 using interpolation.
    
    // Downsample by slicing (stride 4)
    // Burn slice uses ranges.
    // We can't easily stride with slice? Slice is continuous.
    // We can use stride in gathering or simple reshape logic if dimensions allow.
    // Actually, we can just create new dummy data at 16^3 for warping demo 
    // or just assume we want to visualize at low res.
    
    let dims_low = [1, 1, 16, 16, 16];
    let _fixed_low = Tensor::<Backend, 5>::random(dims_low, Distribution::Normal(0.5, 0.1), &device);
    let moving_low = Tensor::<Backend, 5>::random(dims_low, Distribution::Normal(0.5, 0.1), &device);
    
    println!("Warping at 1/4 resolution (16x16x16)...");
    let warped = st.forward(moving_low, flow);
    
    println!("Warped shape: {:?}", warped.shape());
    println!("Example finished successfully!");
}

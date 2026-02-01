use burn::{
    backend::{wgpu::{Wgpu, WgpuDevice}, Autodiff},
    tensor::{Tensor, Distribution},
    prelude::Backend,
    optim::{AdamConfig, Optimizer},
};
use ritk_model::{
    transmorph::{TransMorphConfig, TransMorph, SpatialTransformer},
    losses::{LocalNCCLoss, GlobalNCCLoss, GradLoss},
};
use std::time::Instant;

fn main() {
    println!("Starting TransMorph Training Example...");
    run_training();
}

fn run_training() {
    // 1. Setup Backend (Autodiff + Wgpu)
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    
    println!("Using device: {:?}", device);

    // 2. Model Configuration
    let config = TransMorphConfig {
        in_channels: 2, // Moving + Fixed
        embed_dim: 48,
        out_channels: 3,
        window_size: 4,
        integrate: true,
        integration_steps: 5,
    };

    println!("Initializing model...");
    let mut model: TransMorph<MyBackend> = config.init(&device);
    let st = SpatialTransformer::<MyBackend>::new();
    
    // 3. Loss Functions
    // LocalNCCLoss is standard for registration but computationally heavy.
    // We provide GlobalNCCLoss and MSE for faster training/debugging.
    let _ncc_local = LocalNCCLoss::<MyBackend>::new(5, &device);
    let _ncc_global = GlobalNCCLoss::<MyBackend>::new();
    let grad_loss = GradLoss::<MyBackend>::new();
    let lambda_reg = 1.0; // Regularization weight

    // 4. Optimizer
    let optim_config = AdamConfig::new();
    let mut optimizer = optim_config.init();
    
    // 5. Training Loop
    let num_epochs = 5;
    let batch_size = 1;
    // TransMorph with 4 downsampling stages requires input divisible by 16.
    // 64x64x64 is a good standard size for patches.
    let shape = [batch_size, 1, 64, 64, 64]; 
    
    println!("Starting training loop ({} epochs)...", num_epochs);
    
    for epoch in 0..num_epochs {
        let start = Instant::now();
        
        // Generate dummy batch (In real app, load from dataset)
        // Fixed: Random blob
        // Moving: Slightly different random blob
        let fixed = Tensor::<MyBackend, 5>::random(shape, Distribution::Normal(0.5, 0.1), &device);
        let moving = Tensor::<MyBackend, 5>::random(shape, Distribution::Normal(0.55, 0.1), &device); // Offset mean
        
        // Forward Pass
        println!("  Forward pass...");
        // 1. Concatenate
        let input = Tensor::cat(vec![moving.clone(), fixed.clone()], 1);
        
        // 2. Predict Flow
        let flow = model.forward(input); // [B, 3, D/4, H/4, W/4]
        println!("  Flow shape: {:?}", flow.shape());
        
        // 3. Upsample Flow to match image resolution
        // Naive upsampling by repeat for this demo
        let flow_up = upsample_flow(flow.clone(), 4);
        
        // 4. Warp Moving Image
        let warped = st.forward(moving.clone(), flow_up.clone());
        println!("  Warped shape: {:?}", warped.shape());
        
        // 5. Compute Loss
        // Use MSE for stability/speed in example. Switch to NCC for real tasks.
        println!("  Calculating Loss (MSE + Reg)...");
        let loss_sim = (fixed.clone() - warped.clone()).powf_scalar(2.0).mean();
        // let loss_sim = _ncc_global.forward(fixed.clone(), warped.clone());
        
        let loss_reg = grad_loss.forward(flow.clone()); 
        
        let loss = loss_sim.clone() + loss_reg.clone() * lambda_reg;
        
        // Backward Pass
        println!("  Backward pass...");
        let grads = loss.backward();
        
        // Update Weights
        println!("  Optimizer step...");
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optimizer.step(1e-4, model, grads_params);
        
        let duration = start.elapsed();
        
        // Print stats
        println!(
            "Epoch {} | Loss: {:.4} (Sim: {:.4}, Reg: {:.4}) | Time: {:?}",
            epoch,
            loss.into_scalar(),
            loss_sim.into_scalar(),
            loss_reg.into_scalar(),
            duration
        );
    }
    
    println!("Training finished successfully!");
}

fn upsample_flow<B: Backend>(flow: Tensor<B, 5>, scale: usize) -> Tensor<B, 5> {
    let [b, c, d, h, w] = flow.dims();
    // Nearest neighbor upsampling via reshape/repeat
    // Input: [B, C, D, H, W]
    // Output: [B, C, D*s, H*s, W*s]
    
    flow.reshape([b, c, d, 1, h, 1, w, 1])
        .repeat(&[1, 1, 1, scale, 1, scale, 1, scale])
        .reshape([b, c, d * scale, h * scale, w * scale])
}

use burn::{
    backend::{wgpu::{Wgpu, WgpuDevice}, Autodiff},
    tensor::{Tensor, Distribution},
    prelude::Backend,
    optim::{AdamConfig, Optimizer},
    module::Module,
};
use ritk_model::{
    transmorph::{TransMorphConfig, TransMorph, SpatialTransformer},
    affine::{AffineNetwork, AffineNetworkConfig, AffineTransform},
    losses::{LocalNCCLoss, GlobalNCCLoss, GradLoss},
    io::adapter::images_to_batch,
};
use ritk_core::{
    image::Image,
    spatial::{Point3, Spacing3, Direction3},
};
use std::time::Instant;

#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    affine: AffineNetwork<B>,
    transmorph: TransMorph<B>,
    affine_stn: AffineTransform<B>,
    stn: SpatialTransformer<B>,
}

impl<B: Backend> CombinedModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let affine_config = AffineNetworkConfig::default();
        let affine = affine_config.init(device);
        let affine_stn = AffineTransform::new();
        
        let transmorph_config = TransMorphConfig {
            in_channels: 2,
            embed_dim: 48,
            out_channels: 3,
            window_size: 4,
            integrate: true,
            integration_steps: 5,
        };
        let transmorph = transmorph_config.init(device);
        let stn = SpatialTransformer::new();
        
        Self {
            affine,
            transmorph,
            affine_stn,
            stn,
        }
    }
    
    pub fn forward(&self, moving: Tensor<B, 5>, fixed: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 3>) {
        // 1. Affine Registration
        // Concatenate inputs for affine net
        let input_affine = Tensor::cat(vec![moving.clone(), fixed.clone()], 1);
        let affine_theta = self.affine.forward(input_affine);
        
        // Apply affine transform
        let moving_affine = self.affine_stn.forward(moving.clone(), affine_theta.clone());
        
        // 2. Deformable Registration
        // Concatenate affine-registered moving image with fixed image
        let input_transmorph = Tensor::cat(vec![moving_affine.clone(), fixed.clone()], 1);
        let flow = self.transmorph.forward(input_transmorph);
        
        // Flow is now full resolution from TransMorph
        
        // Apply deformable transform to the ALREADY affine-warped image
        let moving_deformed = self.stn.forward(moving_affine, flow.clone());
        
        (moving_deformed, flow, affine_theta.reshape([moving.dims()[0], 3, 4]))
    }
}

fn main() {
    println!("Starting TransMorph + Affine Training Example...");
    run_training();
}

fn run_training() {
    // 1. Setup Backend (Autodiff + Wgpu)
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    
    println!("Using device: {:?}", device);

    println!("Initializing combined model...");
    let mut model: CombinedModel<MyBackend> = CombinedModel::new(&device);
    
    // 3. Loss Functions
    let _ncc_local = LocalNCCLoss::<MyBackend>::new(5, &device);
    let _ncc_global = GlobalNCCLoss::<MyBackend>::new();
    let grad_loss = GradLoss::<MyBackend>::new();
    let lambda_reg = 1.0; 

    // 4. Optimizer
    let optim_config = AdamConfig::new();
    let mut optimizer = optim_config.init();
    
    // 5. Training Loop
    let num_epochs = 5;
    let batch_size = 1;
    let shape = [64, 64, 64]; // Spatial dimensions only for Image creation
    
    println!("Starting training loop ({} epochs)...", num_epochs);
    
    for epoch in 0..num_epochs {
        let start = Instant::now();
        
        // Simulate loading real data using ritk-core Image and ritk-model IO adapter
        let fixed_batch = create_mock_batch::<MyBackend>(batch_size, shape, &device);
        let moving_batch = create_mock_batch::<MyBackend>(batch_size, shape, &device);
        
        // Forward Pass
        println!("  Forward pass...");
        
        let (warped, flow, _affine_theta) = model.forward(moving_batch.clone(), fixed_batch.clone());
        
        println!("  Flow shape: {:?}", flow.shape());
        println!("  Warped shape: {:?}", warped.shape());
        
        // 5. Compute Loss
        println!("  Calculating Loss (MSE + Reg)...");
        let loss_sim = (fixed_batch.clone() - warped.clone()).powf_scalar(2.0).mean();
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

/// Simulates loading a batch of NIfTI files
fn create_mock_batch<B: Backend>(batch_size: usize, shape: [usize; 3], device: &B::Device) -> Tensor<B, 5> {
    let mut images = Vec::new();
    
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    for _ in 0..batch_size {
        // Create random data [D, H, W]
        let data = Tensor::<B, 3>::random(shape, Distribution::Normal(0.5, 0.1), device);
        let image = Image::new(data, origin.clone(), spacing.clone(), direction.clone());
        images.push(image);
    }

    // Convert using adapter
    images_to_batch(images).expect("Failed to create batch")
}

use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;
use ritk_model::transmorph::{TransMorphConfig, TransMorph};

type Backend = NdArray<f32>;

#[test]
fn test_transmorph_forward() {
    let device = Default::default();
    
    // Initialize model with small config for testing speed
    let config = TransMorphConfig {
        in_channels: 1,
        embed_dim: 12, // Small dim
        out_channels: 3,
        window_size: 4,
        integrate: false,
        integration_steps: 7,
    };
    
    let model: TransMorph<Backend> = config.init(&device);
    
    // Create random input: [Batch=1, Channels=1, D=32, H=32, W=32]
    // Must be divisible by 4 (patch_embed) * 2^3 (downsamples) = 32
    // So minimum size is 32.
    let input = Tensor::<Backend, 5>::random([1, 1, 32, 32, 32], Distribution::Normal(0.0, 1.0), &device);
    
    let output = model.forward(input);
    
    // Output should be [1, 3, 32, 32, 32] (full resolution due to final upsampling)
    // The encoder downsamples by 4 initially, then 3 more times by 2 (total 32x downsampling in depth)
    // But the decoder upsamples back to full resolution with final_up (4x4x4 stride 4)
    let dims = output.flow.dims();
    println!("Flow dims: {:?}", dims);
    println!("Warped dims: {:?}", output.warped.dims());
    
    assert_eq!(dims, [1, 3, 32, 32, 32]);
}

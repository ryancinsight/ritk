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
    };
    
    let model: TransMorph<Backend> = config.init(&device);
    
    // Create random input: [Batch=1, Channels=1, D=32, H=32, W=32]
    // Must be divisible by 4 (patch_embed) * 2^3 (downsamples) = 32
    // So minimum size is 32.
    let input = Tensor::<Backend, 5>::random([1, 1, 32, 32, 32], Distribution::Normal(0.0, 1.0), &device);
    
    let output = model.forward(input);
    
    // Output should be [1, 3, 8, 8, 8] (since we downsample by 4 initially and output at that res)
    let dims = output.dims();
    println!("Output dims: {:?}", dims);
    
    assert_eq!(dims, [1, 3, 8, 8, 8]);
}

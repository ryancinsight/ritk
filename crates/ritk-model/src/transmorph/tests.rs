use super::*;
use burn_ndarray::NdArray;
use burn::tensor::Tensor;

#[test]
fn test_transmorph_forward() {
    type B = NdArray;
    let device = Default::default();

    // Config:
    // in_channels = 1
    // embed_dim = 12 (small for speed)
    // window_size = 4
    // image size = 64x64x64 (must be divisible by 32 for downsampling)
    let config = TransMorphConfig {
        in_channels: 1,
        embed_dim: 12,
        out_channels: 3,
        window_size: 4,
        integrate: true,
        integration_steps: 4,
    };

    let model = config.init::<B>(&device);

    // Input: [1, 1, 64, 64, 64]
    let x = Tensor::<B, 5>::random(
        [1, 1, 64, 64, 64],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let output = model.forward(x);

    // Output should be [1, 1, 64, 64, 64] (Warped Image)
    let dims = output.warped.dims();
    assert_eq!(dims, [1, 1, 64, 64, 64]);

    // Flow should be [1, 3, 64, 64, 64]
    assert_eq!(output.flow.dims(), [1, 3, 64, 64, 64]);
}

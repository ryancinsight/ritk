//! End-to-end shape check for the Coeus-native TransMorph through the public API.

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_model::transmorph::config::TransformIntegration;
use ritk_model::transmorph::{TransMorph, TransMorphConfig};

type Backend = SequentialBackend;

#[test]
fn test_transmorph_forward() {
    // Minimum input is 32Â³: patch-4 embedding followed by three stride-2
    // downsamplings requires divisibility by 32.
    let config = TransMorphConfig {
        in_channels: 1,
        embed_dim: 12,
        out_channels: 3,
        window_size: 4,
        integration: TransformIntegration::Direct,
        integration_steps: 4,
    };
    let model: TransMorph<Backend> = config.init();

    let shape = [1usize, 1, 32, 32, 32];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) / 13.0 - 0.5).collect();
    let input = Var::new(
        Tensor::from_slice_on(shape, &data, &SequentialBackend),
        false,
    );

    let output = model.forward(&input);

    assert_eq!(output.flow.tensor.shape(), &[1, 3, 32, 32, 32]);
    assert_eq!(output.warped.tensor.shape(), &[1, 1, 32, 32, 32]);
}

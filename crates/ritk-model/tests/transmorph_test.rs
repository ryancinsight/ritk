use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use ritk_model::transmorph::{TransMorphConfig, TransformIntegration};

#[test]
fn transmorph_forward_restores_full_resolution() {
    let config = TransMorphConfig {
        in_channels: 1,
        embed_dim: 12,
        out_channels: 3,
        window_size: 4,
        integration: TransformIntegration::Direct,
        integration_steps: 4,
    };
    let model = config.init::<MoiraiBackend>();
    let input = Var::new(
        Tensor::zeros_on([1, 1, 32, 32, 32], &MoiraiBackend::new()),
        false,
    );

    let output = model.forward(&input).expect("test graph contract is valid");

    assert_eq!(output.warped.tensor.shape(), &[1, 1, 32, 32, 32]);
    assert_eq!(output.flow.tensor.shape(), &[1, 3, 32, 32, 32]);
    assert!(output
        .flow
        .tensor
        .as_slice()
        .iter()
        .all(|value| value.is_finite()));
}

use super::*;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn sequence_forward_is_finite_and_differentiable() {
    let layer =
        SelectiveStateSpace::<MoiraiBackend>::new(SelectiveStateSpaceConfig::new_with_dims(2, 2));
    let input = Var::new(Tensor::ones_on([1, 3, 2], &MoiraiBackend::new()), true);

    let output = layer.forward(&input).expect("sequence shape is valid");

    assert_eq!(output.tensor.shape(), &[1, 3, 2]);
    assert!(output
        .tensor
        .as_slice()
        .iter()
        .all(|value| value.is_finite()));
    assert!(output.tensor.as_slice().iter().any(|&value| value != 0.0));
    output.backward();
    assert!(
        input.grad().is_some(),
        "selective scan must retain its input gradient"
    );
    assert!(layer
        .parameters()
        .iter()
        .all(|parameter| parameter.grad().is_some()));
}

#[test]
fn volumetric_forward_preserves_spatial_shape() {
    let layer =
        SelectiveStateSpace::<MoiraiBackend>::new(SelectiveStateSpaceConfig::new_with_dims(2, 3));
    let input = Var::new(
        Tensor::ones_on([1, 2, 2, 2, 2], &MoiraiBackend::new()),
        false,
    );

    let output = layer.forward_3d(&input).expect("volume shape is valid");

    assert_eq!(output.tensor.shape(), &[1, 3, 2, 2, 2]);
    assert!(output
        .tensor
        .as_slice()
        .iter()
        .all(|value| value.is_finite()));
}

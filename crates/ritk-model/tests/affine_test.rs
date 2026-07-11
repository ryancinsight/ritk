use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::Module;
use coeus_tensor::Tensor;
use ritk_model::affine::{AffineNetworkConfig, AffineTransform};

#[test]
fn affine_network_forward_produces_finite_parameters() {
    let model = AffineNetworkConfig::default().init::<MoiraiBackend>();
    let input = Var::new(
        Tensor::ones_on([1, 2, 32, 32, 32], &MoiraiBackend::new()),
        false,
    );
    let output = model.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 12]);
    assert!(output
        .tensor
        .as_slice()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn affine_identity_preserves_voxels_exactly() {
    let transformer = AffineTransform::<MoiraiBackend>::new();
    let image = Var::new(
        Tensor::from_slice_on(
            [1, 1, 2, 2, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &MoiraiBackend::new(),
        ),
        true,
    );
    let theta = Var::new(
        Tensor::from_slice_on(
            [1, 12],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &MoiraiBackend::new(),
        ),
        true,
    );

    let output = transformer
        .forward(&image, &theta)
        .expect("identity affine contract is valid");

    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2, 2]);
    assert_eq!(output.tensor.as_slice(), image.tensor.as_slice());
    output.backward();
    assert!(
        image.grad().is_some(),
        "image gradient must remain connected"
    );
    assert!(
        theta.grad().is_some(),
        "affine gradient must remain connected"
    );
}

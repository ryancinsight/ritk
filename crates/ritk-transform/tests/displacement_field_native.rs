use coeus_autograd::{sum, Var};
use coeus_core::MoiraiBackend;
use coeus_nn::Module;
use coeus_tensor::{StateDict, Tensor};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_transform::{
    DisplacementField, DisplacementFieldError, DisplacementFieldTransform,
    DisplacementTransformError,
};

type B = MoiraiBackend;

#[test]
fn planar_constant_field_has_exact_value_and_named_inventory() {
    let backend = MoiraiBackend;
    let field = DisplacementField::new(
        vec![
            Tensor::from_slice_on([2, 2], &[1.0; 4], &backend),
            Tensor::from_slice_on([2, 2], &[0.5; 4], &backend),
        ],
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
    .expect("valid field");
    let transform = DisplacementFieldTransform::<B, 2>::new(field);
    assert_eq!(
        transform
            .named_parameters()
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>(),
        ["field.component.0", "field.component.1"]
    );

    let points = Var::new(
        Tensor::from_slice_on([2, 2], &[0.5, 0.5, 1.0, 0.0], &backend),
        false,
    );
    let transformed = transform
        .transform_points(&points)
        .expect("valid interpolation");
    assert_eq!(transformed.tensor.as_slice(), &[1.5, 1.0, 2.0, 0.5]);
}

#[test]
fn volume_center_gradient_matches_multilinear_weights() {
    let backend = MoiraiBackend;
    let field = DisplacementField::new(
        vec![
            Tensor::from_slice_on([2, 2, 2], &[1.0; 8], &backend),
            Tensor::from_slice_on([2, 2, 2], &[2.0; 8], &backend),
            Tensor::from_slice_on([2, 2, 2], &[3.0; 8], &backend),
        ],
        Point::origin(),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
    .expect("valid field");
    let transform = DisplacementFieldTransform::<B, 3>::new(field);
    let points = Var::new(
        Tensor::from_slice_on([1, 3], &[0.5, 0.5, 0.5], &backend),
        false,
    );
    let transformed = transform
        .transform_points(&points)
        .expect("valid interpolation");
    assert_eq!(transformed.tensor.as_slice(), &[1.5, 2.5, 3.5]);
    sum(&transformed).backward();

    for component in transform.field().components() {
        assert_eq!(
            component.grad().expect("component gradient").as_slice(),
            &[0.125; 8]
        );
    }
}

#[test]
fn bounded_state_archive_round_trips_geometry_and_components() {
    let backend = MoiraiBackend;
    let direction = Direction::from_rows([[0.0, -1.0], [1.0, 0.0]]);
    let field = DisplacementField::new(
        vec![
            Tensor::from_slice_on([1, 2], &[4.0, 5.0], &backend),
            Tensor::from_slice_on([1, 2], &[6.0, 7.0], &backend),
        ],
        Point::new([2.0, 3.0]),
        Spacing::new([0.5, 2.0]),
        direction,
    )
    .expect("valid field");
    let mut bytes = Vec::new();
    field.state_dict().save(&mut bytes).expect("archive state");
    let loaded = StateDict::<f32, B>::load(&mut bytes.as_slice()).expect("validate archive");
    let restored = DisplacementField::<B, 2>::from_state_dict(&loaded).expect("restore field");

    assert_eq!(restored.origin(), &Point::new([2.0, 3.0]));
    assert_eq!(restored.spacing(), &Spacing::new([0.5, 2.0]));
    assert_eq!(restored.direction(), direction);
    assert_eq!(restored.components()[0].tensor.as_slice(), &[4.0, 5.0]);
    assert_eq!(restored.components()[1].tensor.as_slice(), &[6.0, 7.0]);
}

#[test]
fn state_load_rejects_missing_geometry() {
    let state = StateDict::<f32, B>::new();
    assert!(matches!(
        DisplacementField::<B, 2>::from_state_dict(&state),
        Err(DisplacementFieldError::MissingState(name)) if name == "field.origin"
    ));
}

#[test]
fn transform_rejects_wrong_point_dimension() {
    let backend = MoiraiBackend;
    let field = DisplacementField::new(
        vec![
            Tensor::zeros_on([2, 2], &backend),
            Tensor::zeros_on([2, 2], &backend),
        ],
        Point::origin(),
        Spacing::new([1.0; 2]),
        Direction::identity(),
    )
    .expect("valid field");
    let transform = DisplacementFieldTransform::new(field);
    let wrong = Var::new(Tensor::zeros_on([1, 3], &backend), false);
    assert!(matches!(
        transform.transform_points(&wrong),
        Err(DisplacementTransformError::PointShape { dimension: 2, actual })
            if actual == [1, 3]
    ));
}

#[test]
fn constant_field_resampling_preserves_values() {
    let backend = MoiraiBackend;
    let field = DisplacementField::new(
        vec![
            Tensor::from_slice_on([2, 2], &[2.0; 4], &backend),
            Tensor::from_slice_on([2, 2], &[-1.0; 4], &backend),
        ],
        Point::origin(),
        Spacing::new([1.0; 2]),
        Direction::identity(),
    )
    .expect("valid field");
    let resampled = field
        .resample(
            [3, 3],
            Point::origin(),
            Spacing::new([0.5; 2]),
            Direction::identity(),
        )
        .expect("resample field");
    assert_eq!(resampled.components()[0].tensor.as_slice(), &[2.0; 9]);
    assert_eq!(resampled.components()[1].tensor.as_slice(), &[-1.0; 9]);
}

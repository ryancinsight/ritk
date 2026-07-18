use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_transform::{DisplacementField, DisplacementFieldTransform, RigidTransform, Transform};

const ABS_TOL: f32 = 8.0 * f32::EPSILON;

#[test]
fn rigid_translation_planar_displaces_correctly() {
    let backend = SequentialBackend;
    let transform = RigidTransform::<SequentialBackend, 2>::new(
        Tensor::from_slice_on([2], &[1.0, 1.0], &backend),
        Tensor::from_slice_on([1], &[std::f32::consts::FRAC_PI_2], &backend),
        Tensor::zeros_on([2], &backend),
    );
    let point = Tensor::from_slice_on([1, 2], &[1.0, 0.0], &backend);

    let transformed = transform.transform_points(point);

    let actual = transformed.as_slice();
    assert!((actual[0] - 1.0).abs() <= ABS_TOL);
    assert!((actual[1] - 2.0).abs() <= ABS_TOL);
}

#[test]
fn displacement_field_planar_maps_by_offset() {
    let backend = MoiraiBackend;
    let x_tensor = Tensor::from_slice_on([2, 2], &[1.0; 4], &backend);
    let y_tensor = Tensor::from_slice_on([2, 2], &[0.5; 4], &backend);
    let field = DisplacementField::new(
        vec![x_tensor, y_tensor],
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
    .expect("valid planar field");
    let transform = DisplacementFieldTransform::<MoiraiBackend, 2>::new(field);
    let points = Var::new(Tensor::from_slice_on([1, 2], &[0.5, 0.5], &backend), false);

    let transformed = transform
        .transform_points(&points)
        .expect("valid interpolation contract");

    let actual = transformed.tensor.as_slice();
    assert!((actual[0] - 1.5).abs() <= ABS_TOL);
    assert!((actual[1] - 1.0).abs() <= ABS_TOL);
}

use super::*;
use coeus_core::SequentialBackend;

const ROTATION_BOUND: f32 = 8.0 * f32::EPSILON;

#[test]
fn planar_quarter_turn_maps_x_to_y() {
    let backend = SequentialBackend;
    let transform = RigidTransform::<SequentialBackend, 2>::new(
        Tensor::zeros_on([2], &backend),
        Tensor::from_slice_on([1], &[std::f32::consts::FRAC_PI_2], &backend),
        Tensor::zeros_on([2], &backend),
    );
    let point = Tensor::from_slice_on([1, 2], &[1.0, 0.0], &backend);

    let transformed = transform.transform_points(point);

    let values = transformed.as_slice();
    assert!(values[0].abs() <= ROTATION_BOUND);
    assert!((values[1] - 1.0).abs() <= ROTATION_BOUND);
}

#[test]
fn volume_translation_and_z_rotation_compose() {
    let backend = SequentialBackend;
    let transform = RigidTransform::<SequentialBackend, 3>::new(
        Tensor::from_slice_on([3], &[1.0, 2.0, 3.0], &backend),
        Tensor::from_slice_on([3], &[0.0, 0.0, std::f32::consts::FRAC_PI_2], &backend),
        Tensor::zeros_on([3], &backend),
    );
    let point = Tensor::from_slice_on([1, 3], &[1.0, 0.0, 0.0], &backend);

    let transformed = transform.transform_points(point);

    let values = transformed.as_slice();
    assert!((values[0] - 1.0).abs() <= ROTATION_BOUND);
    assert!((values[1] - 3.0).abs() <= ROTATION_BOUND);
    assert!((values[2] - 3.0).abs() <= ROTATION_BOUND);
}

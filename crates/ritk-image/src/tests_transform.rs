use crate::types::Image;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;
type Point3 = Point<3>;
type Spacing3 = Spacing<3>;
type Direction3 = Direction<3>;

#[test]
fn test_physical_to_index_transform() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([5.0, 5.0, 5.0]);
    let index = image.transform_physical_point_to_continuous_index(&point);

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_index_to_physical_transform() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let index = Point3::new([5.0, 5.0, 5.0]);
    let point = image.transform_continuous_index_to_physical_point(&index);

    assert!((point[0] - 5.0).abs() < 1e-6);
    assert!((point[1] - 5.0).abs() < 1e-6);
    assert!((point[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_transform_roundtrip() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let original_point = Point3::new([3.5, 4.5, 5.5]);
    let index = image.transform_physical_point_to_continuous_index(&original_point);
    let transformed_point = image.transform_continuous_index_to_physical_point(&index);

    assert!((original_point[0] - transformed_point[0]).abs() < 1e-6);
    assert!((original_point[1] - transformed_point[1]).abs() < 1e-6);
    assert!((original_point[2] - transformed_point[2]).abs() < 1e-6);
}

#[test]
fn test_non_unit_spacing() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([2.0, 2.0, 2.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([10.0, 10.0, 10.0]);
    let index = image.transform_physical_point_to_continuous_index(&point);

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_non_zero_origin() {
    let backend = B::default();
    let data = Tensor::<f32, B>::zeros_on([10, 10, 10], &backend);
    let origin = Point3::new([10.0, 20.0, 30.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();

    let image = Image::<f32, B, 3>::new(data, origin, spacing, direction)
        .expect("test tensor is rank three");

    let point = Point3::new([15.0, 25.0, 35.0]);
    let index = image.transform_physical_point_to_continuous_index(&point);

    assert!((index[0] - 5.0).abs() < 1e-6);
    assert!((index[1] - 5.0).abs() < 1e-6);
    assert!((index[2] - 5.0).abs() < 1e-6);
}

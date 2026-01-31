
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use nalgebra::{Vector3, Rotation3};
use std::f64::consts::PI;

type Backend = NdArray<f32>;
type Point3 = Point<3>;
type Spacing3 = Spacing<3>;
type Direction3 = Direction<3>;

#[test]
fn test_rotated_image_transform() {
    let device = Default::default();
    let data = Tensor::<Backend, 3>::zeros([10, 10, 10], &device);
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    
    // Rotate 90 degrees around Z axis
    // X -> Y, Y -> -X, Z -> Z
    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
    let direction = Direction(rotation.into_inner());
    
    let image = Image::new(data, origin, spacing, direction);
    
    // Point at (1, 0, 0) in physical space
    // Should map to index:
    // P = O + D * I * S
    // [1, 0, 0] = [0,0,0] + R_z(90) * I
    // [1, 0, 0] = [0, -1, 0; 1, 0, 0; 0, 0, 1] * I
    // Inverse R_z(90) is R_z(-90)
    // [0, 1, 0; -1, 0, 0; 0, 0, 1] * [1, 0, 0] = [0, -1, 0]
    // So index should be (0, -1, 0)
    // Wait, indices are usually positive in image storage, but physical space can be anywhere.
    
    let point = Point3::new([1.0, 0.0, 0.0]);
    let index = image.transform_physical_point_to_continuous_index(&point);
    
    // Check values
    assert!((index[0] - 0.0).abs() < 1e-5, "Expected index[0] to be 0.0, got {}", index[0]);
    assert!((index[1] - (-1.0)).abs() < 1e-5, "Expected index[1] to be -1.0, got {}", index[1]);
    assert!((index[2] - 0.0).abs() < 1e-5, "Expected index[2] to be 0.0, got {}", index[2]);
    
    // Test tensor version
    let points_tensor = Tensor::<Backend, 2>::from_floats([[1.0, 0.0, 0.0]], &device);
    let indices_tensor = image.world_to_index_tensor(points_tensor);
    let indices_data = indices_tensor.into_data();
    let indices = indices_data.as_slice::<f32>().unwrap();
    
    assert!((indices[0] - 0.0).abs() < 1e-5, "Tensor: Expected index[0] to be 0.0, got {}", indices[0]);
    assert!((indices[1] - (-1.0)).abs() < 1e-5, "Tensor: Expected index[1] to be -1.0, got {}", indices[1]);
    assert!((indices[2] - 0.0).abs() < 1e-5, "Tensor: Expected index[2] to be 0.0, got {}", indices[2]);
}

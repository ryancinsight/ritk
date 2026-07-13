//! Image transform tests migrated to the Atlas-native (Coeus) path.
//!
//! Tensor-batch transform (`world_to_index_tensor`) is excluded pending
//! native-Image implementation (ADR 0002).

use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type Backend = SequentialBackend;
type Point3 = Point<3>;
type Spacing3 = Spacing<3>;

fn make_image_3d(
    dims: [usize; 3],
    origin: Point3,
    spacing: Spacing3,
    direction: Direction<3>,
) -> Image<f32, Backend, 3> {
    let n = dims[0] * dims[1] * dims[2];
    Image::from_flat_on(vec![0.0f32; n], dims, origin, spacing, direction, &SequentialBackend)
        .expect("valid image")
}

#[test]
fn test_rotated_image_transform() {
    // Rotate 90 degrees around Z axis: X → Y, Y → -X, Z → Z
    let direction = Direction::from_row_major([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let image = make_image_3d(
        [10, 10, 10],
        Point3::new([0.0, 0.0, 0.0]),
        Spacing3::new([1.0, 1.0, 1.0]),
        direction,
    );

    // Point at (1, 0, 0) physical → index should be (0, -1, 0)
    let point = Point3::new([1.0, 0.0, 0.0]);
    let index = image.transform_physical_point_to_continuous_index(&point);

    assert!(
        (index[0] - 0.0).abs() < 1e-5,
        "Expected index[0] = 0.0, got {}",
        index[0]
    );
    assert!(
        (index[1] - (-1.0)).abs() < 1e-5,
        "Expected index[1] = -1.0, got {}",
        index[1]
    );
    assert!(
        (index[2] - 0.0).abs() < 1e-5,
        "Expected index[2] = 0.0, got {}",
        index[2]
    );
}

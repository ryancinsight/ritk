//! Coordinate mapping tests migrated to the Atlas-native (Coeus) path.
//!
//! Tensor-batch transform (`world_to_index_tensor`) is excluded pending
//! native-Image implementation of `world_to_index_tensor` (ADR 0002).

use coeus_core::SequentialBackend;
use proptest::prelude::*;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type Backend = SequentialBackend;
const D: usize = 3;

fn make_rotation(angle_x: f64, angle_y: f64, angle_z: f64) -> Direction<D> {
    let (cx, sx) = (angle_x.cos(), angle_x.sin());
    let (cy, sy) = (angle_y.cos(), angle_y.sin());
    let (cz, sz) = (angle_z.cos(), angle_z.sin());
    let rz = Direction::from_row_major([cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0]);
    let ry = Direction::from_row_major([cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy]);
    let rx = Direction::from_row_major([1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx]);
    rx * ry * rz
}

fn make_image(
    origin: Point<D>,
    spacing: Spacing<D>,
    direction: Direction<D>,
) -> Image<f32, Backend, D> {
    Image::from_flat_on(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("valid image")
}

proptest! {
    #[test]
    fn physical_index_roundtrip(
        origin in prop::array::uniform3(-100.0f64..100.0),
        spacing in prop::array::uniform3(0.1f64..5.0),
        angles in prop::array::uniform3(-std::f64::consts::PI..std::f64::consts::PI),
        point in prop::array::uniform3(-50.0f64..50.0),
    ) {
        let origin = Point::<D>::new(origin);
        let spacing = Spacing::<D>::new(spacing);
        let direction = make_rotation(angles[0], angles[1], angles[2]);
        let image = make_image(origin, spacing, direction);
        let point = Point::<D>::new(point);

        let index = image.transform_physical_point_to_continuous_index(&point);
        let recovered = image.transform_continuous_index_to_physical_point(&index);

        prop_assert!((point[0] - recovered[0]).abs() < 1e-4, "X mismatch: {} vs {}", point[0], recovered[0]);
        prop_assert!((point[1] - recovered[1]).abs() < 1e-4, "Y mismatch: {} vs {}", point[1], recovered[1]);
        prop_assert!((point[2] - recovered[2]).abs() < 1e-4, "Z mismatch: {} vs {}", point[2], recovered[2]);
    }
}

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
    let cx = angle_x.cos();
    let sx = angle_x.sin();
    let cy = angle_y.cos();
    let sy = angle_y.sin();
    let cz = angle_z.cos();
    let sz = angle_z.sin();

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
    fn test_coordinate_roundtrip(
        ox in -100.0f64..100.0, oy in -100.0f64..100.0, oz in -100.0f64..100.0,
        sx in 0.1f64..5.0, sy in 0.1f64..5.0, sz in 0.1f64..5.0,
        ax in -std::f64::consts::PI..std::f64::consts::PI,
        ay in -std::f64::consts::PI..std::f64::consts::PI,
        az in -std::f64::consts::PI..std::f64::consts::PI,
        px in -50.0f64..50.0, py in -50.0f64..50.0, pz in -50.0f64..50.0
    ) {
        let origin = Point::<D>::new([ox, oy, oz]);
        let spacing = Spacing::<D>::new([sx, sy, sz]);
        let direction = make_rotation(ax, ay, az);
        let image = make_image(origin, spacing, direction);
        let point = Point::<D>::new([px, py, pz]);

        let index = image.transform_physical_point_to_continuous_index(&point);
        let recovered = image.transform_continuous_index_to_physical_point(&index);

        prop_assert!((point[0] - recovered[0]).abs() < 1e-4, "X mismatch: {} vs {}", point[0], recovered[0]);
        prop_assert!((point[1] - recovered[1]).abs() < 1e-4, "Y mismatch: {} vs {}", point[1], recovered[1]);
        prop_assert!((point[2] - recovered[2]).abs() < 1e-4, "Z mismatch: {} vs {}", point[2], recovered[2]);
    }
}

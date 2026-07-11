use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use proptest::prelude::*;
use ritk_core::image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

const D: usize = 3;
type Image3 = Image<f32, MoiraiBackend, D>;

fn make_rotation(angle_x: f64, angle_y: f64, angle_z: f64) -> Direction<D> {
    let (cx, sx) = (angle_x.cos(), angle_x.sin());
    let (cy, sy) = (angle_y.cos(), angle_y.sin());
    let (cz, sz) = (angle_z.cos(), angle_z.sin());
    let rz = Direction::from_row_major([cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0]);
    let ry = Direction::from_row_major([cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy]);
    let rx = Direction::from_row_major([1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx]);
    rx * ry * rz
}

proptest! {
    #[test]
    fn physical_index_roundtrip(
        origin in prop::array::uniform3(-100.0f64..100.0),
        spacing in prop::array::uniform3(0.1f64..5.0),
        angles in prop::array::uniform3(-std::f64::consts::PI..std::f64::consts::PI),
        point in prop::array::uniform3(-50.0f64..50.0),
    ) {
        let image = Image3::from_flat(
            vec![0.0; 8],
            [2, 2, 2],
            Point::new(origin),
            Spacing::new(spacing),
            make_rotation(angles[0], angles[1], angles[2]),
        ).expect("fixture shape and data length agree");
        let point = Point::new(point);
        let index = image.physical_point_to_continuous_index(&point)
            .expect("rotation matrices are invertible");
        let recovered = image.continuous_index_to_physical_point(&index);

        for axis in 0..D {
            // Two dense 3x3 transforms plus scaling and translation stay below
            // gamma_4096 for this well-conditioned orthogonal direction matrix.
            let bound = 4096.0 * f64::EPSILON * (1.0 + point[axis].abs());
            prop_assert!((point[axis] - recovered[axis]).abs() <= bound,
                "axis {axis} mismatch: {} vs {}", point[axis], recovered[axis]);
        }
    }

    #[test]
    fn batched_scalar_mapping_agrees_with_closed_form(
        origin in -10.0f64..10.0,
        spacing in 0.5f64..2.0,
        points in prop::collection::vec(-10.0f64..10.0, 1..32),
    ) {
        let image = Image3::from_flat(
            vec![0.0; 8],
            [2, 2, 2],
            Point::new([origin; D]),
            Spacing::new([spacing; D]),
            Direction::identity(),
        ).expect("fixture shape and data length agree");
        let flat = points
            .iter()
            .flat_map(|coordinate| [*coordinate as f32; D])
            .collect::<Vec<_>>();
        let backend = MoiraiBackend;
        let point_tensor = Tensor::from_slice_on([points.len(), D], &flat, &backend);
        let mapped = image
            .physical_points_to_continuous_indices(&point_tensor, &backend)
            .expect("identity direction and point tensor are valid");

        for (row, coordinate) in points.into_iter().enumerate() {
            let expected = (coordinate as f32 - origin as f32) * (1.0 / spacing) as f32;
            for axis in 0..D {
                let got = f64::from(mapped.as_slice()[row * D + axis]);
                prop_assert_eq!(got, f64::from(expected));
            }
        }
    }
}

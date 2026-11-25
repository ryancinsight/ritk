use super::reading_orientation;
use crate::ui::coordinate_system::orientation::{
    edge_slot, orientation_labels, remap_labels_by_slots,
};
use crate::ui::{RotationSteps, ViewTransform};

#[test]
fn signed_axis_permutations_have_radiological_edges_before_user_controls() {
    let source = [8, 6];
    let centers = [[0., 3.], [8., 3.], [4., 0.], [4., 6.]];
    for [i, j, k] in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for signs in 0..8 {
            let mut direction = [0.; 9];
            for (index, component) in [i, j, k].into_iter().enumerate() {
                direction[component * 3 + index] = if signs & (1 << index) == 0 { 1. } else { -1. };
            }
            for (axis, normal) in [i, j, k].into_iter().enumerate() {
                let base = reading_orientation(axis, &direction, ViewTransform::default());
                let size = base.output_size(source);
                let slots =
                    centers.map(|p| edge_slot(base.source_to_output_coordinates(p, source), size));
                let labels = remap_labels_by_slots(orientation_labels(axis, &direction), slots);
                let expected = [("A", "S"), ("R", "S"), ("R", "A")][normal];
                assert_eq!((labels.left, labels.top), expected);
                for rotation in [
                    RotationSteps::Zero,
                    RotationSteps::Ninety,
                    RotationSteps::OneEighty,
                    RotationSteps::TwoSeventy,
                ] {
                    for (flip_h, flip_v) in
                        [(false, false), (false, true), (true, false), (true, true)]
                    {
                        let user = ViewTransform {
                            flip_h,
                            flip_v,
                            rotation,
                        };
                        let transform = reading_orientation(axis, &direction, user);
                        let p = [2.25, 4.5];
                        let output = user.source_to_output_coordinates(
                            base.source_to_output_coordinates(p, source),
                            size,
                        );
                        assert_eq!(transform.source_to_output_coordinates(p, source), output);
                        assert_eq!(transform.output_to_source_coordinates(output, source), p);
                    }
                }
            }
        }
    }
}

#[test]
fn ambiguous_dominant_directions_preserve_the_user_transform() {
    let c = std::f64::consts::FRAC_1_SQRT_2;
    let direction = [0., c, c, 0., -c, c, 1., 0., 0.];
    let user = ViewTransform::default().rotate_cw();
    assert_eq!(reading_orientation(0, &direction, user), user);
}

#[test]
fn singleton_plane_edges_keep_four_distinct_orientation_labels() {
    // Map the left, right, top and bottom midpoints through a clockwise turn.
    for [width, height] in [[1_u32, 1_u32], [1, 9], [7, 1], [7, 9]] {
        let size = [
            usize::try_from(width).expect("small image width"),
            usize::try_from(height).expect("small image height"),
        ];
        let (w, h) = (f64::from(width), f64::from(height));
        let midpoints = [[0., h / 2.], [w, h / 2.], [w / 2., 0.], [w / 2., h]];
        let transform = ViewTransform::default().rotate_cw();
        let slots = midpoints.map(|point| {
            edge_slot(
                transform.source_to_output_coordinates(point, size),
                transform.output_size(size),
            )
        });
        assert_eq!(slots, [2, 3, 1, 0], "source {size:?}");
    }
}

use super::{pan_from_drag_delta, PanError};
use crate::tools::interaction::ViewportOffset;

#[test]
fn four_directions_update_only_their_axis_from_negative_origins() {
    for (displacement, expected) in [
        ([4.0, 0.0], [-7.0, -7.0]),
        ([-4.0, 0.0], [-15.0, -7.0]),
        ([0.0, 4.0], [-11.0, -3.0]),
        ([0.0, -4.0], [-11.0, -11.0]),
        ([2.5, -1.75], [-8.5, -8.75]),
        ([100.0, 200.0], [89.0, 193.0]),
        ([-100.0, -100.0], [-111.0, -107.0]),
    ] {
        let result = pan_from_drag_delta(
            ViewportOffset::new(-11.0, -7.0),
            ViewportOffset::new(displacement[0], displacement[1]),
        )
        .expect("finite directional displacement");
        assert_eq!(result, ViewportOffset::new(expected[0], expected[1]));
    }
}

#[test]
fn every_nonfinite_component_reports_its_argument() {
    for component in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for origin in [
            ViewportOffset::new(component, 3.0),
            ViewportOffset::new(3.0, component),
        ] {
            assert_eq!(
                pan_from_drag_delta(origin, ViewportOffset::new(2.0, -5.0)),
                Err(PanError::InvalidOrigin)
            );
        }
        for displacement in [
            ViewportOffset::new(component, 3.0),
            ViewportOffset::new(3.0, component),
        ] {
            assert_eq!(
                pan_from_drag_delta(ViewportOffset::new(-2.0, 5.0), displacement),
                Err(PanError::InvalidDisplacement)
            );
        }
    }
}

#[test]
fn finite_overflow_in_either_direction_and_axis_is_rejected() {
    for (origin, displacement) in [
        ([f32::MAX, 4.0], [f32::MAX, 2.0]),
        ([-f32::MAX, 4.0], [-f32::MAX, 2.0]),
        ([4.0, f32::MAX], [2.0, f32::MAX]),
        ([4.0, -f32::MAX], [2.0, -f32::MAX]),
    ] {
        assert_eq!(
            pan_from_drag_delta(
                ViewportOffset::new(origin[0], origin[1]),
                ViewportOffset::new(displacement[0], displacement[1]),
            ),
            Err(PanError::UnrepresentableResult)
        );
    }
}

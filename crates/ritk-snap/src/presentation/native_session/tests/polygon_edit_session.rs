use super::measurement_overlay::axial_panel;
use super::support::session;
use super::*;
use crate::app::SnapApp;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use metis_platform::native::{ModifierState, MouseButton};

/// Eight binary32 ulps cover two image-coordinate endpoints, two physical
/// axes, and the two edge-length operations in this three-point path.
const F32_LENGTH_ROUNDING_ULPS: f64 = 8.0;

fn screen_path_length(
    points: &[(i32, i32)],
    texel_x: f64,
    texel_y: f64,
    row_spacing: f64,
    column_spacing: f64,
) -> f64 {
    points
        .windows(2)
        .map(|pair| {
            let [start, end] = pair else {
                unreachable!("invariant: windows(2) yields exactly two points")
            };
            let row = f64::from(end.1 - start.1) / texel_y * row_spacing;
            let column = f64::from(end.0 - start.0) / texel_x * column_spacing;
            row.hypot(column)
        })
        .sum()
}

#[test]
fn native_polygon_move_delete_and_session_restore_preserve_measurements() {
    let (mut session, root) = session();
    let (pane, _, _, texel_x, texel_y) = axial_panel(&session);
    let (center_x, center_y) = session.viewports[pane].center();
    let first = (center_x - 30, center_y);
    let second = (center_x - 10, center_y);
    let third = (center_x - 20, center_y + 12);
    let other_first = (center_x + 10, center_y);
    let other_second = (center_x + 30, center_y);
    let other_third = (center_x + 20, center_y + 12);
    let button = MouseButton::Left;
    let down = |(x, y)| WindowEvent::PointerDown { x, y, button };
    let up = |(x, y)| WindowEvent::PointerUp { x, y, button };
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x4f,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            down(first),
            up(first),
            down(second),
            up(second),
            down(third),
            up(third),
            down(third),
            WindowEvent::PointerDoubleClick {
                x: third.0,
                y: third.1,
                button,
            },
            up(third),
            down(other_first),
            up(other_first),
            down(other_second),
            up(other_second),
            down(other_third),
            up(other_third),
            down(other_third),
            WindowEvent::PointerDoubleClick {
                x: other_third.0,
                y: other_third.1,
                button,
            },
            up(other_third),
            WindowEvent::KeyDown {
                virtual_key: 0x4d,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("native polygon fixture gestures");
    assert_eq!(session.app.annotations.len(), 2);

    let first_midpoint = ((first.0 + second.0) / 2, first.1);
    let moved_screen = (first.0 + 4, first.1 + 4);
    session
        .handle_events(&[
            down(first_midpoint),
            up(first_midpoint),
            down(first),
            WindowEvent::PointerMove {
                x: moved_screen.0,
                y: moved_screen.1,
            },
            up(moved_screen),
        ])
        .expect("move first polygon vertex");
    let viewport = session.viewports[pane].mapping();
    let screen_points = [moved_screen, second, third];
    let expected_points = screen_points.map(|(x, y)| {
        let point = viewport
            .map(ViewportPoint::new(f64::from(x), f64::from(y)))
            .expect("fixture vertex projects within the image");
        [point.y(), point.x()]
    });
    let view = &session.views[pane];
    let [row, col] = view.frame().display_spacing().values();
    let expected_length = screen_path_length(&screen_points, texel_x, texel_y, row, col);
    let expected_location = Some(AnnotationLocation {
        plane: view.axis,
        slice: view.slice_index,
    });
    // Each endpoint rounding affects the physical delta on two edges.
    // Include the physical image extents to bound that cancellation error.
    let length_bound = f64::from(f32::EPSILON)
        * F32_LENGTH_ROUNDING_ULPS
        * (f64::from(view.frame().width()) * col + f64::from(view.frame().height()) * row);

    let other_midpoint = ((other_first.0 + other_second.0) / 2, other_first.1);
    session
        .handle_events(&[
            down(other_midpoint),
            up(other_midpoint),
            WindowEvent::KeyDown {
                virtual_key: 0x2e,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("delete selected second polygon");
    let [Annotation::OpenPolygon {
        points,
        length_mm,
        location,
    }] = session.app.annotations.as_slice()
    else {
        panic!("deleting the second mark leaves the edited first polygon");
    };
    assert_eq!(points, &expected_points);
    assert_eq!(*location, expected_location);
    assert!((f64::from(*length_mm) - expected_length).abs() <= length_bound);
    let expected = session.app.annotations.clone();

    let session_path = root.path().join("polygon-edit-session.json");
    crate::session::save_to_file(&session.app.session_snapshot(), &session_path)
        .expect("save edited session");
    let snapshot = crate::session::load_from_file(&session_path).expect("load edited session");
    let mut restored = SnapApp::default();
    restored
        .apply_session_snapshot(snapshot)
        .expect("apply edited session");
    assert_eq!(restored.annotations, expected);
}

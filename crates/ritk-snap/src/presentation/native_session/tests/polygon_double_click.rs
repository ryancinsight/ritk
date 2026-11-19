use super::measurement_overlay::axial_panel;
use super::support::{control_center, session};
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use metis_platform::native::MouseButton;

#[test]
fn native_double_click_finishes_polygon_once_in_its_pane() {
    let (mut session, _root) = session();
    let (pane, _, _, sx, sy) = axial_panel(&session);
    let (x, y) = session.viewports[pane].center();
    let end = (x + 3, y + 4);
    let button = MouseButton::Left;
    let down = |x, y| WindowEvent::PointerDown { x, y, button };
    let up = |x, y| WindowEvent::PointerUp { x, y, button };
    let marker = WindowEvent::PointerDoubleClick {
        x: end.0,
        y: end.1,
        button,
    };
    session
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x4f,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            down(x, y),
            up(x, y),
            down(end.0, end.1),
            up(end.0, end.1),
            WindowEvent::KeyDown {
                virtual_key: 0x10,
                repeated: false,
                modifiers: ModifierState::SHIFT,
            },
            down(end.0, end.1),
            marker.clone(),
            up(end.0, end.1),
        ])
        .expect("Shift does not finish the polygon");
    assert!(session.app.annotations.is_empty());

    session
        .handle_events(&[
            WindowEvent::KeyUp {
                virtual_key: 0x10,
                modifiers: ModifierState::NONE,
            },
            down(end.0, end.1),
            marker,
            up(end.0, end.1),
        ])
        .expect("the native marker follows the second press and precedes release");

    let [Annotation::OpenPolygon {
        points,
        location,
        length_mm,
    }] = session.app.annotations.as_slice()
    else {
        panic!("one completed path belongs to the clicked image");
    };
    let view = &session.views[pane];
    let [row, col] = view.frame().display_spacing().values();
    let expected = (3.0 * col / sx).hypot(4.0 * row / sy);
    // Two endpoint roundings across the image in each coordinate contribute
    // at most one binary32 epsilon times the sum of its physical dimensions.
    let bound = f64::from(f32::EPSILON)
        * (f64::from(view.frame().width()) * col + f64::from(view.frame().height()) * row);
    assert_eq!(points.len(), 2);
    assert_eq!(
        *location,
        Some(AnnotationLocation {
            plane: 0,
            slice: view.slice_index
        })
    );
    assert!((f64::from(*length_mm) - expected).abs() <= bound);
    assert!(session.app.tool_state.is_idle());
}

#[test]
fn native_tool_switch_discards_open_polygon_before_a_fresh_path() {
    let (mut session, _root) = session();
    let (pane, _, _, sx, sy) = axial_panel(&session);
    let view = &session.views[pane];
    let source_slice = view.slice_index;
    let source_plane = view.axis;
    let [row, col] = view.frame().display_spacing().values();
    let frame_width = view.frame().width();
    let frame_height = view.frame().height();
    let (x, y) = session.viewports[pane].center();
    let button = MouseButton::Left;
    let down = |(x, y)| WindowEvent::PointerDown { x, y, button };
    let up = |(x, y)| WindowEvent::PointerUp { x, y, button };
    let open_polygon = WindowEvent::KeyDown {
        virtual_key: 0x4f,
        repeated: false,
        modifiers: ModifierState::NONE,
    };
    let abandoned_end = (x + 3, y + 4);

    session
        .handle_events(&[
            open_polygon.clone(),
            down((x, y)),
            up((x, y)),
            down(abandoned_end),
            up(abandoned_end),
        ])
        .expect("enter two points for an unfinished open polygon");
    assert!(session.app.annotations.is_empty());
    assert!(!session.app.tool_state.is_idle());

    let pan_action = WindowAction::SelectTool(crate::tools::kind::ToolKind::Pan);
    let pan_control = control_center(&session, None, pan_action);
    session
        .handle_events(&[down(pan_control), up(pan_control), up(abandoned_end)])
        .expect("switch tools and ignore the abandoned path's later release");
    assert_eq!(session.app.active_tool, crate::tools::kind::ToolKind::Pan);
    assert!(session.app.tool_state.is_idle());
    assert!(session.app.annotations.is_empty());

    let fresh_start = (x + 10, y - 5);
    let fresh_end = (x + 16, y - 7);
    session
        .handle_events(&[
            open_polygon,
            down(fresh_start),
            up(fresh_start),
            down(fresh_end),
            up(fresh_end),
            WindowEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("complete a fresh open polygon after the tool switch");

    let [Annotation::OpenPolygon {
        points,
        location,
        length_mm,
    }] = session.app.annotations.as_slice()
    else {
        panic!("only the fresh open polygon is committed");
    };
    let expected = (6.0 * col / sx).hypot(2.0 * row / sy);
    let bound =
        f64::from(f32::EPSILON) * (f64::from(frame_width) * col + f64::from(frame_height) * row);
    assert_eq!(points.len(), 2);
    assert_eq!(
        *location,
        Some(AnnotationLocation {
            plane: source_plane,
            slice: source_slice,
        })
    );
    assert!((f64::from(*length_mm) - expected).abs() <= bound);
    assert!(session.app.tool_state.is_idle());
}

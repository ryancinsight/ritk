use super::measurement_overlay::axial_panel;
use super::support::session;
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

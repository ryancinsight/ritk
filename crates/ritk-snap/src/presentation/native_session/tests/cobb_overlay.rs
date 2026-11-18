use super::measurement_overlay::{axial_panel, composed, draw_lines};
use super::support::session;
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation, ToolState};
use metis_platform::native::MouseButton;

#[test]
fn native_cobb_records_two_segments_on_their_slice() {
    let (mut session, _root) = session();
    let (pane, _, _, _, _) = axial_panel(&session);
    let (x, y) = session.viewports[pane].center();
    let slice = session.views[pane].slice_index;
    let button = MouseButton::Left;
    let drag = |x, y, end_x, end_y| {
        [
            WindowEvent::PointerDown { x, y, button },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button,
            },
        ]
    };
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x43,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("select Cobb Angle");
    session
        .handle_events(&drag(x, y, x + 8, y))
        .expect("draw first segment");
    assert!(matches!(
        session.app.tool_state,
        ToolState::CobbPending { .. }
    ));
    let yellow = metis_platform::Color::rgba(255, 255, 0, 255);
    assert_eq!(session.framebuffer.get_pixel(x + 4, y), yellow);

    session
        .handle_events(&[WindowEvent::PointerCaptureLost])
        .expect("cancel waiting segment");
    assert!(session.app.tool_state.is_idle());
    assert!(session.app.annotations.is_empty());

    session
        .handle_events(&drag(x, y, x + 8, y))
        .expect("redraw first segment");
    let second = drag(x + 2, y - 4, x + 2, y + 4);
    let next = (slice + 1) % session.views[pane].slice_count;
    session
        .handle_events(&second[..2])
        .expect("start second segment");
    session.app.set_slice_for_axis(0, next);
    session.app.set_slice_for_axis(0, slice);
    session
        .handle_events(&second[2..])
        .expect("release canceled drag");
    assert!(session.app.annotations.is_empty());

    session
        .handle_events(&drag(x, y, x + 8, y))
        .expect("restore first segment");
    session
        .handle_events(&[
            second[0].clone(),
            WindowEvent::PointerUp {
                x: x + 2,
                y: y - 4,
                button,
            },
        ])
        .expect("a click does not consume the pending segment");
    session.handle_events(&second).expect("draw second segment");
    let [Annotation::CobbAngle {
        location,
        angle_deg,
        ..
    }] = session.app.annotations.as_slice()
    else {
        panic!("two perpendicular drags record one Cobb annotation");
    };
    assert_eq!(*angle_deg, 90.0);
    assert_eq!(*location, Some(AnnotationLocation { plane: 0, slice }));
    assert_eq!(
        draw_lines(&composed(&session, None)),
        [((x, y), (x + 8, y)), ((x + 2, y - 4), (x + 2, y + 4))]
    );
    session
        .app
        .set_slice_for_axis(0, (slice + 1) % session.views[pane].slice_count);
    session.refresh_frame().expect("refresh neighboring slice");
    assert!(draw_lines(&composed(&session, None)).is_empty());
}

//! End-to-end native window-control routing tests.

use super::*;

#[test]
fn native_menus_and_toolbar_drive_the_existing_viewer_actions() {
    let (mut session, _root) = session();
    let viewport_area = session
        .window_chrome
        .viewport_area(session.surface_width, session.surface_height)
        .expect("bounded image workspace");
    let preview_top = viewport_area.y + viewport_area.height;
    assert!(session.viewports[0].contains(251.0, f64::from(preview_top - 1)));
    assert!(!session.viewports[0].contains(251.0, f64::from(preview_top)));
    let initial_crosshair = session.app.show_crosshair;
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 200,
                y: 48,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 200,
                y: 48,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("select Pan from the toolbar");
    assert_eq!(session.app.active_tool, crate::tools::kind::ToolKind::Pan);
    assert_eq!(
        session.app.pan_offset,
        crate::tools::interaction::ViewportOffset::new(0.0, 0.0)
    );
    assert_eq!(session.active_view, None);

    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 70,
                y: 16,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 70,
                y: 16,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("open View menu");
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 80,
                y: 72,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 80,
                y: 72,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("toggle linked crosshair from the menu");
    assert_eq!(session.app.show_crosshair, !initial_crosshair);

    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: 70,
                y: 16,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 70,
                y: 16,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: 80,
                y: 100,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 80,
                y: 100,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("toggle cine playback from the menu");
    assert!(session.app.cine.enabled);
}

#[test]
fn pane_gesture_precedes_a_later_toolbar_action_in_the_same_batch() {
    let (mut session, _root) = session();
    session.app.active_tool = crate::tools::kind::ToolKind::WindowLevel;
    let initial_window_level = session.views[0].window_level;
    let initial_pan = session.app.pan_offset;
    let (start_x, start_y) = session.viewports[0].center();
    let end_x = start_x + 32;
    let end_y = start_y + 16;

    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: 200,
                y: 48,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: 200,
                y: 48,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("apply pane gesture before selecting Pan");

    assert_ne!(session.views[0].window_level, initial_window_level);
    assert_eq!(session.app.pan_offset, initial_pan);
    assert_eq!(session.app.active_tool, crate::tools::kind::ToolKind::Pan);
}

#[test]
fn pane_gestures_finish_when_the_pointer_releases_over_window_chrome() {
    let (mut session, _root) = session();
    session.app.active_tool = crate::tools::kind::ToolKind::Pan;
    let (start_x, start_y) = session.viewports[0].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: start_x + 10,
                y: start_y + 10,
            },
            WindowEvent::PointerMove { x: 200, y: 48 },
            WindowEvent::PointerUp {
                x: 200,
                y: 48,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("pane gesture release across chrome boundary");
    assert_ne!(
        session.app.pan_offset,
        crate::tools::interaction::ViewportOffset::new(0.0, 0.0)
    );
    assert!(matches!(
        session.app.tool_state,
        crate::tools::interaction::ToolState::Idle
    ));
}

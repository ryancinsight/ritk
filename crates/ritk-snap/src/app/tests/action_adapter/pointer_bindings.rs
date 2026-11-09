//! Default tool assignment for the non-primary pointer buttons.

use super::super::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent};
use crate::tools::interaction::{Annotation, ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;
use crate::ui::{window_level_from_drag_delta, zoom_from_drag_delta, WINDOW_LEVEL_SENSITIVITY};

use super::{apply_app_event, viewport};

const SOURCE: [usize; 2] = [64, 64];

fn press(app: &mut SnapApp, button: PointerButton, x: f64, y: f64) {
    apply_app_event(
        app,
        &viewport(SOURCE),
        PresentationEvent::PointerDown { x, y, button },
    );
}

fn release(app: &mut SnapApp, button: PointerButton, x: f64, y: f64) {
    apply_app_event(
        app,
        &viewport(SOURCE),
        PresentationEvent::PointerUp { x, y, button },
    );
}

fn move_to(app: &mut SnapApp, x: f64, y: f64) {
    apply_app_event(
        app,
        &viewport(SOURCE),
        PresentationEvent::PointerMove { x, y },
    );
}

fn drag(app: &mut SnapApp, button: PointerButton, from: [f64; 2], to: [f64; 2]) {
    press(app, button, from[0], from[1]);
    move_to(app, to[0], to[1]);
    release(app, button, to[0], to[1]);
}

fn app_with_window() -> SnapApp {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, SOURCE[0], SOURCE[1]]));
    app.active_tool = ToolKind::PointHu;
    app.viewer_state.window_center = Some(40.0);
    app.viewer_state.window_width = Some(400.0);
    app
}

/// The viewer stores window values as `f32`, so a stored value lies within half
/// a unit in the last place, `EPSILON / 2` relative, of the `f64` it came from.
fn assert_window(app: &SnapApp, center: f64, width: f64) {
    for (stored, expected) in [
        (app.viewer_state.window_center, center),
        (app.viewer_state.window_width, width),
    ] {
        let stored = f64::from(stored.expect("the drag sets the window"));
        let bound = f64::from(f32::EPSILON) / 2.0 * expected.abs();
        assert!((stored - expected).abs() <= bound, "{stored} vs {expected}");
    }
}

#[test]
fn middle_drag_adjusts_the_window_without_changing_the_selected_tool() {
    let mut app = app_with_window();
    let (center, width) =
        window_level_from_drag_delta(40.0, 400.0, 12.0, -8.0, WINDOW_LEVEL_SENSITIVITY);

    drag(&mut app, PointerButton::Middle, [20.0, 20.0], [32.0, 12.0]);

    assert_window(&app, center, width);
    assert_eq!(app.active_tool, ToolKind::PointHu);
    assert!(app.tool_state.is_idle());
    assert!(app.pointer.owner.is_none());
}

#[test]
fn right_drag_zooms_from_the_vertical_displacement() {
    let mut app = app_with_window();
    let expected = zoom_from_drag_delta(app.zoom, -16.0);

    drag(&mut app, PointerButton::Right, [20.0, 30.0], [20.0, 14.0]);

    assert_eq!(app.zoom, expected);
    assert!(app.tool_state.is_idle());
}

#[test]
fn back_drag_pans_by_the_pointer_displacement() {
    let mut app = app_with_window();

    drag(&mut app, PointerButton::X1, [10.0, 10.0], [30.0, 0.0]);

    assert_eq!(app.pan_offset, ViewportOffset::new(20.0, -10.0));
}

#[test]
fn forward_drag_records_the_length_between_its_endpoints() {
    let mut app = app_with_window();

    drag(&mut app, PointerButton::X2, [1.0, 1.0], [4.0, 5.0]);

    match app.annotations.as_slice() {
        [Annotation::Length { p1, p2, length_mm }] => {
            assert_eq!(*p1, [1.0, 1.0]);
            assert_eq!(*p2, [5.0, 4.0]);
            assert_eq!(*length_mm, 5.0);
        }
        other => panic!("expected one length annotation, found {other:?}"),
    }
}

#[test]
fn a_second_button_does_not_take_over_a_running_gesture() {
    let mut app = app_with_window();
    let (center, width) =
        window_level_from_drag_delta(40.0, 400.0, 10.0, 4.0, WINDOW_LEVEL_SENSITIVITY);

    press(&mut app, PointerButton::Middle, 20.0, 20.0);
    press(&mut app, PointerButton::Right, 20.0, 20.0);
    release(&mut app, PointerButton::Right, 20.0, 20.0);
    move_to(&mut app, 30.0, 24.0);
    assert!(matches!(app.tool_state, ToolState::WindowLevelDrag { .. }));
    release(&mut app, PointerButton::Middle, 30.0, 24.0);

    assert_window(&app, center, width);
    assert_eq!(app.zoom, SnapApp::default().zoom);
    assert!(app.pointer.owner.is_none());
}

#[test]
fn a_secondary_click_does_not_run_the_selected_tool() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::MeasureLength;

    press(&mut app, PointerButton::Right, 8.0, 8.0);
    release(&mut app, PointerButton::Right, 8.0, 8.0);

    assert!(app.tool_state.is_idle());
    assert!(app.annotations.is_empty());
}

#[test]
fn a_display_gesture_between_anchors_keeps_the_pending_measurement() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::MeasureLength;

    press(&mut app, PointerButton::Left, 1.0, 1.0);
    release(&mut app, PointerButton::Left, 1.0, 1.0);
    drag(&mut app, PointerButton::Right, [20.0, 30.0], [20.0, 14.0]);
    press(&mut app, PointerButton::Left, 4.0, 5.0);
    release(&mut app, PointerButton::Left, 4.0, 5.0);

    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::Length { length_mm, .. }] if *length_mm == 5.0
    ));
    assert!(app.tool_state.is_idle());
}

#[test]
fn a_left_drag_with_the_length_tool_records_a_length() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::MeasureLength;

    drag(&mut app, PointerButton::Left, [1.0, 1.0], [4.0, 5.0]);

    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::Length { length_mm, .. }] if *length_mm == 5.0
    ));
}

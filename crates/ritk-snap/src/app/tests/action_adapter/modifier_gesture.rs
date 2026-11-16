//! Ctrl/Shift-modified left-button gestures for diagnostic reading.
//!
//! RadiAnt readers adjust the display without leaving the measurement tool:
//! Ctrl-left drags window/level and Shift-left drags pan while the selected
//! tool stays active.

use super::super::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::interaction::ViewportOffset;
use crate::tools::kind::ToolKind;
use crate::ui::{window_level_from_drag_delta, WINDOW_LEVEL_SENSITIVITY};

use super::{apply_app_event, viewport};

const SOURCE: [usize; 2] = [64, 64];

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

fn press_modified(
    app: &mut SnapApp,
    button: PointerButton,
    modifiers: PresentationModifiers,
    x: f64,
    y: f64,
) {
    apply_app_event(
        app,
        &viewport(SOURCE),
        PresentationEvent::PointerDown {
            x,
            y,
            button,
            modifiers,
        },
    );
}

fn press(app: &mut SnapApp, button: PointerButton, x: f64, y: f64) {
    press_modified(app, button, PresentationModifiers::NONE, x, y);
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

fn ctrl() -> PresentationModifiers {
    PresentationModifiers::new(true, false, false, false)
}

fn shift() -> PresentationModifiers {
    PresentationModifiers::new(false, true, false, false)
}

#[test]
fn ctrl_left_drag_adjusts_the_window_without_changing_the_selected_tool() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::MeasureLength;
    let (center, width) =
        window_level_from_drag_delta(40.0, 400.0, 12.0, -8.0, WINDOW_LEVEL_SENSITIVITY);

    press_modified(&mut app, PointerButton::Left, ctrl(), 20.0, 20.0);
    move_to(&mut app, 32.0, 12.0);
    release(&mut app, PointerButton::Left, 32.0, 12.0);

    assert_window(&app, center, width);
    assert_eq!(app.active_tool, ToolKind::MeasureLength);
    assert!(app.tool_state.is_idle());
    assert!(app.annotations.is_empty());
}

#[test]
fn shift_left_drag_pans_without_changing_the_selected_tool() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::PointHu;

    press_modified(&mut app, PointerButton::Left, shift(), 10.0, 10.0);
    move_to(&mut app, 30.0, 0.0);
    release(&mut app, PointerButton::Left, 30.0, 0.0);

    assert_eq!(app.pan_offset, ViewportOffset::new(20.0, -10.0));
    assert_eq!(app.active_tool, ToolKind::PointHu);
    assert!(app.tool_state.is_idle());
}

#[test]
fn ctrl_takes_precedence_over_shift_for_a_modified_left_press() {
    let mut app = app_with_window();
    let both = PresentationModifiers::new(true, true, false, false);
    let (center, width) =
        window_level_from_drag_delta(40.0, 400.0, 12.0, -8.0, WINDOW_LEVEL_SENSITIVITY);

    press_modified(&mut app, PointerButton::Left, both, 20.0, 20.0);
    move_to(&mut app, 32.0, 12.0);
    release(&mut app, PointerButton::Left, 32.0, 12.0);

    assert_window(&app, center, width);
    assert_eq!(app.pan_offset, ViewportOffset::new(0.0, 0.0));
}

#[test]
fn a_modified_left_click_places_no_measurement_anchor() {
    let mut app = app_with_window();
    app.active_tool = ToolKind::MeasureLength;

    press(&mut app, PointerButton::Left, 1.0, 1.0);
    release(&mut app, PointerButton::Left, 1.0, 1.0);
    assert!(app.tool_state.is_pending_measurement());
    press_modified(&mut app, PointerButton::Left, ctrl(), 8.0, 8.0);
    release(&mut app, PointerButton::Left, 8.0, 8.0);

    assert!(app.annotations.is_empty());
    assert!(app.tool_state.is_pending_measurement());
}

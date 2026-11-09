//! A held button cannot outlive what its pane shows, and the unpaired presses
//! and releases a host delivers afterwards are ordinary input.

use super::*;
use crate::presentation::native_session::NativeViewerSession;
use crate::presentation::PointerButton;

fn down(x: i32, y: i32, button: MouseButton) -> WindowEvent {
    WindowEvent::PointerDown { x, y, button }
}

fn up(x: i32, y: i32, button: MouseButton) -> WindowEvent {
    WindowEvent::PointerUp { x, y, button }
}

fn key(virtual_key: u32) -> WindowEvent {
    WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    }
}

const TAB: u32 = 0x09;
const ESCAPE: u32 = 0x1b;
const ARROW_RIGHT: u32 = 0x27;
const F4: u32 = 0x73;

fn show_two_panels(viewer: &mut NativeViewerSession, study: &std::path::Path) {
    viewer
        .open_study_path(study)
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two-panel layout"),
        ))
        .expect("select two-panel layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series");
    viewer.refresh_frame().expect("render two series");
}

fn assert_nothing_held(viewer: &NativeViewerSession) {
    assert_eq!(viewer.active_view, None);
    assert!(viewer.captured_buttons.is_empty());
    assert_eq!(viewer.app.pointer.owner, None);
    assert_eq!(viewer.compare_panels[0].app.pointer.owner, None);
}

#[test]
fn activating_another_panel_ends_the_gesture_held_in_the_first() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[down(x, y, MouseButton::Right), key(TAB)])
        .expect("tab while a button is held");
    assert_nothing_held(&viewer);
    assert_eq!(viewer.active_panel, 1);

    viewer
        .handle_events(&[up(other_x, other_y, MouseButton::Right)])
        .expect("the release has no gesture left to end");
    assert_nothing_held(&viewer);
    assert_eq!(viewer.active_panel, 1);
}

#[test]
fn browsing_a_panel_ends_the_gesture_held_on_the_series_it_replaced() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[down(x, y, MouseButton::Right), key(ARROW_RIGHT)])
        .expect("browse while a button is held");
    assert_nothing_held(&viewer);

    viewer
        .handle_events(&[up(x, y, MouseButton::Right)])
        .expect("the replaced panel holds no press");
    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("the button starts a new gesture");
    assert_eq!(
        viewer.compare_panels[0]
            .app
            .pointer
            .owner
            .map(|owner| owner.button),
        Some(PointerButton::Right)
    );
}

#[test]
fn opening_the_series_picker_ends_the_gesture_held_on_a_pane() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer.refresh_frame().expect("render the study");
    let (x, y) = viewer.viewports[0].center();

    viewer
        .handle_events(&[down(x, y, MouseButton::Right), key(F4)])
        .expect("open the picker while a button is held");
    assert!(viewer.window_chrome.multi_series_dialog_is_open());
    assert_eq!(viewer.active_view, None);
    assert!(viewer.captured_buttons.is_empty());
    assert_eq!(viewer.app.pointer.owner, None);

    viewer
        .handle_events(&[up(x, y, MouseButton::Right), key(ESCAPE)])
        .expect("the dialog takes the release and closes");
    assert!(!viewer.window_chrome.multi_series_dialog_is_open());

    viewer
        .handle_events(&[down(x, y, MouseButton::Left)])
        .expect("another button starts a gesture on the pane");
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Left)
    );
}

#[test]
fn a_button_pressed_in_a_dialog_is_released_over_a_pane_without_effect() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer.refresh_frame().expect("render the study");
    let (x, y) = viewer.viewports[0].center();

    viewer.handle_events(&[key(F4)]).expect("open the picker");
    viewer
        .handle_events(&[down(x, y, MouseButton::Right), key(ESCAPE)])
        .expect("the dialog takes the press and closes");
    assert!(!viewer.window_chrome.multi_series_dialog_is_open());

    viewer
        .handle_events(&[up(x, y, MouseButton::Right)])
        .expect("the pane never saw the press");
    assert_eq!(viewer.active_view, None);
    assert_eq!(viewer.app.pointer.owner, None);
}

#[test]
fn a_press_between_panels_reaches_no_panel() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    viewer.active_panel = 0;
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();
    let gap_x = (x..other_x)
        .find(|candidate| {
            viewer
                .viewports
                .iter()
                .all(|viewport| !viewport.contains(f64::from(*candidate), f64::from(y)))
        })
        .expect("the panels are separated");

    viewer
        .handle_events(&[down(gap_x, y, MouseButton::Right)])
        .expect("a press between panels is ordinary input");
    assert_nothing_held(&viewer);

    viewer
        .handle_events(&[up(other_x, other_y, MouseButton::Right)])
        .expect("no panel holds the press");
    assert_eq!(viewer.active_panel, 0);

    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("the first panel never saw the earlier press");
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );
}

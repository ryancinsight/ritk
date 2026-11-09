//! A held button cannot outlive what its pane shows, and the unpaired presses
//! and releases a host delivers afterwards are ordinary input.

use super::*;
use crate::presentation::native_session::routing::{EventRoute, PanelEventRouter, PointerCapture};
use crate::presentation::native_session::NativeViewerSession;
use crate::presentation::{PointerButton, PresentationEvent};

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
    let viewports = viewer.viewports.clone();
    let layout = viewer.workspace_layout;
    let mut router = PanelEventRouter::new(0, PointerCapture::default(), 0);
    let press = PresentationEvent::PointerDown {
        x: f64::from(gap_x),
        y: f64::from(y),
        button: PointerButton::Right,
    };
    assert_eq!(
        router.route(&press, layout, &viewports),
        EventRoute::Discarded
    );
    let zoom = viewer.app.zoom;
    viewer
        .handle_events(&[
            WindowEvent::PointerMove { x, y },
            WindowEvent::PointerMove { x, y: y + 40 },
        ])
        .expect("the panels hold no press to drag");
    assert_eq!(viewer.app.zoom, zoom);

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

#[test]
fn a_picker_opened_from_the_toolbar_leaves_the_pointer_free() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer.refresh_frame().expect("render the study");
    let (x, y) = viewer.viewports[0].center();
    let (picker_x, picker_y) = control_center(
        &viewer,
        None,
        crate::presentation::native_session::window_controls::WindowAction::OpenSeriesPicker,
    );

    viewer
        .handle_events(&[down(picker_x, picker_y, MouseButton::Left)])
        .expect("the toolbar press opens the picker");
    assert!(viewer.window_chrome.multi_series_dialog_is_open());
    viewer
        .handle_events(&[up(picker_x, picker_y, MouseButton::Left), key(ESCAPE)])
        .expect("the dialog takes the release and closes");
    assert!(!viewer.window_chrome.multi_series_dialog_is_open());

    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("the pane takes the next press");
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );
    assert_eq!(viewer.active_view, Some(0));
}

#[test]
fn a_press_of_a_held_button_in_another_panel_ends_the_first_gesture() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("press in the first panel");
    viewer
        .handle_events(&[down(other_x, other_y, MouseButton::Right)])
        .expect("a lost release is ordinary host input");
    assert_eq!(viewer.app.pointer.owner, None);
    assert_eq!(viewer.active_view, Some(1));

    let zoom = viewer.app.zoom;
    viewer
        .handle_events(&[
            up(other_x, other_y, MouseButton::Right),
            WindowEvent::PointerMove { x, y: y - 40 },
        ])
        .expect("release, then hover over the first panel");
    assert_nothing_held(&viewer);
    assert_eq!(viewer.app.zoom, zoom);
}

#[test]
fn replacing_the_study_ends_the_held_gesture() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer.refresh_frame().expect("render the study");
    let (x, y) = viewer.viewports[0].center();

    viewer
        .handle_events(&[down(x, y, MouseButton::Right)])
        .expect("press on the pane");
    viewer
        .open_study_path(study.path())
        .expect("reopen the study");

    assert!(viewer.captured_buttons.is_empty());
    assert_eq!(viewer.active_view, None);
    assert_eq!(viewer.app.pointer.owner, None);
    viewer.refresh_frame().expect("render the replacement");
    viewer
        .handle_events(&[down(x, y, MouseButton::Left)])
        .expect("another button starts a gesture");
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Left)
    );
}

#[test]
fn a_press_between_panels_joins_a_gesture_already_held() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, _) = viewer.viewports[1].center();
    let gap_x = (x..other_x)
        .find(|candidate| {
            viewer
                .viewports
                .iter()
                .all(|viewport| !viewport.contains(f64::from(*candidate), f64::from(y)))
        })
        .expect("the panels are separated");

    viewer
        .handle_events(&[
            down(x, y, MouseButton::Right),
            down(gap_x, y, MouseButton::Middle),
        ])
        .expect("a second button joins the held gesture");

    assert_eq!(viewer.active_view, Some(0));
    assert!(viewer.captured_buttons.contains(PointerButton::Middle));
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );
}

#[test]
fn a_delivered_release_ends_its_gesture_after_another_button_lost_its_release() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();

    viewer
        .handle_events(&[
            down(x, y, MouseButton::Right),
            down(x, y, MouseButton::Middle),
            down(x, y, MouseButton::Middle),
        ])
        .expect("the repeated press replaces a lost release");
    assert_eq!(viewer.active_view, Some(0));
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );

    viewer
        .handle_events(&[up(x, y, MouseButton::Right)])
        .expect("the owner's release is delivered");
    assert_eq!(viewer.app.pointer.owner, None);
    let zoom = viewer.app.zoom;
    viewer
        .handle_events(&[
            up(x, y, MouseButton::Middle),
            WindowEvent::PointerMove { x, y },
            WindowEvent::PointerMove { x, y: y + 40 },
        ])
        .expect("no button remains held");
    assert_nothing_held(&viewer);
    assert_eq!(viewer.app.zoom, zoom);
}

#[test]
fn a_repeated_press_elsewhere_stays_with_the_pane_another_button_holds() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[
            down(x, y, MouseButton::Right),
            down(x, y, MouseButton::Middle),
            down(other_x, other_y, MouseButton::Middle),
        ])
        .expect("the repeated press replaces a lost release");

    assert_eq!(viewer.active_view, Some(0));
    assert_eq!(
        viewer.app.pointer.owner.map(|owner| owner.button),
        Some(PointerButton::Right)
    );
    assert_eq!(viewer.compare_panels[0].app.pointer.owner, None);
}

#[test]
fn a_lost_release_repeated_between_panels_leaves_the_capture_to_the_next_press() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
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
        .handle_events(&[down(x, y, MouseButton::Middle)])
        .expect("begin a gesture in the first panel");
    viewer
        .handle_events(&[
            down(gap_x, y, MouseButton::Middle),
            down(other_x, other_y, MouseButton::Right),
        ])
        .expect("both presses arrive in one batch");

    assert_eq!(viewer.active_view, Some(1));
    assert_eq!(viewer.app.pointer.owner, None);
    assert_eq!(
        viewer.compare_panels[0]
            .app
            .pointer
            .owner
            .map(|owner| owner.button),
        Some(PointerButton::Right)
    );

    viewer
        .handle_events(&[up(other_x, other_y, MouseButton::Right)])
        .expect("the release reaches the second panel");
    assert_nothing_held(&viewer);
}

#[test]
fn browsing_during_a_series_card_drag_abandons_the_drop() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    viewer.active_panel = 1;
    let (card_x, card_y) = series_card_center(&viewer, 3);
    let (x, y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[down(card_x, card_y, MouseButton::Left), key(ARROW_RIGHT)])
        .expect("browse while a series card is held");
    let browsed = viewer.compare_panels[0].series_index;
    assert_eq!(browsed, Some(2), "the key must browse to the next series");

    viewer
        .handle_events(&[up(x, y, MouseButton::Left)])
        .expect("the release has no drop left to perform");
    assert_eq!(viewer.compare_panels[0].series_index, browsed);
    assert_nothing_held(&viewer);
}

//! A pointer release ends a gesture whose press already chose the panel, so it
//! never changes the panel that keyboard events after it reach.

use super::held_gestures::{down, key, show_two_panels, up};
use super::*;
use crate::presentation::native_session::routing::activates_panel;
use crate::presentation::native_session::NativeViewerSession;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};

const TAB: u32 = 0x09;
const END: u32 = 0x23;
const HOME: u32 = 0x24;

/// A two-panel session showing the first series in panel 0 and the last in
/// panel 1. Home acts only on a panel not showing the first series, and End
/// only on one not showing the last, so each key reaches the chrome only when
/// the panel the batch holds active is the one it names.
fn panels_showing_first_and_last_series(
    study: &std::path::Path,
) -> (NativeViewerSession, tempfile::TempDir) {
    let (mut viewer, root) = session();
    show_two_panels(&mut viewer, study);
    viewer
        .assign_series_to_panel(3, 1)
        .expect("show the last series in the second panel");
    viewer.refresh_frame().expect("render both series");
    viewer.active_panel = 0;
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(3));
    (viewer, root)
}

#[test]
fn a_key_after_a_queued_release_acts_on_the_panel_the_batch_activated() {
    let study = four_series_study();
    let (mut viewer, _root) = panels_showing_first_and_last_series(study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[
            down(x, y, MouseButton::Right),
            key(TAB),
            up(other_x, other_y, MouseButton::Right),
            key(HOME),
        ])
        .expect("the batch switches panels, releases the button, then browses");

    assert_eq!(viewer.active_panel, 1);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(0));
}

#[test]
fn a_release_over_another_panel_leaves_the_pressed_panel_active_for_the_next_key() {
    let study = four_series_study();
    let (mut viewer, _root) = panels_showing_first_and_last_series(study.path());
    let (x, y) = viewer.viewports[0].center();
    let (other_x, other_y) = viewer.viewports[1].center();

    viewer
        .handle_events(&[
            down(x, y, MouseButton::Right),
            up(other_x, other_y, MouseButton::Right),
            key(END),
        ])
        .expect("press in the first panel, release over the second, then browse");

    assert_eq!(viewer.active_panel, 0);
    assert_eq!(viewer.primary_series_index, Some(3));
    assert_eq!(viewer.compare_panels[0].series_index, Some(3));
}

#[test]
fn only_pointer_input_that_presses_moves_or_wheels_activates_a_panel() {
    let (x, y) = (10.0, 20.0);
    let button = PointerButton::Right;
    let activating = [
        PresentationEvent::PointerDown { x, y, button },
        PresentationEvent::PointerMove { x, y },
        PresentationEvent::PointerWheel {
            x,
            y,
            delta_x: 0.0,
            delta_y: -120.0,
            modifiers: PresentationModifiers::default(),
        },
    ];
    let inert = [
        PresentationEvent::PointerUp { x, y, button },
        PresentationEvent::PointerCancel { x, y, button },
        PresentationEvent::FocusLost,
    ];

    for event in &activating {
        assert!(activates_panel(event), "{event:?} activates a panel");
    }
    for event in &inert {
        assert!(!activates_panel(event), "{event:?} activates no panel");
    }
}

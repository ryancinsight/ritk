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

/// Open the first and last series in distinct panels to exercise key ownership.
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
fn repeated_series_endpoints_preserve_both_reading_positions() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let arrange = tracing::info_span!("arrange").entered();
    let study = four_series_study();
    let (mut viewer, _root) = panels_showing_first_and_last_series(study.path());
    let repeat = |virtual_key| WindowEvent::KeyDown {
        virtual_key,
        repeated: true,
        modifiers: ModifierState::NONE,
    };
    let first = viewer.app.axis;
    let last = viewer.compare_panels[0].app.axis;
    viewer.app.set_slice_for_axis(first, 1);
    viewer.compare_panels[0].app.set_slice_for_axis(last, 0);
    drop(arrange);
    tracing::info_span!("act").in_scope(|| {
        viewer
            .handle_events(&[key(HOME), repeat(HOME), key(TAB), key(END), repeat(END)])
            .expect("route both endpoint presses and their repeats");
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(viewer.app.axis_slice_info(first).0, 1);
    assert_eq!(viewer.compare_panels[0].app.axis_slice_info(last).0, 0);
    assert_eq!(viewer.primary_series_index, Some(0));
    assert_eq!(viewer.compare_panels[0].series_index, Some(3));
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
        PresentationEvent::PointerDown {
            x,
            y,
            button,
            modifiers: PresentationModifiers::NONE,
        },
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

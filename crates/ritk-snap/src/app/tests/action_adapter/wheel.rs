//! Wheel semantics: zoom modifier, slice navigation, and bounds.

use super::super::test_volume;
use crate::app::action_adapter::{ViewerActionDisposition, ViewerActionError, ViewerInputError};
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};

use super::viewport;

#[test]
fn wheel_actions_preserve_zoom_and_slice_navigation_semantics() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);

    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: 120.0,
            modifiers: PresentationModifiers::new(true, false, false, false),
        }],
        Some(&viewport),
    )
    .expect("Ctrl+wheel zoom is supported");
    assert!(app.zoom > 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
    let zoomed = app.zoom;

    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: -120.0,
            modifiers: PresentationModifiers::NONE,
        }],
        Some(&viewport),
    )
    .expect("plain wheel slice navigation is supported");
    assert_eq!(app.viewer_state.slice_index, 1);
    assert_eq!(app.zoom, zoomed);
}

#[test]
fn wheel_actions_use_meta_as_the_platform_zoom_modifier() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    app.apply_presentation_events(
        &[PresentationEvent::PointerWheel {
            x: 2.0,
            y: 2.0,
            delta_x: 0.0,
            delta_y: 120.0,
            modifiers: PresentationModifiers::new(false, false, false, true),
        }],
        Some(&viewport),
    )
    .expect("Meta+wheel zoom is supported");
    assert!(app.zoom > 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn horizontal_wheel_does_not_change_viewer_state() {
    for modifiers in [
        PresentationModifiers::new(true, false, false, false),
        PresentationModifiers::new(false, false, false, true),
    ] {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 4, 3]));
        app.status_message = "steady".to_owned();
        let viewport = viewport([4, 4]);
        let before = (
            app.zoom,
            app.viewer_state.slice_index,
            app.status_message.clone(),
        );

        let disposition = app
            .apply_presentation_events(
                &[PresentationEvent::PointerWheel {
                    x: 2.0,
                    y: 2.0,
                    delta_x: 120.0,
                    delta_y: 0.0,
                    modifiers,
                }],
                Some(&viewport),
            )
            .expect("horizontal wheel is a supported no-op");

        assert_eq!(
            disposition,
            ViewerActionDisposition::Continue { repaint: false }
        );
        assert_eq!(
            (app.zoom, app.viewer_state.slice_index, app.status_message),
            before
        );
    }
}

#[test]
fn wheel_actions_ignore_events_outside_the_viewport() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    let disposition = app
        .apply_presentation_events(
            &[PresentationEvent::PointerWheel {
                x: 20.0,
                y: 20.0,
                delta_x: 0.0,
                delta_y: 120.0,
                modifiers: PresentationModifiers::new(true, false, false, false),
            }],
            Some(&viewport),
        )
        .expect("outside wheel is ignored");
    assert_eq!(
        disposition,
        ViewerActionDisposition::Continue { repaint: false }
    );
    assert_eq!(app.zoom, 1.0);
    assert_eq!(app.viewer_state.slice_index, 0);
}

#[test]
fn out_of_range_wheel_keeps_the_dispatcher_transactional() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 4, 3]));
    let viewport = viewport([4, 4]);
    let error = app
        .apply_presentation_events(
            &[
                PresentationEvent::PointerDown {
                    x: 2.0,
                    y: 2.0,
                    button: PointerButton::Left,
                },
                PresentationEvent::PointerWheel {
                    x: 2.0,
                    y: 2.0,
                    delta_x: 0.0,
                    delta_y: f64::MAX,
                    modifiers: PresentationModifiers::new(true, false, false, false),
                },
            ],
            Some(&viewport),
        )
        .expect_err("out-of-range zoom delta");
    assert!(matches!(
        error,
        ViewerInputError::Action(ViewerActionError::WheelDeltaOutOfRange { .. })
    ));
    app.apply_presentation_events(
        &[PresentationEvent::PointerUp {
            x: 2.0,
            y: 2.0,
            button: PointerButton::Left,
        }],
        Some(&viewport),
    )
    .expect_err("rejected batch must not press the pointer");
    assert_eq!(app.zoom, 1.0);
}

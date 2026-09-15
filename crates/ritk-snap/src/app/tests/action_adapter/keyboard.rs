use super::*;
use crate::app::action_adapter::{
    VIRTUAL_KEY_CINE_FPS_DOWN, VIRTUAL_KEY_CINE_FPS_UP, VIRTUAL_KEY_CINE_TOGGLE,
};

#[test]
fn space_toggles_cine_once_and_ignores_key_repeat() {
    let mut app = SnapApp::default();
    app.loaded = Some(super::super::test_volume([3, 4, 5]));

    let started = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_TOGGLE,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }],
            None,
        )
        .expect("space starts cine");
    assert_eq!(
        started,
        crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: true }
    );
    assert!(app.cine.enabled);

    let repeated = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_TOGGLE,
                repeated: true,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }],
            None,
        )
        .expect("repeated space is ignored");
    assert_eq!(
        repeated,
        crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: false }
    );
    assert!(app.cine.enabled);
}

#[test]
fn cine_rate_keys_change_bounded_rate_and_ignore_repeats() {
    let mut app = SnapApp::default();
    app.loaded = Some(super::super::test_volume([3, 4, 5]));

    let increased = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_FPS_UP,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }],
            None,
        )
        .expect("increase cine rate");
    assert_eq!(
        increased,
        crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: true }
    );
    assert_eq!(app.cine.fps, 13.0);
    assert_eq!(app.status_message, "Cine playback rate: 13 FPS.");

    let repeated = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_FPS_UP,
                repeated: true,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }],
            None,
        )
        .expect("repeated increase is ignored");
    assert_eq!(
        repeated,
        crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: false }
    );
    assert_eq!(app.cine.fps, 13.0);

    app.cine.set_fps(1.0);
    let unchanged = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_FPS_DOWN,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }],
            None,
        )
        .expect("minimum rate remains valid");
    assert_eq!(
        unchanged,
        crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: false }
    );
    assert_eq!(app.cine.fps, 1.0);
}

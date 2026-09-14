use super::*;
use crate::app::action_adapter::VIRTUAL_KEY_CINE_TOGGLE;

#[test]
fn space_toggles_cine_once_and_ignores_key_repeat() {
    let mut app = SnapApp::default();
    app.loaded = Some(super::super::test_volume([3, 4, 5]));

    let started = app
        .apply_presentation_events(
            &[PresentationEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_CINE_TOGGLE,
                repeated: false,
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

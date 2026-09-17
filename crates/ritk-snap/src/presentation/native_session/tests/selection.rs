//! Native selector/recovery tests for the Métis session.

use super::session;
use crate::dicom::loader::tests::fixtures;
use metis_platform::native::{ModifierState, NativeApplication, NativeFlow, WindowEvent};

#[test]
fn native_session_reopen_enters_series_selection_and_loads_exact_choice() {
    let (mut session, _initial_root) = session();
    let replacement_root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(replacement_root.path(), "CT", fixtures::SERIES_UID)
        .expect("write primary series");
    fixtures::write_study(replacement_root.path(), "MR", "2.25.20260905002")
        .expect("write secondary series");

    session
        .open_study_path(replacement_root.path())
        .expect("discover replacement series");
    assert!(session.selection.is_some());
    assert_eq!(
        session
            .app
            .loaded
            .as_ref()
            .expect("current study remains loaded")
            .modality
            .as_deref(),
        Some("CT")
    );

    let moved = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x28,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("move series selection");
    assert_eq!(moved, NativeFlow::Continue { repaint: true });
    assert_eq!(session.selection.as_ref().expect("selection").selected(), 1);

    let loaded = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("load selected series");
    assert_eq!(loaded, NativeFlow::Continue { repaint: true });
    assert!(session.selection.is_none());
    assert_eq!(
        session
            .app
            .loaded
            .as_ref()
            .expect("selected study")
            .metadata
            .as_ref()
            .expect("selected metadata")
            .series_instance_uid
            .as_deref(),
        Some("2.25.20260905002")
    );
}

#[test]
fn native_session_selection_cancel_preserves_current_frame() {
    let (mut session, _initial_root) = session();
    let replacement_root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(replacement_root.path(), "CT", fixtures::SERIES_UID)
        .expect("write primary series");
    fixtures::write_study(replacement_root.path(), "MR", "2.25.20260905002")
        .expect("write secondary series");
    session
        .open_study_path(replacement_root.path())
        .expect("discover replacement series");
    let previous_shape = session.app.loaded.as_ref().expect("current study").shape;
    let previous_frame = session.framebuffer.clone();

    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("cancel series selection");

    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.selection.is_none());
    assert_eq!(
        session.app.loaded.as_ref().expect("current study").shape,
        previous_shape
    );
    assert_eq!(
        session.app.status_message,
        "DICOM series selection canceled; current study remains displayed."
    );
    assert_eq!(session.framebuffer.pixels(), previous_frame.pixels());
}

#[test]
fn native_session_selection_decode_failure_keeps_selector_and_viewer_alive() {
    let (mut session, _initial_root) = session();
    let replacement_root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(replacement_root.path(), "CT", fixtures::SERIES_UID)
        .expect("write primary series");
    fixtures::write_study(replacement_root.path(), "MR", "2.25.20260905002")
        .expect("write secondary series");
    session
        .open_study_path(replacement_root.path())
        .expect("discover replacement series");
    let previous_shape = session.app.loaded.as_ref().expect("current study").shape;
    for entry in std::fs::read_dir(replacement_root.path()).expect("replacement entries") {
        let path = entry.expect("replacement entry").path();
        std::fs::remove_file(path).expect("remove replacement instance");
    }

    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("failed selection remains recoverable");

    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.selection.is_some());
    assert_eq!(
        session.app.loaded.as_ref().expect("current study").shape,
        previous_shape
    );
    assert!(session.app.status_message.contains("could not be opened"));
}

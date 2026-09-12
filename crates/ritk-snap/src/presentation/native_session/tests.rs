use super::frame::{OVERLAY_BAR_HEIGHT, OVERLAY_TEXT};
use super::*;
use crate::dicom::loader::tests::fixtures;
use metis_platform::native::{ModifierState, WindowEvent};

fn session() -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let path = root.path().to_path_buf();
    fixtures::write_study(&path, "CT", fixtures::SERIES_UID).expect("write study");
    let mut app = SnapApp::default();
    let volume = load_volume_from_path(&path).expect("load study fixture");
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            false,
        )
        .expect("native session"),
        root,
    )
}

#[test]
fn native_session_renders_and_steps_the_loaded_slice() {
    let (mut session, _root) = session();
    let initial = session.views[0].frame().clone();
    let (x, y) = session.viewports[0].center();
    let flow = session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: -120,
            modifiers: ModifierState::NONE,
        }])
        .expect("wheel transition");
    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert_eq!(session.app.viewer_state.slice_index, 2);
    assert_ne!(session.views[0].frame(), &initial);
    assert_eq!(session.views[0].frame().width(), 4);
    assert_eq!(session.views[0].frame().height(), 2);
}

#[test]
fn native_session_keyboard_navigation_updates_presented_frame() {
    let (mut session, _root) = session();
    let initial_frame = session.framebuffer.clone();
    let initial_slice = session.app.viewer_state.slice_index;
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x22,
            repeated: false,
        }])
        .expect("page-down transition");
    assert_eq!(session.app.viewer_state.slice_index, initial_slice + 1);
    assert_ne!(session.framebuffer.pixels(), initial_frame.pixels());
    assert!(
        session
            .observation
            .frame_generations
            .load(Ordering::Relaxed)
            > 1
    );
}

#[test]
fn native_session_drag_updates_pan_and_presented_frame() {
    let (mut session, _root) = session();
    session.app.active_tool = crate::tools::kind::ToolKind::Pan;
    let initial_frame = session.framebuffer.clone();
    let (start_x, start_y) = session.viewports[0].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: start_x + 24,
                y: start_y + 12,
            },
            WindowEvent::PointerUp {
                x: start_x + 24,
                y: start_y + 12,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("pan transition");
    assert!(session.app.pan_offset.x > 0.0);
    assert!(session.app.pan_offset.y > 0.0);
    assert_ne!(session.framebuffer.pixels(), initial_frame.pixels());
    assert!(
        session
            .observation
            .frame_generations
            .load(Ordering::Relaxed)
            > 1
    );
}

#[test]
fn native_session_composes_three_views_and_routes_wheels_by_panel() {
    let (mut session, _root) = session();
    assert_eq!(session.views.len(), 3);
    assert_eq!(
        session
            .viewports
            .iter()
            .map(|viewport| viewport.axis())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(session.framebuffer.width(), INITIAL_WIDTH);
    assert_eq!(session.framebuffer.height(), INITIAL_HEIGHT);
    let (x, y) = session.viewports[1].center();
    session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: -120,
            modifiers: ModifierState::NONE,
        }])
        .expect("coronal wheel transition");
    assert_eq!(session.app.axis, 1);
    assert_eq!(session.app.coronal_slice, 1);
}

#[test]
fn native_application_capture_adds_bounded_ritk_overlays() {
    let (session, _root) = session();
    let (content, _) = surface_frames(
        &session.views,
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        session.app.zoom,
        session.app.pan_offset,
        false,
    )
    .expect("content capture");
    let (application, _) = surface_frames(
        &session.views,
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        session.app.zoom,
        session.app.pan_offset,
        true,
    )
    .expect("application capture");
    assert_ne!(content.pixels(), application.pixels());
    let panel_width = i32::try_from(session.viewports[0].panel_width()).expect("panel width");
    let has_overlay_text = (0..panel_width)
        .any(|x| (0..OVERLAY_BAR_HEIGHT).any(|y| application.get_pixel(x, y) == OVERLAY_TEXT));
    assert!(
        has_overlay_text,
        "application capture includes plane label text"
    );
}

#[test]
fn native_session_zoom_resize_and_minimize_are_bounded() {
    let (mut session, _root) = session();
    session
        .handle_events(&[WindowEvent::Resized {
            width: 400,
            height: 300,
        }])
        .expect("resize transition");
    assert_eq!(session.framebuffer.width(), 400);
    assert_eq!(session.framebuffer.height(), 300);
    session
        .handle_events(&[WindowEvent::Resized {
            width: 0,
            height: 0,
        }])
        .expect("minimize transition");
    assert!(session.minimized);
    assert_eq!(session.framebuffer.width(), 400);
    session
        .handle_events(&[WindowEvent::Resized {
            width: 400,
            height: 300,
        }])
        .expect("restore transition");
    assert!(!session.minimized);
}

#[test]
fn native_session_focus_loss_cancels_pointer_gesture() {
    let (mut session, _root) = session();
    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::FocusLost,
        ])
        .expect("focus cancellation");
    assert!(session.app.tool_state.is_idle());
    assert!(session.active_view.is_none());
}

#[test]
fn native_session_rejects_zero_dpi_and_records_close() {
    let (mut session, _root) = session();
    let initial_slice = session.app.viewer_state.slice_index;
    let (x, y) = session.viewports[0].center();
    let error = session
        .handle_events(&[
            WindowEvent::PointerWheel {
                x,
                y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
            WindowEvent::DpiChanged { dpi: 0 },
        ])
        .expect_err("zero DPI");
    assert!(error.to_string().contains("DPI"));
    assert_eq!(session.app.viewer_state.slice_index, initial_slice);
    assert_eq!(
        session
            .handle_events(&[WindowEvent::CloseRequested])
            .expect("close transition"),
        NativeFlow::Exit
    );
}

#[test]
fn native_session_capture_closes_after_one_idle_batch() {
    let (mut session, _root) = session();
    session.capture_after_idle = true;
    assert_eq!(
        session.handle_events(&[]).expect("capture idle transition"),
        NativeFlow::Exit
    );
    assert!(session
        .observation
        .final_frame
        .lock()
        .expect("capture observation")
        .as_ref()
        .is_some_and(|frame| {
            frame.width() == INITIAL_WIDTH && frame.height() == INITIAL_HEIGHT
        }));
}

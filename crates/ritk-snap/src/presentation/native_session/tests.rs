use super::layout::{
    crosshair_overlay, surface_frames, surface_frames_with_projection, CROSSHAIR_COLOR,
};
use super::layout::{OVERLAY_BAR_HEIGHT, OVERLAY_TEXT};
use super::*;
use crate::dicom::loader::tests::fixtures;
#[cfg(feature = "eframe-shell")]
use crate::presentation::PresentationFrame;
use crate::ui::{RotationSteps, ViewTransform};
#[cfg(feature = "eframe-shell")]
use crate::LoadedVolume;
use metis_platform::native::{ModifierState, NativeApplication, NativeFlow, WindowEvent};
use metis_ui_lang::{DisplayCommand, DisplayList};
use std::time::{Duration, Instant};
mod interaction;
mod selection;

fn session() -> (NativeViewerSession, tempfile::TempDir) {
    session_with_mode(NativePresentationMode::Orthogonal)
}

fn session_with_mode(
    presentation_mode: NativePresentationMode,
) -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let path = root.path().to_path_buf();
    fixtures::write_study(&path, "CT", fixtures::SERIES_UID).expect("write study");
    let mut app = SnapApp::default();
    let volume = load_volume_from_path(&path).expect("load study fixture");
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new_with_selection(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            presentation_mode,
            false,
            None,
        )
        .expect("native session"),
        root,
    )
}

#[cfg(feature = "eframe-shell")]
fn session_with_volume(volume: LoadedVolume) -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let mut app = SnapApp::default();
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new_with_selection(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            NativePresentationMode::Orthogonal,
            false,
            None,
        )
        .expect("native session"),
        root,
    )
}

#[cfg(feature = "eframe-shell")]
fn expected_native_frame(session: &NativeViewerSession, axis: usize) -> PresentationFrame {
    let volume = session.app.loaded.as_ref().expect("loaded volume");
    let (index, _) = session.app.axis_slice_info(axis);
    PresentationFrame::from_slice(
        volume,
        axis,
        index,
        super::frame::window_level_for_app(&session.app),
        session.app.colormap,
    )
    .expect("expected native frame")
}

#[test]
fn native_session_renders_and_steps_the_loaded_slice() {
    let (mut session, _root) = session();
    let initial_snapshot = session
        .observation
        .snapshot
        .lock()
        .expect("snapshot lock")
        .expect("native session snapshot");
    assert!(initial_snapshot.loaded());
    assert_eq!(initial_snapshot.slice_counts(), [3, 2, 4]);
    assert_eq!(initial_snapshot.axis(), session.app.axis);
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
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.viewer_state.slice_index, 2);
    assert_ne!(session.views[0].frame(), &initial);
    assert_eq!(session.views[0].frame().width(), 4);
    assert_eq!(session.views[0].frame().height(), 2);
    let updated_snapshot = session
        .observation
        .snapshot
        .lock()
        .expect("snapshot lock")
        .expect("updated native session snapshot");
    assert_eq!(updated_snapshot.slice_index(0), Some(2));
    assert_eq!(updated_snapshot.zoom(), session.app.zoom);
}

#[test]
#[cfg(feature = "eframe-shell")]
fn native_session_multiframe_navigation_preserves_frame_values() {
    let root = tempfile::tempdir().expect("multiframe study root");
    fixtures::write_multiframe(root.path(), fixtures::MULTIFRAME_SHAPE[0], None)
        .expect("write multiframe fixture");
    let volume = load_volume_from_path(root.path()).expect("load multiframe study");
    let (mut session, _session_root) = session_with_volume(volume);

    assert_eq!(
        session.app.loaded.as_ref().expect("loaded volume").shape,
        fixtures::MULTIFRAME_SHAPE
    );
    assert_eq!(
        session.app.loaded.as_ref().expect("loaded volume").channels,
        1
    );
    assert_eq!(
        session.views[0].frame(),
        &expected_native_frame(&session, 0)
    );
    let second_frame = session.views[0].frame().rgba().to_vec();

    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: 120,
            modifiers: ModifierState::NONE,
        }])
        .expect("previous multiframe slice");
    assert_eq!(session.app.viewer_state.slice_index, 0);
    assert_eq!(
        session.views[0].frame(),
        &expected_native_frame(&session, 0)
    );
    assert_ne!(session.views[0].frame().rgba(), second_frame.as_slice());
    assert_ne!(
        session.framebuffer.get_pixel(100, 100),
        metis_platform::Color::BLACK
    );
}

#[test]
#[cfg(feature = "eframe-shell")]
fn native_session_preserves_rgb_multiframe_channels() {
    let root = tempfile::tempdir().expect("RGB multiframe study root");
    fixtures::write_color_multiframe(root.path()).expect("write RGB multiframe fixture");
    let volume = load_volume_from_path(root.path()).expect("load RGB multiframe study");
    let (session, _session_root) = session_with_volume(volume);

    let loaded = session.app.loaded.as_ref().expect("loaded volume");
    assert_eq!(loaded.shape, fixtures::COLOR_MULTIFRAME_SHAPE);
    assert_eq!(loaded.channels, 3);
    for axis in 0..3 {
        assert_eq!(
            session.views[axis].frame(),
            &expected_native_frame(&session, axis),
            "native RGB frame changed on axis {axis}"
        );
    }
    assert!(
        session
            .views
            .iter()
            .flat_map(|view| view.frame().rgba().chunks_exact(4))
            .any(|pixel| pixel == [0, 255, 255, 255]),
        "native RGB presentation must retain a cyan source channel"
    );
    assert!(
        session
            .framebuffer
            .pixels()
            .chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|&channel| channel != 0)),
        "composed native framebuffer must retain non-black RGB content"
    );
}

#[test]
fn native_session_reuses_transformed_frame_storage_across_refreshes() {
    let (mut session, _root) = session();
    session.app.view_transform = ViewTransform {
        rotation: RotationSteps::Ninety,
        ..ViewTransform::default()
    };

    session.refresh_frame().expect("first transformed refresh");
    let first_frames = session
        .views
        .iter()
        .map(|view| {
            (
                view.frame().width(),
                view.frame().height(),
                view.frame().rgba().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let first_capacities = session
        .render_scratch
        .iter()
        .map(|scratch| scratch.rgba.capacity())
        .collect::<Vec<_>>();

    session.refresh_frame().expect("repeat transformed refresh");
    let second_frames = session
        .views
        .iter()
        .map(|view| {
            (
                view.frame().width(),
                view.frame().height(),
                view.frame().rgba().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let second_capacities = session
        .render_scratch
        .iter()
        .map(|scratch| scratch.rgba.capacity())
        .collect::<Vec<_>>();

    assert_eq!(
        second_frames, first_frames,
        "repeated native render changed pixels"
    );
    assert_eq!(
        second_capacities, first_capacities,
        "native render scratch grew after warmup"
    );
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
            modifiers: ModifierState::NONE,
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
fn native_session_reopens_selected_study_through_the_ritk_loader() {
    let (mut session, _initial_root) = session();
    let replacement_root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_grayscale_presentation(replacement_root.path(), "MONOCHROME2", None)
        .expect("write replacement study");

    session
        .open_study_path(replacement_root.path())
        .expect("reopen selected study");
    session.refresh_frame().expect("render replacement study");

    let loaded = session.app.loaded.as_ref().expect("replacement volume");
    assert_eq!(loaded.shape, [1, 1, 4]);
    assert_eq!(session.views[0].frame().width(), 4);
    assert_eq!(session.views[0].frame().height(), 1);
    assert!(!session.app.cine.enabled);
    assert!(session
        .app
        .status_message
        .contains("Loaded native Métis series"));
    assert!(
        session
            .observation
            .frame_generations
            .load(Ordering::Relaxed)
            > 1
    );
}

#[test]
fn native_session_reopen_failure_preserves_the_current_study() {
    let (mut session, _initial_root) = session();
    let empty_root = tempfile::tempdir().expect("empty study root");
    let previous_shape = session.app.loaded.as_ref().expect("initial volume").shape;

    let error = session
        .open_study_path(empty_root.path())
        .expect_err("empty folder must fail to load");

    assert!(error.to_string().contains("open selected RITK study"));
    assert_eq!(
        session.app.loaded.as_ref().expect("initial volume").shape,
        previous_shape
    );
}

#[test]
fn native_session_space_toggles_cine_and_ignores_repeat() {
    let (mut session, _root) = session();
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CINE_TOGGLE,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("cine toggle");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.app.cine.enabled);

    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CINE_TOGGLE,
            repeated: true,
            modifiers: ModifierState::NONE,
        }])
        .expect("repeated cine toggle");
    assert!(session.app.cine.enabled);
}

#[test]
fn native_session_crosshair_key_repaints_and_updates_snapshot() {
    let (mut session, _root) = session();
    let hidden = session.framebuffer.clone();
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CROSSHAIR_TOGGLE,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("crosshair toggle");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.app.show_crosshair);
    assert_ne!(session.framebuffer.pixels(), hidden.pixels());
    assert!(session
        .observation
        .snapshot
        .lock()
        .expect("snapshot lock")
        .expect("crosshair snapshot")
        .crosshair_visible());

    let shown = session.framebuffer.clone();
    let repeated = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CROSSHAIR_TOGGLE,
            repeated: true,
            modifiers: ModifierState::NONE,
        }])
        .expect("repeated crosshair toggle");
    assert_eq!(repeated, NativeFlow::Continue { repaint: false });
    assert!(session.app.show_crosshair);
    assert_eq!(session.framebuffer.pixels(), shown.pixels());
}

#[test]
fn native_crosshair_overlay_maps_one_linked_voxel_into_each_plane() {
    let (session, _root) = session();
    let shape = session.app.loaded.as_ref().map(|volume| volume.shape);
    let cursor = session.app.linked_cursor.map(|cursor| cursor.voxel());
    let overlay = crosshair_overlay(&session.views, &session.viewports, shape, cursor, true)
        .expect("crosshair display list");
    assert_eq!(
        overlay
            .commands
            .iter()
            .filter(|command| matches!(command, DisplayCommand::DrawLine { color, .. } if *color == CROSSHAIR_COLOR))
            .count(),
        6,
        "each orthogonal plane receives one horizontal and one vertical line"
    );

    let vertical_x = |display_list: &DisplayList| {
        display_list
            .commands
            .iter()
            .find_map(|command| match command {
                DisplayCommand::DrawLine {
                    start: (x, _),
                    end: (end_x, _),
                    color,
                } if *color == CROSSHAIR_COLOR && x == end_x => Some(*x),
                _ => None,
            })
    };
    let mut flipped_views = session.views.clone();
    flipped_views[0].transform = ViewTransform {
        flip_h: true,
        ..ViewTransform::default()
    };
    let flipped_overlay =
        crosshair_overlay(&flipped_views, &session.viewports, shape, cursor, true)
            .expect("flipped crosshair display list");
    assert_ne!(
        vertical_x(&overlay),
        vertical_x(&flipped_overlay),
        "horizontal flip must move the projected cursor line"
    );
    assert!(
        crosshair_overlay(&session.views, &session.viewports, shape, cursor, false)
            .expect("hidden crosshair display list")
            .commands
            .is_empty()
    );
}

#[test]
fn native_session_cine_rate_controls_repaint_and_bound_overlay() {
    let (mut session, _root) = session();
    let initial_rate = session.app.cine.fps;
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CINE_FPS_UP,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("cine rate increase");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.cine.fps, initial_rate + 1.0);
    assert!(session.app.status_message.contains("FPS"));

    let overlay = super::layout::application_overlay(
        &session.views,
        &session.viewports,
        true,
        session.app.cine.fps,
    )
    .expect("active cine overlay");
    assert!(overlay.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text.contains("Cine:13fps") && text.contains("Space/-/+")
    )));

    let repeated = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_CINE_FPS_UP,
            repeated: true,
            modifiers: ModifierState::NONE,
        }])
        .expect("repeated cine rate increase");
    assert_eq!(repeated, NativeFlow::Continue { repaint: false });
    assert_eq!(session.app.cine.fps, initial_rate + 1.0);
}

#[test]
fn native_session_empty_batch_ticks_cine_from_the_session_clock() {
    let (mut session, _root) = session();
    session.app.cine.set_fps(10.0);
    session.app.cine.set_enabled(true, 0.0);
    session.clock_start = Instant::now()
        .checked_sub(Duration::from_millis(250))
        .expect("invariant: test clock duration is representable");
    let initial_slice = session.app.viewer_state.slice_index;

    let flow = session.handle_events(&[]).expect("native animation tick");

    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_ne!(session.app.viewer_state.slice_index, initial_slice);
    assert!(
        session
            .observation
            .frame_generations
            .load(Ordering::Relaxed)
            > 1
    );
}

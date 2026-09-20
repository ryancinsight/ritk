use super::layout::{surface_frames, surface_frames_with_projection};
use super::layout::{OVERLAY_BAR_HEIGHT, OVERLAY_TEXT};
use super::*;
use crate::dicom::loader::tests::fixtures;
use crate::ui::{RotationSteps, ViewTransform};
use metis_platform::native::{ModifierState, NativeApplication, NativeFlow, WindowEvent};
use metis_ui_lang::DisplayCommand;
use std::time::{Duration, Instant};
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
    assert!(session.app.pan_offset.x() > 0.0);
    assert!(session.app.pan_offset.y() > 0.0);
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
fn native_session_window_level_drag_updates_the_presented_study() {
    let (mut session, _root) = session();
    session.app.active_tool = crate::tools::kind::ToolKind::WindowLevel;
    let initial_frame = session.framebuffer.clone();
    let initial_center = session
        .app
        .viewer_state
        .window_center
        .expect("loaded fixture supplies a window center");
    let initial_width = session
        .app
        .viewer_state
        .window_width
        .expect("loaded fixture supplies a window width");
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
                y: start_y - 12,
            },
            WindowEvent::PointerUp {
                x: start_x + 24,
                y: start_y - 12,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("window-level transition");

    assert_ne!(
        session.app.viewer_state.window_center,
        Some(initial_center),
        "vertical drag changes the window center"
    );
    assert_ne!(
        session.app.viewer_state.window_width,
        Some(initial_width),
        "horizontal drag changes the window width"
    );
    assert_ne!(
        session.framebuffer.pixels(),
        initial_frame.pixels(),
        "window-level transition changes presented pixels"
    );
    assert!(session.app.tool_state.is_idle());
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
fn native_session_mip_layout_composes_a_fourth_display_panel() {
    let (session, _root) = session_with_mode(NativePresentationMode::OrthogonalWithMip);
    let projection = session.projection.as_ref().expect("MIP layout projection");
    assert_eq!(projection.frame.width(), 4);
    assert_eq!(projection.frame.height(), 2);
    assert_eq!(session.framebuffer.width(), INITIAL_WIDTH);
    assert_eq!(session.framebuffer.height(), INITIAL_HEIGHT);
    assert_eq!(
        session
            .viewports
            .iter()
            .map(|viewport| viewport.axis())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_ne!(
        session.framebuffer.get_pixel(960, 600),
        metis_platform::Color::BLACK,
        "the fourth panel contains the rendered MIP"
    );
}

#[test]
fn native_session_scalar_projection_modes_render_and_label() {
    for (mode, label) in [
        (NativePresentationMode::OrthogonalWithMip, "MIP"),
        (NativePresentationMode::OrthogonalWithMinip, "MinIP"),
        (NativePresentationMode::OrthogonalWithAverage, "Average"),
    ] {
        let (session, _root) = session_with_mode(mode);
        let projection = session.projection.as_ref().expect("projection layout");
        assert_eq!(projection.statistic.label(), label);
        assert_ne!(
            session.framebuffer.get_pixel(960, 600),
            metis_platform::Color::BLACK,
            "the {label} panel contains rendered scalar pixels"
        );
        let overlay = super::layout::projection_overlay(projection, 640, 400, 640, 400)
            .expect("projection overlay");
        assert!(overlay.commands.iter().any(|command| matches!(
            command,
            DisplayCommand::DrawText { text, .. } if text.contains(label)
        )));
    }
}

#[test]
fn native_session_mip_application_overlay_labels_the_fourth_panel() {
    let (session, _root) = session_with_mode(NativePresentationMode::OrthogonalWithMip);
    let projection = session.projection.as_ref().expect("MIP layout projection");
    let (application, _) = surface_frames_with_projection(
        &session.views,
        projection,
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        session.app.zoom,
        session.app.pan_offset,
        false,
        session.app.cine.fps,
        true,
    )
    .expect("MIP application capture");
    let has_overlay_text =
        (644..1280).any(|x| (404..424).any(|y| application.get_pixel(x, y) == OVERLAY_TEXT));
    assert!(
        has_overlay_text,
        "MIP application capture includes panel text"
    );
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
        session.app.cine.fps,
        false,
    )
    .expect("content capture");
    let (application, _) = surface_frames(
        &session.views,
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        session.app.zoom,
        session.app.pan_offset,
        false,
        session.app.cine.fps,
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
fn native_overlay_is_emitted_as_metis_display_commands() {
    let (session, _root) = session();
    let overlay = super::layout::application_overlay(
        &session.views,
        &session.viewports,
        false,
        session.app.cine.fps,
    )
    .expect("Métis overlay display list");
    assert_eq!(
        overlay
            .commands
            .iter()
            .filter(|command| matches!(command, DisplayCommand::FillRect { .. }))
            .count(),
        6,
        "each orthogonal panel receives two Metis chrome bars"
    );
    assert!(overlay.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text == "METIS  RITK-SNAP  Axial"
    )));
    assert!(overlay.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text.starts_with(&format!(
            "Slice {}/{}",
            session.views[0].slice_index + 1,
            session.views[0].slice_count
        ))
    )));
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

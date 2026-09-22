//! Native pointer, layout, capture, and lifecycle tests.

use super::*;
use crate::presentation::PaneLayout;

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
fn responsive_native_layout_selects_single_dual_and_quad_panes() {
    let (session, _root) = session_with_responsive_mode();
    assert_eq!(session.viewports[0].panel_width(), 638);
    assert_eq!(session.viewports[1].panel_width(), 638);
    assert_eq!(session.viewports[2].panel_width(), 638);
    assert_ne!(
        session.framebuffer.get_pixel(960, 600),
        metis_platform::Color::BLACK,
        "responsive quad layout presents the real scalar projection"
    );

    for (layout, width, height, expected_visible) in [
        (PaneLayout::Single, 500, 400, 1_usize),
        (PaneLayout::Dual, 800, 600, 2_usize),
        (PaneLayout::Quad, 1_200, 800, 4_usize),
    ] {
        let (frame, viewports) = super::layout::surface_frames_responsive(
            &session.views,
            session.projection.as_ref(),
            layout,
            width,
            height,
            session.app.zoom,
            session.app.pan_offset,
            false,
            session.app.cine.fps,
            false,
        )
        .expect("responsive layout");
        assert_eq!(layout.pane_count(), expected_visible);
        assert_eq!(
            viewports
                .iter()
                .filter(|viewport| viewport.panel_width() > 0)
                .count(),
            usize::min(expected_visible, 3),
            "only orthogonal panes receive input viewports"
        );
        assert!(
            frame.pixels().iter().any(|pixel| *pixel != 0xFF00_0000),
            "responsive {layout:?} layout presents real MRI pixels"
        );
    }
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

use super::*;

#[test]
fn invalid_oblique_depth_scroll_keeps_the_last_frame_and_recovery_path() {
    let mut session = oblique_session();
    let initial_plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("initial physical plane");
    let (x, y) = screen_point(
        &session,
        [
            f64::from(u32::try_from(initial_plane.dimensions()[0] - 1).expect("width")) * 0.5,
            f64::from(u32::try_from(initial_plane.dimensions()[1] - 1).expect("height")) * 0.5,
        ],
    );
    click(&mut session, x, y);

    let original_framebuffer = session.framebuffer.pixels().to_vec();
    let original_viewport = session.oblique_viewport.expect("oblique mapper");
    let original_oblique = session.oblique.as_ref().expect("oblique view");
    let original_pixels = original_oblique.frame.rgba().to_vec();
    let original_plane = original_oblique.plane.expect("current plane");
    let original_orientation = original_oblique.orientation;
    let volume = session
        .app
        .loaded
        .as_ref()
        .expect("analytic fixture volume");
    let volume_diagonal = volume
        .shape
        .into_iter()
        .zip(volume.spacing)
        .map(|(extent, spacing)| {
            let last_index = extent.checked_sub(1).expect("positive fixture extent");
            let span =
                f64::from(u32::try_from(last_index).expect("fixture extent fits u32")) * spacing;
            span * span
        })
        .sum::<f64>()
        .sqrt();
    let depth_step_mm = initial_plane
        .depth_step()
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let minimum_wheel_shift_mm = f64::from(i16::MIN.unsigned_abs())
        / crate::presentation::native_session::oblique::NATIVE_WHEEL_DELTA
        * depth_step_mm;
    assert!(
        minimum_wheel_shift_mm > volume_diagonal,
        "the native wheel boundary exceeds the complete source-volume diagonal"
    );
    let flow = session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: i16::MIN,
            modifiers: ModifierState::NONE,
        }])
        .expect("out-of-volume oblique shift is a rejected gesture");

    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert!(
        session
            .app
            .status_message
            .starts_with("Oblique plane shift rejected:")
    );
    assert_eq!(session.framebuffer.pixels(), original_framebuffer);
    assert_eq!(
        session.oblique_viewport,
        Some(original_viewport),
        "rejected geometry leaves the pixel-to-patient mapper unchanged"
    );
    let oblique = session.oblique.as_ref().expect("retained oblique view");
    assert_eq!(oblique.frame.rgba(), original_pixels);
    assert_eq!(oblique.plane, Some(original_plane));
    assert_eq!(oblique.orientation, original_orientation);

    let recovered = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x27,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("valid rotation remains available after a rejected shift");
    assert_eq!(recovered, NativeFlow::Continue { repaint: true });
    let oblique = session.oblique.as_ref().expect("retained oblique view");
    assert_eq!(
        oblique.orientation.yaw_degrees(),
        original_orientation.yaw_degrees() + 5.0
    );
    assert_ne!(oblique.frame.rgba(), original_pixels);
    let plane = oblique.plane.expect("recovered physical plane");
    session
        .oblique_viewport
        .expect("recovered mapper")
        .validate_dimensions(plane.dimensions())
        .expect("recovered mapper matches rendered plane");
}

#[test]
fn non_finite_oblique_rotation_preserves_the_presented_plane() {
    let mut session = oblique_session();
    let original_framebuffer = session.framebuffer.pixels().to_vec();
    let original_viewport = session.oblique_viewport;
    let oblique = session.oblique.as_ref().expect("oblique view");
    let original_pixels = oblique.frame.rgba().to_vec();
    let original_plane = oblique.plane.expect("physical plane");
    let original_orientation = oblique.orientation;

    let accepted =
        session
            .oblique
            .as_mut()
            .expect("oblique view")
            .rotate(&mut session.app, f64::NAN, 0.0);

    assert!(!accepted);
    assert!(
        session
            .app
            .status_message
            .starts_with("Oblique rotation rejected:")
    );
    assert_eq!(session.framebuffer.pixels(), original_framebuffer);
    assert_eq!(session.oblique_viewport, original_viewport);
    let oblique = session.oblique.as_ref().expect("retained oblique view");
    assert_eq!(oblique.frame.rgba(), original_pixels);
    assert_eq!(oblique.plane, Some(original_plane));
    assert_eq!(oblique.orientation, original_orientation);
}

#[test]
fn changing_oblique_plane_clears_an_unfinished_patient_length() {
    let mut session = oblique_session();
    session.app.active_tool = ToolKind::MeasureLength;
    let plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("initial physical plane");
    let point = [
        f64::from(u32::try_from(plane.dimensions()[0] - 1).expect("width")) * 0.5,
        f64::from(u32::try_from(plane.dimensions()[1] - 1).expect("height")) * 0.5,
    ];
    let (x, y) = screen_point(&session, point);
    click(&mut session, x, y);
    assert!(matches!(
        session.app.tool_state,
        crate::tools::interaction::ToolState::PatientLength1 { .. }
    ));

    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x27,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("rotate the oblique plane");
    assert!(matches!(
        session.app.tool_state,
        crate::tools::interaction::ToolState::Idle
    ));
    assert!(
        session
            .app
            .status_message
            .contains("pending length start cleared")
    );
    assert!(session.app.annotations.is_empty());

    let shifted_plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("rotated physical plane");
    let point = [
        f64::from(u32::try_from(shifted_plane.dimensions()[0] - 1).expect("width")) * 0.5,
        f64::from(u32::try_from(shifted_plane.dimensions()[1] - 1).expect("height")) * 0.5,
    ];
    let (x, y) = screen_point(&session, point);
    click(&mut session, x, y);
    assert!(matches!(
        session.app.tool_state,
        crate::tools::interaction::ToolState::PatientLength1 { .. }
    ));
    session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: -30,
            modifiers: ModifierState::NONE,
        }])
        .expect("shift the oblique plane");
    assert!(matches!(
        session.app.tool_state,
        crate::tools::interaction::ToolState::Idle
    ));
    assert!(
        session
            .app
            .status_message
            .contains("pending length start cleared")
    );
    assert!(session.app.annotations.is_empty());
}

#[test]
fn native_oblique_rotation_and_depth_wheel_leave_orthogonal_slices_unchanged() {
    let mut session = oblique_session();
    let initial_axis = session.app.axis;
    let initial_slice = session.app.viewer_state.slice_index;
    let initial_yaw = session
        .oblique
        .as_ref()
        .expect("oblique view")
        .orientation
        .yaw_degrees();
    let plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("validated oblique plane");
    let dimensions = plane.dimensions();
    let pixel = [
        f64::from(u32::try_from(dimensions[0] - 1).expect("bounded width")) * 0.5,
        f64::from(u32::try_from(dimensions[1] - 1).expect("bounded height")) * 0.5,
    ];
    let (x, y) = screen_point(&session, pixel);
    click(&mut session, x, y);

    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x27,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("rotate selected oblique plane");
    assert_eq!(
        session
            .oblique
            .as_ref()
            .expect("oblique view")
            .orientation
            .yaw_degrees(),
        initial_yaw + 5.0
    );
    let rotated_plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("rotated plane remains valid");
    let before_shift = rotated_plane.origin();
    let (x, y) = screen_point(
        &session,
        [
            f64::from(u32::try_from(rotated_plane.dimensions()[0] - 1).expect("width")) * 0.5,
            f64::from(u32::try_from(rotated_plane.dimensions()[1] - 1).expect("height")) * 0.5,
        ],
    );
    session
        .handle_events(&[WindowEvent::PointerWheel {
            x,
            y,
            delta_x: 0,
            delta_y: -30,
            modifiers: ModifierState::NONE,
        }])
        .expect("shift selected oblique plane");

    assert_eq!(session.app.axis, initial_axis);
    assert_eq!(session.app.viewer_state.slice_index, initial_slice);
    let shifted_plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("current oblique plane");
    assert_ne!(shifted_plane.origin(), before_shift);
}

#[test]
fn resized_rotation_refreshes_the_frame_and_mapper_after_input() {
    let mut session = oblique_session();
    let initial_plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("initial physical plane");
    let initial_yaw = session
        .oblique
        .as_ref()
        .expect("oblique view")
        .orientation
        .yaw_degrees();
    let center = [
        f64::from(u32::try_from(initial_plane.dimensions()[0] - 1).expect("width")) * 0.5,
        f64::from(u32::try_from(initial_plane.dimensions()[1] - 1).expect("height")) * 0.5,
    ];
    let (x, y) = screen_point(&session, center);
    click(&mut session, x, y);
    let initial_pixels = session
        .oblique
        .as_ref()
        .expect("oblique view")
        .frame
        .rgba()
        .to_vec();

    let flow = session
        .handle_events(&[
            WindowEvent::Resized {
                width: 1440,
                height: 900,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x27,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("coalesced resize and oblique rotation");

    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    let oblique = session.oblique.as_ref().expect("oblique view");
    let plane = oblique.plane.expect("rotated physical plane");
    assert_eq!(oblique.orientation.yaw_degrees(), initial_yaw + 5.0);
    assert_eq!(
        [oblique.frame.width(), oblique.frame.height()],
        [
            u32::try_from(plane.dimensions()[0]).expect("bounded plane width"),
            u32::try_from(plane.dimensions()[1]).expect("bounded plane height"),
        ]
    );
    assert_ne!(oblique.frame.rgba(), initial_pixels.as_slice());
    session
        .oblique_viewport
        .expect("refreshed oblique viewport")
        .validate_dimensions(plane.dimensions())
        .expect("pointer mapper matches the rotated plane");
}

#[test]
fn coalesced_pointer_batch_routes_each_sample_to_its_panel() {
    let mut session = oblique_session();
    let plane = session
        .oblique
        .as_ref()
        .and_then(|view| view.plane)
        .expect("validated oblique plane");
    let viewport = session.oblique_viewport.expect("oblique mapper");
    let pixel = [
        f64::from(u32::try_from(plane.dimensions()[0] - 1).expect("width")) * 0.5,
        f64::from(u32::try_from(plane.dimensions()[1] - 1).expect("height")) * 0.5,
    ];
    let (oblique_x, oblique_y) = screen_point(&session, pixel);
    let mapped = viewport
        .map(ViewportPoint::new(
            f64::from(oblique_x),
            f64::from(oblique_y),
        ))
        .expect("coalesced click falls inside the oblique pane");
    let volume = session.app.loaded.as_ref().expect("analytic volume");
    let expected_voxel = plane
        .sample_pixel(volume, mapped)
        .expect("screen pixel samples the rendered plane")
        .nearest_voxel();
    let (orthogonal_x, orthogonal_y) = session.viewports[0].center();

    session
        .handle_events(&[
            WindowEvent::PointerMove {
                x: orthogonal_x,
                y: orthogonal_y,
            },
            WindowEvent::PointerDown {
                x: oblique_x,
                y: oblique_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: oblique_x,
                y: oblique_y,
                button: MouseButton::Left,
            },
        ])
        .expect("ordered coalesced events route by their individual panel");

    assert_eq!(
        session.app.linked_cursor.map(|cursor| cursor.voxel()),
        Some(expected_voxel)
    );
    assert!(session.selected_oblique);
}

#[test]
fn native_oblique_pan_and_zoom_change_only_the_selected_panel() {
    let mut session = oblique_session();
    let app_pan = session.app.pan_offset;
    let app_zoom = session.app.zoom;
    let orthogonal_bounds = session.viewports.map(|viewport| viewport.image_bounds());
    let (start_x, start_y) = screen_point(
        &session,
        [
            f64::from(session.oblique.as_ref().expect("view").frame.width() - 1) * 0.5,
            f64::from(session.oblique.as_ref().expect("view").frame.height() - 1) * 0.5,
        ],
    );

    session.app.active_tool = ToolKind::Pan;
    let pan_before = session.oblique.as_ref().expect("view").pan_offset;
    let image_before_pan = session
        .oblique_viewport
        .expect("oblique mapper")
        .image_bounds();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: start_x + 16,
                y: start_y + 8,
            },
            WindowEvent::PointerUp {
                x: start_x + 16,
                y: start_y + 8,
                button: MouseButton::Left,
            },
        ])
        .expect("pan the selected oblique panel");
    let pan_after = session.oblique.as_ref().expect("view").pan_offset;
    assert_eq!(pan_after.x(), pan_before.x() + 16.0);
    assert_eq!(pan_after.y(), pan_before.y() + 8.0);
    assert_eq!(session.app.pan_offset, app_pan);
    assert_eq!(
        session.viewports.map(|viewport| viewport.image_bounds()),
        orthogonal_bounds
    );
    assert_ne!(
        session
            .oblique_viewport
            .expect("oblique mapper after pan")
            .image_bounds(),
        image_before_pan
    );

    session.app.active_tool = ToolKind::Zoom;
    let (zoom_x, zoom_y) = screen_point(
        &session,
        [
            f64::from(session.oblique.as_ref().expect("view").frame.width() - 1) * 0.5,
            f64::from(session.oblique.as_ref().expect("view").frame.height() - 1) * 0.5,
        ],
    );
    let local_zoom = session.oblique.as_ref().expect("view").zoom;
    let image_before_zoom = session
        .oblique_viewport
        .expect("oblique mapper before zoom")
        .image_bounds();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: zoom_x,
                y: zoom_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: zoom_x,
                y: zoom_y + 24,
            },
            WindowEvent::PointerUp {
                x: zoom_x,
                y: zoom_y + 24,
                button: MouseButton::Left,
            },
        ])
        .expect("zoom the selected oblique panel");
    assert_ne!(session.oblique.as_ref().expect("view").zoom, local_zoom);
    assert_eq!(session.app.zoom, app_zoom);
    assert_eq!(session.app.pan_offset, app_pan);
    assert_eq!(
        session.viewports.map(|viewport| viewport.image_bounds()),
        orthogonal_bounds
    );
    assert_ne!(
        session
            .oblique_viewport
            .expect("oblique mapper after zoom")
            .image_bounds(),
        image_before_zoom
    );
}

#[test]
fn native_oblique_window_level_drag_updates_the_shared_intensity_mapping() {
    let mut session = oblique_session();
    session.app.viewer_state.window_center = Some(600.0);
    session.app.viewer_state.window_width = Some(1200.0);
    session.app.bump_visual_revision();
    session
        .refresh_frame()
        .expect("render a window spanning the analytic fixture range");
    let [width, height] = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("plane")
        .dimensions();
    let (x, y) = screen_point(
        &session,
        [
            f64::from(u32::try_from(width - 1).expect("width")) * 0.5,
            f64::from(u32::try_from(height - 1).expect("height")) * 0.5,
        ],
    );
    click(&mut session, x, y);
    session.app.active_tool = ToolKind::WindowLevel;
    let initial_center = session
        .app
        .viewer_state
        .window_center
        .expect("fixture has a center");
    let initial_width = session
        .app
        .viewer_state
        .window_width
        .expect("fixture has a width");
    let initial_pixels = session
        .oblique
        .as_ref()
        .expect("view")
        .frame
        .rgba()
        .to_vec();
    assert!(
        initial_pixels.chunks_exact(4).map(|pixel| pixel[0]).min()
            < initial_pixels.chunks_exact(4).map(|pixel| pixel[0]).max()
    );
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: x + 24,
                y: y - 12,
            },
            WindowEvent::PointerUp {
                x: x + 24,
                y: y - 12,
                button: MouseButton::Left,
            },
        ])
        .expect("window-level drag in oblique panel");

    assert_ne!(session.app.viewer_state.window_center, Some(initial_center));
    assert_ne!(session.app.viewer_state.window_width, Some(initial_width));
    assert_ne!(
        session.oblique.as_ref().expect("view").frame.rgba(),
        initial_pixels.as_slice()
    );
    assert!(session.app.tool_state.is_idle());
}

#[test]
fn failed_oblique_pointer_dispatch_restores_panel_routing_state() {
    let mut session = oblique_session();
    let axis = session.app.axis;
    let plane = session
        .oblique
        .as_ref()
        .and_then(|oblique| oblique.plane)
        .expect("validated oblique plane");
    let dimensions = plane.dimensions();
    let (x, y) = screen_point(
        &session,
        [
            f64::from(u32::try_from(dimensions[0] - 1).expect("width")) * 0.5,
            f64::from(u32::try_from(dimensions[1] - 1).expect("height")) * 0.5,
        ],
    );
    let error = session
        .handle_events(&[WindowEvent::PointerDown {
            x,
            y,
            button: MouseButton::Right,
        }])
        .expect_err("unsupported oblique pointer button is rejected");
    assert!(error.to_string().contains("pointer button"));
    assert_eq!(session.app.axis, axis);
    assert!(!session.selected_oblique);
    assert!(!session.active_oblique);
    assert!(session.active_view.is_none());
}

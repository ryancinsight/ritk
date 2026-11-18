//! Native display controls: window/level presets through menus and keys.

use super::support::{control_center, session};
use super::*;
use crate::app::action_adapter::{
    VIRTUAL_KEY_COLORMAP_CYCLE, VIRTUAL_KEY_FLIP_HORIZONTAL, VIRTUAL_KEY_FLIP_VERTICAL,
    VIRTUAL_KEY_ORIENTATION_RESET, VIRTUAL_KEY_PRESET_KEYS, VIRTUAL_KEY_ROTATE,
    VIRTUAL_KEY_ZOOM_FIT,
};
use crate::presentation::ViewportPoint;
use crate::render::NamedColorMap;
use crate::ui::RotationSteps;

#[test]
fn digit_keys_apply_ct_presets_and_repaint_presented_pixels() {
    let (mut session, _root) = session();
    let before = session.framebuffer.clone();
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_PRESET_KEYS[4],
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("preset digit key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.viewer_state.window_center, Some(-400.0));
    assert_eq!(session.app.viewer_state.window_width, Some(1_500.0));
    assert_eq!(session.views[0].window_level.center, -400.0);
    assert_eq!(session.views[0].window_level.width, 1_500.0);
    assert_ne!(session.framebuffer.pixels(), before.pixels());
}

#[test]
fn repeated_digit_key_is_consumed_without_repaint() {
    let (mut session, _root) = session();
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_PRESET_KEYS[4],
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("initial preset digit key");
    let shown = session.framebuffer.clone();
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_PRESET_KEYS[4],
            repeated: true,
            modifiers: ModifierState::NONE,
        }])
        .expect("repeated preset digit key");
    assert_eq!(flow, NativeFlow::Continue { repaint: false });
    assert_eq!(session.framebuffer.pixels(), shown.pixels());
}

#[test]
fn display_menu_applies_the_checked_modality_preset() {
    let (mut session, _root) = session();
    let (tab_x, tab_y) = control_center(
        &session,
        None,
        WindowAction::OpenMenu(super::super::window_controls::Menu::Display),
    );
    let (row_x, row_y) = control_center(
        &session,
        Some(super::super::window_controls::Menu::Display),
        WindowAction::ApplyWindowPreset { index: 4 },
    );
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("apply Lung preset from the Display menu");
    assert_eq!(session.app.viewer_state.window_center, Some(-400.0));
    assert_eq!(session.app.viewer_state.window_width, Some(1_500.0));
}

#[test]
fn invert_key_toggles_negative_display_and_repaints() {
    let (mut session, _root) = session();
    assert!(!session.app.invert_display);
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::app::action_adapter::VIRTUAL_KEY_INVERT_DISPLAY,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("invert display key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.app.invert_display);
}

#[test]
fn invert_produces_channel_exact_negative_pixels() {
    let (mut session, _root) = session();
    session.refresh_frame().expect("baseline native frame");
    let baseline: Vec<u8> = session.views[0].frame().rgba().to_vec();
    assert!(
        baseline.chunks_exact(4).any(|pixel| pixel[3] == 255),
        "baseline frame must carry opaque pixels"
    );
    session
        .apply_window_action(WindowAction::ToggleInvert)
        .expect("toggle invert");
    session.refresh_frame().expect("inverted native frame");
    let inverted: Vec<u8> = session.views[0].frame().rgba().to_vec();
    assert_eq!(baseline.len(), inverted.len());
    for (plain, negative) in baseline.chunks_exact(4).zip(inverted.chunks_exact(4)) {
        assert_eq!(negative[3], plain[3], "alpha channel is never inverted");
        assert_eq!(negative[0], 255 - plain[0]);
        assert_eq!(negative[1], 255 - plain[1]);
        assert_eq!(negative[2], 255 - plain[2]);
    }
    assert_ne!(baseline, inverted, "negative display must change pixels");
    session
        .apply_window_action(WindowAction::ToggleInvert)
        .expect("restore normal display");
    session.refresh_frame().expect("restored native frame");
    assert_eq!(session.views[0].frame().rgba(), baseline.as_slice());
}

#[test]
fn display_menu_invert_row_toggles_negative_display() {
    let (mut session, _root) = session();
    let (tab_x, tab_y) = control_center(
        &session,
        None,
        WindowAction::OpenMenu(super::super::window_controls::Menu::Display),
    );
    let (row_x, row_y) = control_center(
        &session,
        Some(super::super::window_controls::Menu::Display),
        WindowAction::ToggleInvert,
    );
    for expected in [true, false] {
        session
            .handle_events(&[
                WindowEvent::PointerDown {
                    x: tab_x,
                    y: tab_y,
                    button: metis_platform::native::MouseButton::Left,
                },
                WindowEvent::PointerUp {
                    x: tab_x,
                    y: tab_y,
                    button: metis_platform::native::MouseButton::Left,
                },
                WindowEvent::PointerDown {
                    x: row_x,
                    y: row_y,
                    button: metis_platform::native::MouseButton::Left,
                },
                WindowEvent::PointerUp {
                    x: row_x,
                    y: row_y,
                    button: metis_platform::native::MouseButton::Left,
                },
            ])
            .expect("toggle invert from the Display menu");
        assert_eq!(session.app.invert_display, expected);
    }
}

#[test]
fn colormap_key_cycles_all_maps_in_stable_order_and_wraps() {
    let (mut session, _root) = session();
    assert_eq!(session.app.colormap, NamedColorMap::Grayscale);
    // Grayscale sits at index 1 of the stable display order.
    let mut expected = NamedColorMap::Grayscale;
    for _ in 0..NamedColorMap::ALL.len() {
        let position = NamedColorMap::ALL
            .iter()
            .position(|map| *map == expected)
            .expect("current map is listed");
        expected = NamedColorMap::ALL[(position + 1) % NamedColorMap::ALL.len()];
        let flow = session
            .handle_events(&[WindowEvent::KeyDown {
                virtual_key: VIRTUAL_KEY_COLORMAP_CYCLE,
                repeated: false,
                modifiers: ModifierState::SHIFT,
            }])
            .expect("colormap cycle key");
        assert_eq!(flow, NativeFlow::Continue { repaint: true });
        assert_eq!(session.app.colormap, expected);
    }
    assert_eq!(session.app.colormap, NamedColorMap::Grayscale);
}

#[test]
fn display_menu_applies_each_colormap_to_presented_pixels() {
    let (mut session, _root) = session();
    session.refresh_frame().expect("baseline native frame");
    let grayscale = session.views[0].frame().rgba().to_vec();
    let (tab_x, tab_y) = control_center(
        &session,
        None,
        WindowAction::OpenMenu(super::super::window_controls::Menu::Display),
    );
    let (row_x, row_y) = control_center(
        &session,
        Some(super::super::window_controls::Menu::Display),
        WindowAction::SetColormap(NamedColorMap::Hot),
    );
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("apply Hot colormap from the Display menu");
    assert_eq!(session.app.colormap, NamedColorMap::Hot);
    assert_ne!(session.views[0].frame().rgba(), grayscale.as_slice());
}

#[test]
fn transform_keys_route_orientation_and_repaint() {
    let (mut session, _root) = session();
    assert!(session.app.view_transform.is_identity());
    let steps = [
        (VIRTUAL_KEY_ROTATE, ModifierState::NONE, 0),
        (VIRTUAL_KEY_ROTATE, ModifierState::SHIFT, 1),
        (VIRTUAL_KEY_FLIP_HORIZONTAL, ModifierState::NONE, 2),
        (VIRTUAL_KEY_FLIP_VERTICAL, ModifierState::NONE, 3),
    ];
    for (key, modifiers, step) in steps {
        let flow = session
            .handle_events(&[WindowEvent::KeyDown {
                virtual_key: key,
                repeated: false,
                modifiers,
            }])
            .expect("orientation key");
        assert_eq!(flow, NativeFlow::Continue { repaint: true });
        let transform = session.app.view_transform;
        match step {
            0 => assert_eq!(transform.rotation, RotationSteps::Ninety),
            1 => assert_eq!(transform.rotation, RotationSteps::Zero),
            2 => assert!(transform.flip_h, "H key must set flip_h"),
            3 => assert!(transform.flip_v, "V key must set flip_v"),
            _ => unreachable!("fixed orientation step table"),
        }
    }
    let transform = session.app.view_transform;
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: crate::ui::tool_shortcuts::VIRTUAL_KEY_ROI_OPEN_POLYGON,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("Open Polygon key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(
        session.app.active_tool,
        crate::tools::kind::ToolKind::RoiOpenPolygon
    );
    assert_eq!(session.app.view_transform, transform);

    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_ORIENTATION_RESET,
            repeated: false,
            modifiers: ModifierState::SHIFT,
        }])
        .expect("orientation reset key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert!(session.app.view_transform.is_identity());
}

#[test]
fn rotate_cw_moves_exact_corner_pixels() {
    let (mut session, _root) = session();
    session.refresh_frame().expect("baseline native frame");
    let (width, height) = {
        let frame = session.views[0].frame();
        (
            usize::try_from(frame.width()).expect("frame width fits"),
            usize::try_from(frame.height()).expect("frame height fits"),
        )
    };
    assert!((width, height) == (4, 2), "fixture frame is 4x2");
    let baseline = session.views[0].frame().rgba().to_vec();
    // 90-degree CW maps output(0, 0) to input(bottom-left).
    let bottom_left = baseline[((height - 1) * width * 4)..][..4].to_vec();
    session
        .apply_window_action(WindowAction::RotateCw)
        .expect("rotate clockwise");
    session.refresh_frame().expect("rotated native frame");
    let rotated = session.views[0].frame();
    assert_eq!(
        (
            usize::try_from(rotated.width()).expect("rotated width fits"),
            usize::try_from(rotated.height()).expect("rotated height fits")
        ),
        (height, width)
    );
    assert_eq!(&rotated.rgba()[..4], bottom_left.as_slice());
    let mut sorted_baseline = baseline.clone();
    sorted_baseline.sort_unstable();
    let mut sorted_rotated = rotated.rgba().to_vec();
    sorted_rotated.sort_unstable();
    assert_eq!(sorted_rotated, sorted_baseline);
}

#[test]
fn flipped_mapping_resolves_the_same_source_voxel() {
    type Viewport = super::super::layout::NativeViewport;
    let (mut session, _root) = session();
    session.refresh_frame().expect("baseline native frame");
    let viewport = session.viewports[0];
    let frame = session.views[0].frame();
    let (width, height) = (frame.width(), frame.height());
    let source_of = |viewport: Viewport, pixel: (u32, u32)| {
        let output_size = (viewport.image.width, viewport.image.height);
        let frame_size = (f64::from(width), f64::from(height));
        viewport
            .mapping()
            .map(ViewportPoint::new(
                viewport.image.x + (f64::from(pixel.0) + 0.5) * output_size.0 / frame_size.0,
                viewport.image.y + (f64::from(pixel.1) + 0.5) * output_size.1 / frame_size.1,
            ))
            .expect("interior pixel maps")
    };
    let plain = source_of(viewport, (0, 0));
    session.app.view_transform = session.app.view_transform.toggle_flip_h();
    session.refresh_frame().expect("flipped native frame");
    let flipped_viewport = session.viewports[0];
    assert_eq!(
        (
            session.views[0].frame().width(),
            session.views[0].frame().height()
        ),
        (width, height)
    );
    assert_eq!(
        source_of(flipped_viewport, (width - 1, 0)),
        plain,
        "horizontal flip moves the pixel but not its source voxel"
    );
}

#[test]
fn display_menu_transform_rows_change_presented_pixels() {
    let (mut session, _root) = session();
    let (tab_x, tab_y) = control_center(
        &session,
        None,
        WindowAction::OpenMenu(super::super::window_controls::Menu::Display),
    );
    let (row_x, row_y) = control_center(
        &session,
        Some(super::super::window_controls::Menu::Display),
        WindowAction::RotateCw,
    );
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: tab_x,
                y: tab_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: row_x,
                y: row_y,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("rotate from the Display menu");
    assert_eq!(session.app.view_transform.rotation, RotationSteps::Ninety);
    assert_eq!(
        (
            session.views[0].frame().width(),
            session.views[0].frame().height()
        ),
        (2, 4)
    );
}

#[test]
fn fit_key_restores_unit_zoom_and_repaints() {
    let (mut session, _root) = session();
    session.app.zoom = 2.5;
    session.app.pan_offset = crate::tools::interaction::ViewportOffset::new(3.0, -7.0);
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_ZOOM_FIT,
            repeated: false,
            modifiers: ModifierState::CONTROL,
        }])
        .expect("fit key");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.zoom, 1.0);
    assert_eq!(
        session.app.pan_offset,
        crate::tools::interaction::ViewportOffset::new(0.0, 0.0)
    );
}

#[test]
fn one_to_one_action_undoes_the_panel_fit_scale() {
    let (mut session, _root) = session();
    let before = session.framebuffer.clone();
    let viewport = session.viewports[0];
    let frame = session.views[0].frame();
    let expected = crate::ui::zoom::one_to_one_zoom(
        [viewport.panel_width, viewport.panel_height],
        [frame.width(), frame.height()],
        frame.display_spacing().values(),
    )
    .expect("finite fixture panel extent");
    let changed = session
        .apply_window_action(WindowAction::ZoomOneToOne)
        .expect("zoom 1:1");
    assert!(changed);
    assert_eq!(session.app.zoom, expected);
    session.refresh_frame().expect("recompose at 1:1 zoom");
    assert_ne!(session.framebuffer.pixels(), before.pixels());
}

#[test]
fn out_of_range_preset_index_reports_without_changing_state() {
    let (mut session, _root) = session();
    let before_state = session.app.viewer_state;
    let before_revision = session.app.visual_revision;
    let changed = session
        .apply_window_action(WindowAction::ApplyWindowPreset { index: 99 })
        .expect("preset rejection reports");
    assert!(changed);
    assert_eq!(session.app.viewer_state, before_state);
    assert_eq!(session.app.visual_revision, before_revision);
    assert!(
        session.app.status_message.contains("Window preset"),
        "unexpected status: {}",
        session.app.status_message
    );
}

//! Native display controls: window/level presets through menus and keys.

use super::support::{control_center, session};
use super::*;
use crate::app::action_adapter::VIRTUAL_KEY_PRESET_KEYS;

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

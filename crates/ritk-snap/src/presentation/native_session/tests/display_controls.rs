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

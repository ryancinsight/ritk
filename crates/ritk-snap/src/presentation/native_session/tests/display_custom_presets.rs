//! Native display controls: numbered custom window/level presets.

use super::support::{control_center, session};
use super::*;
use crate::app::action_adapter::VIRTUAL_KEY_PRESET_KEYS;

#[test]
fn custom_preset_save_and_apply_roundtrip() {
    let (mut session, _root) = session();
    session
        .apply_window_action(WindowAction::ApplyWindowPreset { index: 4 })
        .expect("baseline lung preset");
    session
        .apply_window_action(WindowAction::SaveCustomPreset { index: Some(1) })
        .expect("save custom slot");
    assert_eq!(
        session.app.custom_presets[1],
        Some((-400.0, 1_500.0)),
        "custom slot captures the current window"
    );
    session
        .apply_window_action(WindowAction::ApplyWindowPreset { index: 0 })
        .expect("change window");
    let changed = session
        .apply_window_action(WindowAction::ApplyCustomPreset { index: 1 })
        .expect("apply custom slot");
    assert!(changed);
    assert_eq!(session.app.viewer_state.window_center, Some(-400.0));
    assert_eq!(session.app.viewer_state.window_width, Some(1_500.0));
}

#[test]
fn custom_menu_save_fills_first_empty_slot_then_reports_full() {
    let (mut session, _root) = session();
    let (tab_x, tab_y) = control_center(
        &session,
        None,
        WindowAction::OpenMenu(super::super::window_controls::Menu::Display),
    );
    let (row_x, row_y) = control_center(
        &session,
        Some(super::super::window_controls::Menu::Display),
        WindowAction::SaveCustomPreset { index: None },
    );
    for expected_slot in 0..crate::app::state::MAX_CUSTOM_PRESETS {
        let expected = session.app.browser_window_level_values();
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
            .expect("save custom from the Display menu");
        assert_eq!(
            session.app.custom_presets[expected_slot],
            Some(expected),
            "menu save must fill slot {expected_slot} with the current window"
        );
    }
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
        .expect("full custom table reports");
    assert!(
        session.app.status_message.contains("Custom"),
        "unexpected status: {}",
        session.app.status_message
    );
}

#[test]
fn custom_keys_apply_and_save_numbered_slots() {
    let (mut session, _root) = session();
    session
        .apply_window_action(WindowAction::ApplyWindowPreset { index: 4 })
        .expect("baseline lung preset");
    let shift = ModifierState::SHIFT;
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_PRESET_KEYS[2],
            repeated: false,
            modifiers: shift,
        }])
        .expect("shift-digit saves custom slot");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.custom_presets[2], Some((-400.0, 1_500.0)));
    session
        .apply_window_action(WindowAction::ApplyWindowPreset { index: 0 })
        .expect("change window");
    let alt = ModifierState::ALT;
    let flow = session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: VIRTUAL_KEY_PRESET_KEYS[2],
            repeated: false,
            modifiers: alt,
        }])
        .expect("alt-digit applies custom slot");
    assert_eq!(flow, NativeFlow::Continue { repaint: true });
    assert_eq!(session.app.viewer_state.window_center, Some(-400.0));
    assert_eq!(session.app.viewer_state.window_width, Some(1_500.0));
}

#[test]
fn empty_custom_slot_reports_without_changing_state() {
    let (mut session, _root) = session();
    let before_state = session.app.viewer_state;
    let changed = session
        .apply_window_action(WindowAction::ApplyCustomPreset { index: 2 })
        .expect("empty slot reports");
    assert!(changed);
    assert_eq!(session.app.viewer_state, before_state);
    assert!(
        session.app.status_message.contains("Custom"),
        "unexpected status: {}",
        session.app.status_message
    );
}

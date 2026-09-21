//! Native event translation and frame presentation tests.

use super::*;

#[test]
fn native_events_preserve_provider_values() {
    let events = [
        WindowEvent::CloseRequested,
        WindowEvent::Destroyed,
        WindowEvent::FocusGained,
        WindowEvent::FocusLost,
        WindowEvent::AccessibilityAction {
            request: metis_platform::native::AccessibilityActionRequest {
                target_node: 41,
                action: metis_platform::native::AccessibilityAction::SetValue,
                value: Some("patient".to_owned()),
                delta: None,
            },
        },
        WindowEvent::PointerMove { x: -4, y: 8 },
        WindowEvent::PointerDown {
            x: 1,
            y: 2,
            button: metis_platform::native::MouseButton::X1,
        },
        WindowEvent::PointerUp {
            x: 3,
            y: 4,
            button: metis_platform::native::MouseButton::Right,
        },
        WindowEvent::PointerWheel {
            x: -5,
            y: 6,
            delta_x: -240,
            delta_y: 120,
            modifiers: metis_platform::native::ModifierState::NONE,
        },
        WindowEvent::KeyDown {
            virtual_key: 0x41,
            repeated: true,
            modifiers: metis_platform::native::ModifierState::NONE,
        },
        WindowEvent::KeyUp {
            virtual_key: 0x41,
            modifiers: metis_platform::native::ModifierState::NONE,
        },
        WindowEvent::TextInput { character: '中' },
        WindowEvent::TextComposition {
            phase: metis_platform::native::CompositionPhase::Updated,
            text: "A😀".to_owned(),
        },
        WindowEvent::Resized {
            width: 800,
            height: 600,
        },
        WindowEvent::DpiChanged { dpi: 144 },
    ];
    let translated = translate_native_events(&events).expect("translated events");
    assert_eq!(translated.len(), events.len());
    assert_eq!(
        translated.as_ref(),
        &[
            PresentationEvent::CloseRequested,
            PresentationEvent::Destroyed,
            PresentationEvent::FocusGained,
            PresentationEvent::FocusLost,
            PresentationEvent::AccessibilityAction {
                request: AccessibilityActionRequest {
                    target_node: 41,
                    action: AccessibilityAction::SetValue,
                    value: Some("patient".into()),
                    delta: None,
                },
            },
            PresentationEvent::PointerMove { x: -4.0, y: 8.0 },
            PresentationEvent::PointerDown {
                x: 1.0,
                y: 2.0,
                button: PointerButton::X1,
            },
            PresentationEvent::PointerUp {
                x: 3.0,
                y: 4.0,
                button: PointerButton::Right,
            },
            PresentationEvent::PointerWheel {
                x: -5.0,
                y: 6.0,
                delta_x: -240.0,
                delta_y: 120.0,
                modifiers: PresentationModifiers::NONE,
            },
            PresentationEvent::KeyDown {
                virtual_key: 0x41,
                repeated: true,
                modifiers: PresentationModifiers::NONE,
            },
            PresentationEvent::KeyUp {
                virtual_key: 0x41,
                modifiers: PresentationModifiers::NONE,
            },
            PresentationEvent::TextInput { character: '中' },
            PresentationEvent::TextComposition {
                phase: CompositionPhase::Updated,
                text: "A😀".into(),
            },
            PresentationEvent::Resized {
                width: 800,
                height: 600,
            },
            PresentationEvent::DpiChanged { dpi: 144 },
        ]
    );
}

#[test]
fn native_coordinates_preserve_values_above_f32_integer_precision() {
    let events = [WindowEvent::PointerMove {
        x: 16_777_217,
        y: -16_777_217,
    }];
    let translated = translate_native_events(&events).expect("translated events");
    assert_eq!(
        translated.as_ref(),
        &[PresentationEvent::PointerMove {
            x: 16_777_217.0,
            y: -16_777_217.0,
        }]
    );
}

#[test]
fn native_events_reject_oversized_batch() {
    let events = vec![WindowEvent::FocusGained; MAX_PRESENTATION_EVENTS + 1];
    let error = translate_native_events(&events).expect_err("oversized batch");
    assert!(error.to_string().contains("exceeds host limit"));
}

#[test]
fn native_accessibility_value_bound_is_enforced() {
    let events = [WindowEvent::AccessibilityAction {
        request: metis_platform::native::AccessibilityActionRequest {
            target_node: 41,
            action: metis_platform::native::AccessibilityAction::SetValue,
            value: Some("x".repeat(MAX_ACCESSIBILITY_VALUE_BYTES + 1)),
            delta: None,
        },
    }];
    let error = translate_native_events(&events).expect_err("oversized action value");
    assert!(error.to_string().contains("action value length"));
}

#[test]
fn native_events_bound_composition_utf16_units_and_preserve_supplementary_text() {
    let valid_text = "😀".repeat(MAX_COMPOSITION_UNITS / 2);
    let valid = [WindowEvent::TextComposition {
        phase: metis_platform::native::CompositionPhase::Updated,
        text: valid_text.clone(),
    }];
    let translated = translate_native_events(&valid).expect("boundary composition");
    assert_eq!(translated.len(), 1);
    assert_eq!(
        translated.as_ref(),
        &[PresentationEvent::TextComposition {
            phase: CompositionPhase::Updated,
            text: valid_text.into_boxed_str(),
        }]
    );

    let invalid = [WindowEvent::TextComposition {
        phase: metis_platform::native::CompositionPhase::Updated,
        text: format!("{}A", "😀".repeat(MAX_COMPOSITION_UNITS / 2)),
    }];
    let error = translate_native_events(&invalid).expect_err("over-budget composition");
    assert!(error.to_string().contains("UTF-16 units"));
}

#[test]
fn frame_conversion_preserves_rgba_channels() {
    let frame =
        PresentationFrame::from_rgba(2, 1, &[0, 0, 0, 0, 200, 150, 100, 255]).expect("valid frame");
    let framebuffer = to_framebuffer(&frame).expect("framebuffer");
    assert_eq!(framebuffer.get_pixel(0, 0), Color::rgba(0, 0, 0, 0));
    assert_eq!(framebuffer.get_pixel(1, 0), Color::rgba(200, 150, 100, 255));
}

#[test]
fn native_host_presents_one_frame_and_closes() {
    let frame = PresentationFrame::from_rgba(1, 1, &[12, 34, 56, 255]).expect("valid frame");
    let outcome = run_native_frame(frame, "RITK presentation frame test").expect("native frame");
    assert_eq!((outcome.width(), outcome.height()), (1, 1));
    assert_eq!(outcome.pixel_count(), 1);
    assert_eq!(outcome.frame_requests(), 1);
    assert_eq!(outcome.event_batches(), 1);
    assert!(outcome.translated_events() >= 1);
}

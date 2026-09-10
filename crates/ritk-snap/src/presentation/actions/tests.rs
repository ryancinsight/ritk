use super::*;

#[test]
fn click_and_drag_actions_preserve_positions_and_gesture() {
    let mut dispatcher = PresentationDispatcher::new();
    let click = dispatcher
        .dispatch(&[
            PresentationEvent::PointerDown {
                x: 10,
                y: 20,
                button: PointerButton::Left,
            },
            PresentationEvent::PointerUp {
                x: 10,
                y: 20,
                button: PointerButton::Left,
            },
        ])
        .expect("click sequence");
    assert_eq!(
        click.as_ref(),
        &[
            ViewerAction::PointerPressed {
                button: PointerButton::Left,
                position: ViewportPoint::new(10, 20),
            },
            ViewerAction::PointerReleased {
                button: PointerButton::Left,
                position: ViewportPoint::new(10, 20),
                gesture: PointerGesture::Click,
            },
        ]
    );

    let drag = dispatcher
        .dispatch(&[
            PresentationEvent::PointerDown {
                x: 1,
                y: 2,
                button: PointerButton::Right,
            },
            PresentationEvent::PointerMove { x: 4, y: 7 },
            PresentationEvent::PointerUp {
                x: 4,
                y: 7,
                button: PointerButton::Right,
            },
        ])
        .expect("drag sequence");
    assert_eq!(
        drag.as_ref(),
        &[
            ViewerAction::PointerPressed {
                button: PointerButton::Right,
                position: ViewportPoint::new(1, 2),
            },
            ViewerAction::PointerMoved {
                position: ViewportPoint::new(4, 7),
            },
            ViewerAction::PointerDragged {
                button: PointerButton::Right,
                start: ViewportPoint::new(1, 2),
                current: ViewportPoint::new(4, 7),
                delta: PointerDelta { x: 3, y: 5 },
            },
            ViewerAction::PointerReleased {
                button: PointerButton::Right,
                position: ViewportPoint::new(4, 7),
                gesture: PointerGesture::Drag,
            },
        ]
    );
}

#[test]
fn simultaneous_button_drags_have_fixed_order() {
    let mut dispatcher = PresentationDispatcher::new();
    let actions = dispatcher
        .dispatch(&[
            PresentationEvent::PointerDown {
                x: 0,
                y: 0,
                button: PointerButton::Right,
            },
            PresentationEvent::PointerDown {
                x: 0,
                y: 0,
                button: PointerButton::Left,
            },
            PresentationEvent::PointerMove { x: 2, y: 3 },
        ])
        .expect("multi-button drag");
    assert_eq!(
        actions[3..],
        [
            ViewerAction::PointerDragged {
                button: PointerButton::Left,
                start: ViewportPoint::new(0, 0),
                current: ViewportPoint::new(2, 3),
                delta: PointerDelta { x: 2, y: 3 },
            },
            ViewerAction::PointerDragged {
                button: PointerButton::Right,
                start: ViewportPoint::new(0, 0),
                current: ViewportPoint::new(2, 3),
                delta: PointerDelta { x: 2, y: 3 },
            },
        ]
    );
}

#[test]
fn focus_loss_cancels_pressed_buttons_and_clears_state() {
    let mut dispatcher = PresentationDispatcher::new();
    let actions = dispatcher
        .dispatch(&[
            PresentationEvent::PointerDown {
                x: 5,
                y: 6,
                button: PointerButton::X2,
            },
            PresentationEvent::FocusLost,
        ])
        .expect("focus loss");
    assert_eq!(
        actions.as_ref(),
        &[
            ViewerAction::PointerPressed {
                button: PointerButton::X2,
                position: ViewportPoint::new(5, 6),
            },
            ViewerAction::FocusChanged { focused: false },
            ViewerAction::PointerCancelled {
                button: PointerButton::X2,
                position: ViewportPoint::new(5, 6),
            },
        ]
    );
    let release = dispatcher.dispatch(&[PresentationEvent::PointerUp {
        x: 5,
        y: 6,
        button: PointerButton::X2,
    }]);
    assert!(matches!(
        release,
        Err(ActionDispatchError::PointerReleaseWithoutPress {
            button: PointerButton::X2
        })
    ));
}

#[test]
fn malformed_batch_is_atomic_and_composition_is_bounded() {
    let mut dispatcher = PresentationDispatcher::new();
    let malformed = dispatcher.dispatch(&[
        PresentationEvent::PointerDown {
            x: i32::MIN,
            y: 0,
            button: PointerButton::Left,
        },
        PresentationEvent::PointerMove { x: i32::MAX, y: 0 },
    ]);
    assert!(matches!(
        malformed,
        Err(ActionDispatchError::CoordinateDeltaOverflow { .. })
    ));
    let release = dispatcher.dispatch(&[PresentationEvent::PointerUp {
        x: i32::MIN,
        y: 0,
        button: PointerButton::Left,
    }]);
    assert!(matches!(
        release,
        Err(ActionDispatchError::PointerReleaseWithoutPress {
            button: PointerButton::Left
        })
    ));

    let duplicate = dispatcher.dispatch(&[
        PresentationEvent::PointerDown {
            x: 1,
            y: 1,
            button: PointerButton::Middle,
        },
        PresentationEvent::PointerDown {
            x: 2,
            y: 2,
            button: PointerButton::Middle,
        },
    ]);
    assert!(matches!(
        duplicate,
        Err(ActionDispatchError::DuplicatePointerPress {
            button: PointerButton::Middle
        })
    ));

    let oversized = "😀".repeat(MAX_COMPOSITION_UNITS / 2 + 1);
    let error = dispatcher.dispatch(&[PresentationEvent::TextComposition {
        phase: CompositionPhase::Updated,
        text: oversized.into_boxed_str(),
    }]);
    let Err(ActionDispatchError::CompositionTooLong { actual, limit }) = error else {
        panic!("unexpected composition error");
    };
    assert_eq!(actual, MAX_COMPOSITION_UNITS + 2);
    assert_eq!(limit, MAX_COMPOSITION_UNITS);
}

#[test]
fn non_pointer_actions_preserve_all_values() {
    let mut dispatcher = PresentationDispatcher::new();
    let actions = dispatcher
        .dispatch(&[
            PresentationEvent::FocusGained,
            PresentationEvent::KeyDown {
                virtual_key: 0x41,
                repeated: true,
            },
            PresentationEvent::KeyUp { virtual_key: 0x41 },
            PresentationEvent::TextInput { character: '中' },
            PresentationEvent::TextComposition {
                phase: CompositionPhase::Committed,
                text: "A😀".into(),
            },
            PresentationEvent::Resized {
                width: 800,
                height: 600,
            },
            PresentationEvent::DpiChanged { dpi: 144 },
            PresentationEvent::CloseRequested,
            PresentationEvent::Destroyed,
        ])
        .expect("lifecycle and input actions");
    assert_eq!(
        actions.as_ref(),
        &[
            ViewerAction::FocusChanged { focused: true },
            ViewerAction::KeyPressed {
                virtual_key: 0x41,
                repeated: true,
            },
            ViewerAction::KeyReleased { virtual_key: 0x41 },
            ViewerAction::TextInput { character: '中' },
            ViewerAction::TextComposition {
                phase: CompositionPhase::Committed,
                text: "A😀".into(),
            },
            ViewerAction::Resized {
                width: 800,
                height: 600,
            },
            ViewerAction::DpiChanged { dpi: 144 },
            ViewerAction::CloseRequested,
            ViewerAction::Destroyed,
        ]
    );
}

#[test]
fn event_batch_bound_is_enforced_before_state_changes() {
    let mut dispatcher = PresentationDispatcher::new();
    let events = vec![PresentationEvent::FocusGained; MAX_PRESENTATION_EVENTS + 1];
    let error = dispatcher.dispatch(&events).expect_err("oversized batch");
    let ActionDispatchError::BatchTooLarge { actual, limit } = error else {
        panic!("unexpected batch error");
    };
    assert_eq!(actual, MAX_PRESENTATION_EVENTS + 1);
    assert_eq!(limit, MAX_PRESENTATION_EVENTS);
    let actions = dispatcher
        .dispatch(&[PresentationEvent::FocusLost])
        .expect("dispatcher remains unchanged");
    assert_eq!(
        actions.as_ref(),
        &[ViewerAction::FocusChanged { focused: false }]
    );
}

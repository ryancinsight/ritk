//! Event-to-action reduction operations for the presentation dispatcher.

use super::*;

impl PresentationDispatcher {
    pub(super) fn apply_event(
        &mut self,
        event: &PresentationEvent,
        actions: &mut Vec<ViewerAction>,
        action_limit: usize,
    ) -> Result<(), ActionDispatchError> {
        match event {
            PresentationEvent::CloseRequested => {
                self.clear_pointers();
                Self::push_action(actions, ViewerAction::CloseRequested, action_limit)
            }
            PresentationEvent::Destroyed => {
                self.clear_pointers();
                Self::push_action(actions, ViewerAction::Destroyed, action_limit)
            }
            PresentationEvent::FocusGained => Self::push_action(
                actions,
                ViewerAction::FocusChanged { focused: true },
                action_limit,
            ),
            PresentationEvent::FocusLost => {
                Self::push_action(
                    actions,
                    ViewerAction::FocusChanged { focused: false },
                    action_limit,
                )?;
                for button in POINTER_BUTTONS {
                    if let Some(press) = self.pointers[button_slot(button)] {
                        Self::push_action(
                            actions,
                            ViewerAction::PointerCancelled {
                                button,
                                position: press.last,
                            },
                            action_limit,
                        )?;
                    }
                }
                self.clear_pointers();
                Ok(())
            }
            PresentationEvent::PointerMove { x, y } => {
                let position = viewport_point(*x, *y)?;
                Self::push_action(
                    actions,
                    ViewerAction::PointerMoved { position },
                    action_limit,
                )?;
                for button in POINTER_BUTTONS {
                    let slot = button_slot(button);
                    let Some(mut press) = self.pointers[slot] else {
                        continue;
                    };
                    let delta = press.last.delta_to(position)?;
                    press.last = position;
                    if !delta.is_zero() {
                        press.moved = true;
                        Self::push_action(
                            actions,
                            ViewerAction::PointerDragged {
                                button,
                                start: press.origin,
                                current: position,
                                delta,
                            },
                            action_limit,
                        )?;
                    }
                    self.pointers[slot] = Some(press);
                }
                Ok(())
            }
            PresentationEvent::PointerDown { x, y, button } => {
                let slot = button_slot(*button);
                if self.pointers[slot].is_some() {
                    return Err(ActionDispatchError::DuplicatePointerPress { button: *button });
                }
                let position = viewport_point(*x, *y)?;
                self.pointers[slot] = Some(PointerPress {
                    origin: position,
                    last: position,
                    moved: false,
                });
                Self::push_action(
                    actions,
                    ViewerAction::PointerPressed {
                        button: *button,
                        position,
                    },
                    action_limit,
                )
            }
            PresentationEvent::PointerUp { x, y, button } => {
                let slot = button_slot(*button);
                let Some(press) = self.pointers[slot].take() else {
                    return Err(ActionDispatchError::PointerReleaseWithoutPress {
                        button: *button,
                    });
                };
                let position = viewport_point(*x, *y)?;
                let gesture = if press.moved || position != press.origin {
                    PointerGesture::Drag
                } else {
                    PointerGesture::Click
                };
                Self::push_action(
                    actions,
                    ViewerAction::PointerReleased {
                        button: *button,
                        position,
                        gesture,
                    },
                    action_limit,
                )
            }
            PresentationEvent::KeyDown {
                virtual_key,
                repeated,
            } => Self::push_action(
                actions,
                ViewerAction::KeyPressed {
                    virtual_key: *virtual_key,
                    repeated: *repeated,
                },
                action_limit,
            ),
            PresentationEvent::KeyUp { virtual_key } => Self::push_action(
                actions,
                ViewerAction::KeyReleased {
                    virtual_key: *virtual_key,
                },
                action_limit,
            ),
            PresentationEvent::TextInput { character } => Self::push_action(
                actions,
                ViewerAction::TextInput {
                    character: *character,
                },
                action_limit,
            ),
            PresentationEvent::TextComposition { phase, text } => {
                let units = text.encode_utf16().count();
                if units > MAX_COMPOSITION_UNITS {
                    return Err(ActionDispatchError::CompositionTooLong {
                        actual: units,
                        limit: MAX_COMPOSITION_UNITS,
                    });
                }
                let mut owned = String::new();
                owned.try_reserve_exact(text.len()).map_err(|_| {
                    ActionDispatchError::AllocationFailure {
                        requested: text.len(),
                    }
                })?;
                owned.push_str(text);
                Self::push_action(
                    actions,
                    ViewerAction::TextComposition {
                        phase: *phase,
                        text: owned.into_boxed_str(),
                    },
                    action_limit,
                )
            }
            PresentationEvent::Resized { width, height } => Self::push_action(
                actions,
                ViewerAction::Resized {
                    width: *width,
                    height: *height,
                },
                action_limit,
            ),
            PresentationEvent::DpiChanged { dpi } => Self::push_action(
                actions,
                ViewerAction::DpiChanged { dpi: *dpi },
                action_limit,
            ),
        }
    }

    fn push_action(
        actions: &mut Vec<ViewerAction>,
        action: ViewerAction,
        action_limit: usize,
    ) -> Result<(), ActionDispatchError> {
        if actions.len() >= action_limit {
            return Err(ActionDispatchError::ActionBatchTooLarge {
                actual: actions.len(),
                limit: action_limit,
            });
        }
        actions.push(action);
        Ok(())
    }

    fn clear_pointers(&mut self) {
        self.pointers = [None; POINTER_BUTTONS.len()];
    }
}

fn button_slot(button: PointerButton) -> usize {
    match button {
        PointerButton::Left => 0,
        PointerButton::Right => 1,
        PointerButton::Middle => 2,
        PointerButton::X1 => 3,
        PointerButton::X2 => 4,
    }
}

fn viewport_point(x: f32, y: f32) -> Result<ViewportPoint, ActionDispatchError> {
    let point = ViewportPoint::new(x, y);
    if point.is_finite() {
        Ok(point)
    } else {
        Err(ActionDispatchError::NonFiniteCoordinate { point })
    }
}

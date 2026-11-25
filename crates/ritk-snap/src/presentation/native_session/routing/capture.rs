//! Pointer capture and recovery within a routed batch.
use super::{view_at, EventRoute, NativeViewport, PresentationEvent, PressedButtons};
/// The pane receiving pointer input and the buttons holding it there.
///
/// The first press takes the capture. Further presses join it, and it ends
/// only when the last held button is released, so every press and release of a
/// multi-button sequence reaches the same pane.
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::presentation::native_session) struct PointerCapture {
    pub(super) view: Option<usize>,
    pub(super) pressed: PressedButtons,
}

impl PointerCapture {
    /// A second press of a held button proves its release never arrived, so
    /// that button stops holding the capture. Another held button keeps the
    /// pane, and the press joins it; otherwise the press hit-tests afresh.
    pub(super) fn forget_lost_release(&mut self, event: &PresentationEvent) {
        if let PresentationEvent::PointerDown { button, .. } = event
            && self.pressed.contains(*button)
        {
            self.pressed.remove(*button);
            if self.pressed.is_empty() {
                self.view = None;
            }
        }
    }

    /// The pane still holding a gesture that this press has moved elsewhere.
    pub(super) fn lost_to(&self, event: &PresentationEvent, route: EventRoute) -> Option<usize> {
        match event {
            PresentationEvent::PointerDown { button, .. }
                if self.pressed.contains(*button) && route.view() != self.view =>
            {
                self.view
            }
            _ => None,
        }
    }

    pub(super) fn discards(&self, event: &PresentationEvent, viewports: &[NativeViewport]) -> bool {
        match event {
            PresentationEvent::PointerDown { x, y, .. } => {
                self.view.is_none() && view_at(viewports, *x, *y).is_none()
            }
            PresentationEvent::PointerUp { button, .. }
            | PresentationEvent::PointerCancel { button, .. } => !self.pressed.contains(*button),
            _ => false,
        }
    }

    pub(super) fn update(&mut self, event: &PresentationEvent, viewports: &[NativeViewport]) {
        self.forget_lost_release(event);
        match event {
            PresentationEvent::PointerDown { x, y, button, .. } => {
                if self.view.is_none() {
                    self.pressed = PressedButtons::EMPTY;
                    self.view = view_at(viewports, *x, *y);
                }
                if self.view.is_some() {
                    self.pressed.insert(*button);
                }
            }
            PresentationEvent::PointerUp { button, .. }
            | PresentationEvent::PointerCancel { button, .. } => {
                self.pressed.remove(*button);
                if self.pressed.is_empty() {
                    self.view = None;
                }
            }
            PresentationEvent::FocusLost
            | PresentationEvent::PointerCaptureLost
            | PresentationEvent::CloseRequested
            | PresentationEvent::Destroyed => {
                self.view = None;
                self.pressed = PressedButtons::EMPTY;
            }
            _ => {}
        }
    }
}

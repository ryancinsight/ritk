//! Classifies pointer input for pane routing and activation.

use crate::presentation::PresentationEvent;

/// Whether the event can activate its pane. A release keeps the pane chosen
/// at press time, even when a window action has moved the pane.
pub(in crate::presentation::native_session) fn activates_panel(event: &PresentationEvent) -> bool {
    matches!(
        event,
        PresentationEvent::PointerDown { .. }
            | PresentationEvent::PointerDoubleClick { .. }
            | PresentationEvent::PointerMove { .. }
            | PresentationEvent::PointerWheel { .. }
    )
}

pub(in crate::presentation::native_session) fn is_pointer_event(event: &PresentationEvent) -> bool {
    activates_panel(event)
        || matches!(
            event,
            PresentationEvent::PointerUp { .. }
                | PresentationEvent::PointerCancel { .. }
                | PresentationEvent::PointerCaptureLost
        )
}

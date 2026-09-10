//! Panel selection for the native presentation event stream.

use super::{NativeViewerError, NativeViewerSession};
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::PresentationEvent;

impl NativeViewerSession {
    fn view_at(&self, x: f64, y: f64) -> Option<usize> {
        self.viewports
            .iter()
            .position(|viewport| viewport.contains(x, y))
    }

    fn event_view_index(&self, events: &[PresentationEvent]) -> Option<usize> {
        for event in events {
            let candidate = match event {
                PresentationEvent::PointerDown { x, y, .. } => self.view_at(*x, *y),
                PresentationEvent::PointerMove { .. }
                | PresentationEvent::PointerUp { .. }
                | PresentationEvent::PointerWheel { .. } => self.active_view.or_else(|| {
                    let (x, y) = match event {
                        PresentationEvent::PointerMove { x, y }
                        | PresentationEvent::PointerUp { x, y, .. }
                        | PresentationEvent::PointerWheel { x, y, .. } => (*x, *y),
                        _ => return None,
                    };
                    self.view_at(x, y)
                }),
                _ => None,
            };
            if candidate.is_some() {
                return candidate;
            }
        }
        self.viewports
            .iter()
            .position(|viewport| viewport.axis() == self.app.axis)
    }

    pub(super) fn apply_events(
        &mut self,
        events: &[PresentationEvent],
    ) -> std::result::Result<ViewerActionDisposition, NativeViewerError> {
        let previous_axis = self.app.axis;
        let previous_active_view = self.active_view;
        let selected_view = self.event_view_index(events);
        if let Some(index) = selected_view {
            self.app.axis = self.viewports[index].axis();
            if events
                .iter()
                .any(|event| matches!(event, PresentationEvent::PointerDown { .. }))
            {
                self.active_view = Some(index);
            }
        }
        let viewport = (!self.minimized)
            .then_some(selected_view.unwrap_or(0))
            .and_then(|index| self.viewports.get(index))
            .map(|viewport| viewport.mapping());
        let result = self
            .app
            .apply_presentation_events(events, viewport.as_ref());
        let disposition = match result {
            Ok(disposition) => disposition,
            Err(error) => {
                self.app.axis = previous_axis;
                self.active_view = previous_active_view;
                return Err(NativeViewerError::new(format!(
                    "apply RITK presentation events: {error}"
                )));
            }
        };
        if events.iter().any(|event| {
            matches!(
                event,
                PresentationEvent::PointerUp { .. }
                    | PresentationEvent::FocusLost
                    | PresentationEvent::CloseRequested
                    | PresentationEvent::Destroyed
            )
        }) || matches!(disposition, ViewerActionDisposition::Exit)
        {
            self.active_view = None;
        }
        Ok(disposition)
    }
}

//! Panel selection and event routing for the native presentation surface.

use super::{NativeViewerError, NativeViewerSession};
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::{PointerButton, PresentationEvent, ViewportPoint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedPanel {
    Orthogonal(usize),
    Oblique,
}

impl NativeViewerSession {
    fn panel_at(&self, x: f64, y: f64) -> Option<SelectedPanel> {
        self.viewports
            .iter()
            .position(|viewport| viewport.contains(x, y))
            .map(SelectedPanel::Orthogonal)
            .or_else(|| {
                self.oblique_viewport
                    .and_then(|viewport| viewport.map(ViewportPoint::new(x, y)))
                    .map(|_| SelectedPanel::Oblique)
            })
    }

    fn event_panel(&self, event: &PresentationEvent) -> Option<SelectedPanel> {
        match event {
            PresentationEvent::PointerDown { x, y, .. } => self.panel_at(*x, *y),
            PresentationEvent::PointerMove { x, y } => {
                if self.ignored_pointer_capture {
                    None
                } else if self.active_oblique {
                    Some(SelectedPanel::Oblique)
                } else if let Some(index) = self.active_view {
                    Some(SelectedPanel::Orthogonal(index))
                } else {
                    self.panel_at(*x, *y)
                }
            }
            PresentationEvent::PointerUp { x, y, .. }
            | PresentationEvent::PointerCancel { x, y, .. } => {
                if self.ignored_pointer_capture {
                    None
                } else if self.active_oblique {
                    Some(SelectedPanel::Oblique)
                } else if let Some(index) = self.active_view {
                    Some(SelectedPanel::Orthogonal(index))
                } else {
                    self.panel_at(*x, *y)
                }
            }
            PresentationEvent::PointerWheel { x, y, .. } => {
                if self.ignored_pointer_capture {
                    None
                } else if self.active_oblique {
                    Some(SelectedPanel::Oblique)
                } else if let Some(index) = self.active_view {
                    Some(SelectedPanel::Orthogonal(index))
                } else {
                    self.panel_at(*x, *y)
                }
            }
            PresentationEvent::KeyDown { .. } | PresentationEvent::KeyUp { .. }
                if self.selected_oblique =>
            {
                Some(SelectedPanel::Oblique)
            }
            PresentationEvent::KeyDown { .. } | PresentationEvent::KeyUp { .. } => {
                self.default_panel()
            }
            PresentationEvent::FocusGained
            | PresentationEvent::FocusLost
            | PresentationEvent::CloseRequested
            | PresentationEvent::Destroyed
            | PresentationEvent::AccessibilityAction { .. }
            | PresentationEvent::TextInput { .. }
            | PresentationEvent::TextComposition { .. }
            | PresentationEvent::Resized { .. }
            | PresentationEvent::DpiChanged { .. } => self.default_panel(),
        }
    }

    fn default_panel(&self) -> Option<SelectedPanel> {
        if self.selected_oblique {
            Some(SelectedPanel::Oblique)
        } else {
            self.active_view.map(SelectedPanel::Orthogonal).or_else(|| {
                self.viewports
                    .iter()
                    .position(|viewport| viewport.axis() == self.app.axis)
                    .map(SelectedPanel::Orthogonal)
            })
        }
    }

    pub(super) fn apply_events(
        &mut self,
        events: &[PresentationEvent],
    ) -> std::result::Result<ViewerActionDisposition, NativeViewerError> {
        let mut staged_dispatcher = self.app.presentation_dispatcher.clone();
        staged_dispatcher
            .dispatch(events)
            .map_err(|error| NativeViewerError::new(format!("preflight native events: {error}")))?;
        for event in events {
            let button = match event {
                PresentationEvent::PointerDown { button, .. }
                | PresentationEvent::PointerUp { button, .. }
                | PresentationEvent::PointerCancel { button, .. } => Some(*button),
                _ => None,
            };
            if button.is_some_and(|button| button != PointerButton::Left) {
                return Err(NativeViewerError::new(
                    "native viewer interaction supports only the primary pointer button",
                ));
            }
        }

        let previous_axis = self.app.axis;
        let previous_active_view = self.active_view;
        let previous_selected_oblique = self.selected_oblique;
        let previous_active_oblique = self.active_oblique;
        let previous_ignored_pointer_capture = self.ignored_pointer_capture;
        let previous_oblique_gesture = self.oblique_gesture;
        let viewport_snapshot = self.oblique_viewport;
        let plane_snapshot = self.oblique.as_ref().and_then(|view| view.plane);
        if let (Some(viewport), Some(plane)) = (viewport_snapshot, plane_snapshot) {
            viewport
                .validate_dimensions(plane.dimensions())
                .map_err(|error| {
                    NativeViewerError::new(format!("validate oblique input: {error}"))
                })?;
            if let Some(volume) = self.app.loaded.as_ref() {
                plane.validate_source(volume).map_err(|error| {
                    NativeViewerError::new(format!("validate oblique source: {error}"))
                })?;
            }
        }

        let mut repaint = false;
        let mut exit = false;
        for event in events {
            let panel = self.event_panel(event);
            if matches!(event, PresentationEvent::PointerDown { .. }) {
                self.ignored_pointer_capture = panel.is_none();
                self.active_view = None;
                self.active_oblique = false;
                self.oblique_gesture = None;
            }
            if let Some(panel) = panel {
                match panel {
                    SelectedPanel::Orthogonal(index) => {
                        self.app.axis = self.viewports[index].axis();
                        self.selected_oblique = false;
                        self.active_oblique = false;
                        if matches!(event, PresentationEvent::PointerDown { .. }) {
                            self.active_view = Some(index);
                        }
                    }
                    SelectedPanel::Oblique => {
                        self.selected_oblique = true;
                        if matches!(event, PresentationEvent::PointerDown { .. }) {
                            self.active_oblique = true;
                        }
                    }
                }
            }

            let event_slice = std::slice::from_ref(event);
            let applied = match panel {
                Some(SelectedPanel::Orthogonal(index)) => {
                    let viewport = if self.minimized {
                        None
                    } else {
                        self.viewports.get(index).map(|item| item.mapping())
                    };
                    self.app
                        .apply_presentation_events(event_slice, viewport.as_ref())
                        .map_err(|error| {
                            NativeViewerError::new(format!(
                                "apply RITK presentation event: {error}"
                            ))
                        })
                }
                Some(SelectedPanel::Oblique) => match (viewport_snapshot, plane_snapshot) {
                    (Some(viewport), Some(plane)) if !self.minimized => self
                        .app
                        .apply_oblique_presentation_events(event_slice, &viewport, &plane)
                        .map_err(|error| {
                            NativeViewerError::new(format!(
                                "apply RITK oblique presentation event: {error}"
                            ))
                        }),
                    _ => self
                        .app
                        .apply_presentation_events(event_slice, None)
                        .map_err(|error| {
                            NativeViewerError::new(format!(
                                "apply RITK presentation event: {error}"
                            ))
                        }),
                },
                None => self
                    .app
                    .apply_presentation_events(event_slice, None)
                    .map_err(|error| {
                        NativeViewerError::new(format!("apply RITK presentation event: {error}"))
                    }),
            };
            match applied {
                Err(error) => {
                    return self.restore_route_after_error(
                        previous_axis,
                        previous_active_view,
                        previous_selected_oblique,
                        previous_active_oblique,
                        previous_ignored_pointer_capture,
                        previous_oblique_gesture,
                        error,
                    );
                }
                Ok(ViewerActionDisposition::Continue { repaint: needed }) => repaint |= needed,
                Ok(ViewerActionDisposition::Exit) => {
                    exit = true;
                    break;
                }
            }

            if panel == Some(SelectedPanel::Oblique) {
                match self.reduce_oblique_events(event_slice, self.selected_oblique) {
                    Ok(needed) => repaint |= needed,
                    Err(error) => {
                        return self.restore_route_after_error(
                            previous_axis,
                            previous_active_view,
                            previous_selected_oblique,
                            previous_active_oblique,
                            previous_ignored_pointer_capture,
                            previous_oblique_gesture,
                            NativeViewerError::from(error),
                        );
                    }
                }
            }

            if matches!(
                event,
                PresentationEvent::PointerUp { .. }
                    | PresentationEvent::PointerCancel { .. }
                    | PresentationEvent::FocusLost
                    | PresentationEvent::CloseRequested
                    | PresentationEvent::Destroyed
            ) {
                self.active_view = None;
                self.active_oblique = false;
                self.ignored_pointer_capture = false;
                self.oblique_gesture = None;
            }
        }

        if exit {
            return Ok(ViewerActionDisposition::Exit);
        }
        Ok(ViewerActionDisposition::Continue { repaint })
    }

    fn restore_route_after_error(
        &mut self,
        axis: usize,
        active_view: Option<usize>,
        selected_oblique: bool,
        active_oblique: bool,
        ignored_pointer_capture: bool,
        oblique_gesture: Option<super::session::ObliqueGesture>,
        error: NativeViewerError,
    ) -> std::result::Result<ViewerActionDisposition, NativeViewerError> {
        self.app.axis = axis;
        self.active_view = active_view;
        self.selected_oblique = selected_oblique;
        self.active_oblique = active_oblique;
        self.ignored_pointer_capture = ignored_pointer_capture;
        self.oblique_gesture = oblique_gesture;
        Err(error)
    }
}

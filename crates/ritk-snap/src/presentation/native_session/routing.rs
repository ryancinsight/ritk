//! Panel selection for the native presentation event stream.

use super::layout::{NativeViewport, WorkspaceLayout, MAX_GRID_PANELS};
use super::panels::PanelRouteMap;
use super::pressed_buttons::PressedButtons;
use super::{NativeViewerError, NativeViewerSession};
use crate::app::action_adapter::{preflight_presentation_events, ViewerActionDisposition};
use crate::app::viewer_viewport::ViewerViewport;
use crate::presentation::PresentationEvent;
use arrayvec::ArrayVec;

/// Where one host event is delivered within a native batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EventRoute {
    /// The pane at this index of the batch's framebuffer.
    View(usize),
    /// The active panel, for input that names no pane.
    ActivePanel,
    /// Nothing: a press outside every pane or a release no pane holds.
    Discarded,
}

impl EventRoute {
    pub(super) const fn view(self) -> Option<usize> {
        match self {
            Self::View(index) => Some(index),
            Self::ActivePanel | Self::Discarded => None,
        }
    }
}

/// Input routes bound to the framebuffer visible when the host batch begins.
pub(super) struct RoutedBatch {
    pub(super) layout: WorkspaceLayout,
    routes: ArrayVec<EventRoute, { crate::presentation::MAX_PRESENTATION_EVENTS }>,
    view_mappings: ArrayVec<ViewerViewport, MAX_GRID_PANELS>,
}

impl RoutedBatch {
    pub(super) fn routes(&self) -> &[EventRoute] {
        &self.routes
    }

    pub(super) fn view_mappings(&self) -> &[ViewerViewport] {
        &self.view_mappings
    }
}

/// The pane receiving pointer input and the buttons holding it there.
///
/// The first press takes the capture. Further presses join it, and it ends
/// only when the last held button is released, so every press and release of a
/// multi-button sequence reaches the same pane.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PointerCapture {
    view: Option<usize>,
    pressed: PressedButtons,
}

impl PointerCapture {
    /// A second press of a held button proves its release never arrived, so
    /// that button stops holding the capture. Another held button keeps the
    /// pane, and the press joins it; otherwise the press hit-tests afresh.
    fn forget_lost_release(&mut self, event: &PresentationEvent) {
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
    fn lost_to(&self, event: &PresentationEvent, route: EventRoute) -> Option<usize> {
        match event {
            PresentationEvent::PointerDown { button, .. }
                if self.pressed.contains(*button) && route.view() != self.view =>
            {
                self.view
            }
            _ => None,
        }
    }

    fn discards(&self, event: &PresentationEvent, viewports: &[NativeViewport]) -> bool {
        match event {
            PresentationEvent::PointerDown { x, y, .. } => {
                self.view.is_none() && view_at(viewports, *x, *y).is_none()
            }
            PresentationEvent::PointerUp { button, .. }
            | PresentationEvent::PointerCancel { button, .. } => !self.pressed.contains(*button),
            _ => false,
        }
    }

    fn update(&mut self, event: &PresentationEvent, viewports: &[NativeViewport]) {
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
            | PresentationEvent::CloseRequested
            | PresentationEvent::Destroyed => {
                self.view = None;
                self.pressed = PressedButtons::EMPTY;
            }
            _ => {}
        }
    }
}

/// Tracks pointer capture and keyboard fallback within one host event batch.
pub(super) struct PanelEventRouter {
    panel: usize,
    capture: PointerCapture,
    axis: usize,
}

impl PanelEventRouter {
    pub(super) const fn new(
        active_panel: usize,
        capture: PointerCapture,
        active_axis: usize,
    ) -> Self {
        Self {
            panel: active_panel,
            capture,
            axis: active_axis,
        }
    }

    pub(super) const fn active_panel(&self) -> usize {
        self.panel
    }

    pub(super) fn route(
        &mut self,
        event: &PresentationEvent,
        layout: WorkspaceLayout,
        viewports: &[NativeViewport],
    ) -> EventRoute {
        self.capture.forget_lost_release(event);
        if self.capture.discards(event, viewports) {
            return EventRoute::Discarded;
        }
        let selected_view = event_view_index(
            event,
            self.capture.view,
            self.panel,
            self.axis,
            layout,
            viewports,
        );
        if let Some(index) = selected_view {
            if layout.is_grid() {
                self.panel = index;
            }
            if let Some(viewport) = viewports.get(index) {
                self.axis = viewport.axis();
            }
        }
        self.capture.update(event, viewports);
        selected_view.map_or(EventRoute::ActivePanel, EventRoute::View)
    }
}

fn event_view_index(
    event: &PresentationEvent,
    active_view: Option<usize>,
    active_panel: usize,
    active_axis: usize,
    layout: WorkspaceLayout,
    viewports: &[NativeViewport],
) -> Option<usize> {
    let candidate = match event {
        PresentationEvent::PointerDown { x, y, .. }
        | PresentationEvent::PointerMove { x, y }
        | PresentationEvent::PointerUp { x, y, .. }
        | PresentationEvent::PointerCancel { x, y, .. }
        | PresentationEvent::PointerWheel { x, y, .. } => {
            active_view.or_else(|| view_at(viewports, *x, *y))
        }
        _ => None,
    };
    candidate.or_else(|| {
        if layout.is_grid() {
            viewports.get(active_panel).map(|_| active_panel)
        } else {
            viewports
                .iter()
                .position(|viewport| viewport.axis() == active_axis)
        }
    })
}

fn view_at(viewports: &[NativeViewport], x: f64, y: f64) -> Option<usize> {
    viewports
        .iter()
        .position(|viewport| viewport.contains(x, y))
}

impl NativeViewerSession {
    pub(super) const fn pointer_capture(&self) -> PointerCapture {
        PointerCapture {
            view: self.active_view,
            pressed: self.captured_buttons,
        }
    }

    fn app_for_panel_mut(
        &mut self,
        panel_index: usize,
    ) -> std::result::Result<&mut crate::app::SnapApp, NativeViewerError> {
        if panel_index > 0 {
            self.compare_panels
                .get_mut(panel_index - 1)
                .map(|panel| &mut panel.app)
                .ok_or_else(|| NativeViewerError::new("series panel state is missing"))
        } else {
            Ok(&mut self.app)
        }
    }

    pub(super) fn prepare_event_routes(
        &self,
        events: &[PresentationEvent],
    ) -> std::result::Result<RoutedBatch, NativeViewerError> {
        if events.len() > crate::presentation::MAX_PRESENTATION_EVENTS {
            return Err(NativeViewerError::new(format!(
                "apply RITK presentation events: {}",
                crate::presentation::ActionDispatchError::BatchTooLarge {
                    actual: events.len(),
                    limit: crate::presentation::MAX_PRESENTATION_EVENTS,
                }
            )));
        }

        let grid = self.workspace_layout.grid();
        let panel_count = grid.map_or(1, |layout| layout.panel_count());
        let mut dispatchers = ArrayVec::<_, { super::layout::MAX_GRID_PANELS }>::new();
        dispatchers
            .try_push(self.app.presentation_dispatcher.clone())
            .map_err(|_| NativeViewerError::new("viewer event dispatcher capacity exceeded"))?;
        for panel in self
            .compare_panels
            .iter()
            .take(panel_count.saturating_sub(1))
        {
            dispatchers
                .try_push(panel.app.presentation_dispatcher.clone())
                .map_err(|_| NativeViewerError::new("viewer event dispatcher capacity exceeded"))?;
        }
        if dispatchers.len() != panel_count {
            return Err(NativeViewerError::new(
                "series panel event dispatcher is missing",
            ));
        }

        let mut routes = ArrayVec::<_, { crate::presentation::MAX_PRESENTATION_EVENTS }>::new();
        let mut view_mappings = ArrayVec::<_, MAX_GRID_PANELS>::new();
        for viewport in &self.viewports {
            view_mappings
                .try_push(viewport.mapping())
                .map_err(|_| NativeViewerError::new("viewer viewport mapping capacity exceeded"))?;
        }
        let mut route_cursor = PanelEventRouter::new(
            self.active_panel,
            self.pointer_capture(),
            self.active_app().axis,
        );
        for event in events {
            let route = route_cursor.route(event, self.workspace_layout, &self.viewports);
            if route == EventRoute::Discarded {
                routes
                    .try_push(route)
                    .map_err(|_| NativeViewerError::new("viewer event route capacity exceeded"))?;
                continue;
            }
            let dispatcher_index = if grid.is_some() {
                route
                    .view()
                    .unwrap_or(route_cursor.active_panel())
                    .min(panel_count.saturating_sub(1))
            } else {
                0
            };
            let dispatcher = dispatchers
                .get_mut(dispatcher_index)
                .ok_or_else(|| NativeViewerError::new("viewer event route is outside its grid"))?;
            preflight_presentation_events(dispatcher, std::slice::from_ref(event)).map_err(
                |error| NativeViewerError::new(format!("apply RITK presentation events: {error}")),
            )?;
            routes
                .try_push(route)
                .map_err(|_| NativeViewerError::new("viewer event route capacity exceeded"))?;
        }
        Ok(RoutedBatch {
            layout: self.workspace_layout,
            routes,
            view_mappings,
        })
    }

    pub(super) fn apply_events(
        &mut self,
        events: &[PresentationEvent],
        routes: &[EventRoute],
        panel_routes: &PanelRouteMap,
        view_mappings: &[ViewerViewport],
        layout: WorkspaceLayout,
    ) -> std::result::Result<ViewerActionDisposition, NativeViewerError> {
        if events.len() != routes.len() {
            return Err(NativeViewerError::new(
                "prepared native event routes do not match the event segment",
            ));
        }
        let grid = layout.grid();
        let previous_panel = self.active_panel;
        let previous_active_view = self.active_view;
        let mut capture = self.pointer_capture();
        let mut repaint = false;
        let mut start = 0;
        while start < events.len() {
            let route = routes[start];
            if let Some(view) = capture.lost_to(&events[start], route) {
                let stale_panel = self.panel_for_event(
                    &events[start],
                    EventRoute::View(view),
                    panel_routes,
                    grid.is_some(),
                );
                if let Some(panel) =
                    stale_panel.filter(|index| panel_routes.contains_current_panel(*index))
                {
                    self.app_for_panel_mut(panel)?.cancel_presentation_gesture();
                }
            }
            let Some(panel_index) = self
                .panel_for_event(&events[start], route, panel_routes, grid.is_some())
                .filter(|index| {
                    !is_pointer_event(&events[start]) || panel_routes.contains_current_panel(*index)
                })
            else {
                capture.update(&events[start], &self.viewports);
                start += 1;
                continue;
            };
            let view_index =
                self.event_view_mapping(&events[start], route, panel_index, grid.is_some());
            let mut end = start + 1;
            while end < events.len() {
                let Some(next_panel) = self
                    .panel_for_event(&events[end], routes[end], panel_routes, grid.is_some())
                    .filter(|index| {
                        !is_pointer_event(&events[end])
                            || panel_routes.contains_current_panel(*index)
                    })
                else {
                    break;
                };
                let next_view =
                    self.event_view_mapping(&events[end], routes[end], next_panel, grid.is_some());
                if next_panel != panel_index || next_view != view_index {
                    break;
                }
                end += 1;
            }
            if grid.is_some()
                && self
                    .workspace_layout
                    .grid()
                    .is_some_and(|current| panel_index < current.panel_count())
                && route.view().is_some()
            {
                self.active_panel = panel_index;
            }
            let viewport = if self.minimized {
                None
            } else {
                view_index
                    .and_then(|index| view_mappings.get(index))
                    .copied()
            };
            let (result, previous_axis) = {
                let app = self.app_for_panel_mut(panel_index)?;
                let previous_axis = app.axis;
                if let Some(viewport) = viewport {
                    app.axis = viewport.axis();
                }
                (
                    app.apply_presentation_events(&events[start..end], viewport.as_ref()),
                    previous_axis,
                )
            };
            let disposition = match result {
                Ok(disposition) => disposition,
                Err(error) => {
                    self.app_for_panel_mut(panel_index)?.axis = previous_axis;
                    self.active_panel = previous_panel;
                    self.active_view = previous_active_view;
                    return Err(NativeViewerError::new(format!(
                        "apply RITK presentation events: {error}"
                    )));
                }
            };
            match disposition {
                ViewerActionDisposition::Continue {
                    repaint: event_repaint,
                } => repaint |= event_repaint,
                ViewerActionDisposition::Exit => {
                    self.active_view = None;
                    return Ok(ViewerActionDisposition::Exit);
                }
            }
            for event in &events[start..end] {
                capture.update(event, &self.viewports);
            }
            start = end;
        }
        self.active_view = capture.view;
        self.captured_buttons = capture.pressed;
        Ok(ViewerActionDisposition::Continue { repaint })
    }

    fn panel_for_event(
        &self,
        event: &PresentationEvent,
        route: EventRoute,
        panel_routes: &PanelRouteMap,
        grid: bool,
    ) -> Option<usize> {
        if route == EventRoute::Discarded {
            return None;
        }
        if !grid {
            return Some(0);
        }
        if is_pointer_event(event) {
            route.view().map_or(Some(self.active_panel), |view| {
                panel_routes.current_panel(view)
            })
        } else {
            Some(self.active_panel)
        }
    }

    fn event_view_mapping(
        &self,
        event: &PresentationEvent,
        route: EventRoute,
        panel_index: usize,
        grid: bool,
    ) -> Option<usize> {
        if grid && !is_pointer_event(event) {
            Some(panel_index)
        } else {
            route.view()
        }
    }
}

pub(super) fn is_pointer_event(event: &PresentationEvent) -> bool {
    matches!(
        event,
        PresentationEvent::PointerDown { .. }
            | PresentationEvent::PointerMove { .. }
            | PresentationEvent::PointerUp { .. }
            | PresentationEvent::PointerCancel { .. }
            | PresentationEvent::PointerWheel { .. }
    )
}

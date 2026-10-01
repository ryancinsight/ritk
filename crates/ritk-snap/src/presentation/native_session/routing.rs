//! Panel selection for the native presentation event stream.

use super::layout::{WorkspaceLayout, MAX_GRID_PANELS};
use super::panels::PanelRouteMap;
use super::{NativeViewerError, NativeViewerSession};
use crate::app::action_adapter::{preflight_presentation_events, ViewerActionDisposition};
use crate::app::viewer_viewport::ViewerViewport;
use crate::presentation::PresentationEvent;
use arrayvec::ArrayVec;

/// Input routes bound to the framebuffer visible when the host batch begins.
pub(super) struct RoutedBatch {
    pub(super) layout: WorkspaceLayout,
    pub(super) active_view: Option<usize>,
    routes: ArrayVec<Option<usize>, { crate::presentation::MAX_PRESENTATION_EVENTS }>,
    view_mappings: ArrayVec<ViewerViewport, MAX_GRID_PANELS>,
}

impl RoutedBatch {
    pub(super) fn routes(&self) -> &[Option<usize>] {
        &self.routes
    }

    pub(super) fn view_mappings(&self) -> &[ViewerViewport] {
        &self.view_mappings
    }
}

/// Tracks pointer capture and keyboard fallback within one host event batch.
pub(super) struct PanelEventRouter {
    active_panel: usize,
    active_view: Option<usize>,
    active_axis: usize,
}

impl PanelEventRouter {
    pub(super) const fn new(
        active_panel: usize,
        active_view: Option<usize>,
        active_axis: usize,
    ) -> Self {
        Self {
            active_panel,
            active_view,
            active_axis,
        }
    }

    pub(super) const fn active_panel(&self) -> usize {
        self.active_panel
    }

    pub(super) const fn active_view(&self) -> Option<usize> {
        self.active_view
    }

    pub(super) fn route(
        &mut self,
        event: &PresentationEvent,
        layout: WorkspaceLayout,
        viewports: &[super::layout::NativeViewport],
    ) -> Option<usize> {
        let selected_view = event_view_index(
            event,
            self.active_view,
            self.active_panel,
            self.active_axis,
            layout,
            viewports,
        );
        if let Some(index) = selected_view {
            if layout.is_grid() {
                self.active_panel = index;
            }
            if let Some(viewport) = viewports.get(index) {
                self.active_axis = viewport.axis();
            }
        }
        update_active_view(&mut self.active_view, event, viewports);
        selected_view
    }
}

fn update_active_view(
    active_view: &mut Option<usize>,
    event: &PresentationEvent,
    viewports: &[super::layout::NativeViewport],
) {
    match event {
        PresentationEvent::PointerDown { x, y, .. } => {
            *active_view = view_at(viewports, *x, *y).or(*active_view);
        }
        PresentationEvent::PointerUp { .. }
        | PresentationEvent::PointerCancel { .. }
        | PresentationEvent::FocusLost
        | PresentationEvent::CloseRequested
        | PresentationEvent::Destroyed => *active_view = None,
        _ => {}
    }
}

fn event_view_index(
    event: &PresentationEvent,
    active_view: Option<usize>,
    active_panel: usize,
    active_axis: usize,
    layout: WorkspaceLayout,
    viewports: &[super::layout::NativeViewport],
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

fn view_at(viewports: &[super::layout::NativeViewport], x: f64, y: f64) -> Option<usize> {
    viewports
        .iter()
        .position(|viewport| viewport.contains(x, y))
}

impl NativeViewerSession {
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
        let mut route_cursor =
            PanelEventRouter::new(self.active_panel, self.active_view, self.active_app().axis);
        for event in events {
            let selected_view = route_cursor.route(event, self.workspace_layout, &self.viewports);
            let dispatcher_index = if grid.is_some() {
                selected_view
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
                .try_push(selected_view)
                .map_err(|_| NativeViewerError::new("viewer event route capacity exceeded"))?;
        }
        Ok(RoutedBatch {
            layout: self.workspace_layout,
            active_view: route_cursor.active_view(),
            routes,
            view_mappings,
        })
    }

    pub(super) fn apply_events(
        &mut self,
        events: &[PresentationEvent],
        routes: &[Option<usize>],
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
        let mut active_view = self.active_view;
        let mut repaint = false;
        let mut start = 0;
        while start < events.len() {
            let selected_view = routes[start];
            let Some(panel_index) = self
                .panel_for_event(&events[start], selected_view, panel_routes, grid.is_some())
                .filter(|index| {
                    !is_pointer_event(&events[start]) || panel_routes.contains_current_panel(*index)
                })
            else {
                start += 1;
                continue;
            };
            let view_index =
                self.event_view_mapping(&events[start], selected_view, panel_index, grid.is_some());
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
                && selected_view.is_some()
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
                update_active_view(&mut active_view, event, &self.viewports);
            }
            start = end;
        }
        self.active_view = active_view;
        Ok(ViewerActionDisposition::Continue { repaint })
    }

    fn panel_for_event(
        &self,
        event: &PresentationEvent,
        selected_view: Option<usize>,
        panel_routes: &PanelRouteMap,
        grid: bool,
    ) -> Option<usize> {
        if !grid {
            return Some(0);
        }
        if is_pointer_event(event) {
            selected_view.map_or(Some(self.active_panel), |view| {
                panel_routes.current_panel(view)
            })
        } else {
            Some(self.active_panel)
        }
    }

    fn event_view_mapping(
        &self,
        event: &PresentationEvent,
        selected_view: Option<usize>,
        panel_index: usize,
        grid: bool,
    ) -> Option<usize> {
        if grid && !is_pointer_event(event) {
            Some(panel_index)
        } else {
            selected_view
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

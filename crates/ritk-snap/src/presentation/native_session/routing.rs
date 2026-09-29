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

impl NativeViewerSession {
    fn view_at(&self, x: f64, y: f64) -> Option<usize> {
        self.viewports
            .iter()
            .position(|viewport| viewport.contains(x, y))
    }

    fn update_active_view(&self, active_view: &mut Option<usize>, event: &PresentationEvent) {
        match event {
            PresentationEvent::PointerDown { x, y, .. } => {
                *active_view = self.view_at(*x, *y).or(*active_view);
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
        &self,
        event: &PresentationEvent,
        active_view: Option<usize>,
        active_panel: usize,
        active_axis: usize,
    ) -> Option<usize> {
        let candidate = match event {
            PresentationEvent::PointerDown { x, y, .. }
            | PresentationEvent::PointerMove { x, y }
            | PresentationEvent::PointerUp { x, y, .. }
            | PresentationEvent::PointerCancel { x, y, .. }
            | PresentationEvent::PointerWheel { x, y, .. } => {
                active_view.or_else(|| self.view_at(*x, *y))
            }
            _ => None,
        };
        candidate.or_else(|| {
            if self.workspace_layout.is_grid() {
                self.viewports.get(active_panel).map(|_| active_panel)
            } else {
                self.viewports
                    .iter()
                    .position(|viewport| viewport.axis() == active_axis)
            }
        })
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
        let mut active_panel = self.active_panel;
        let mut active_view = self.active_view;
        let mut active_axis = self.active_app().axis;
        for event in events {
            let selected_view =
                self.event_view_index(event, active_view, active_panel, active_axis);
            let dispatcher_index = if grid.is_some() {
                selected_view
                    .unwrap_or(active_panel)
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
            if let Some(index) = selected_view {
                if grid.is_some() {
                    active_panel = index;
                }
                active_axis = self.viewports[index].axis();
            }
            self.update_active_view(&mut active_view, event);
        }
        Ok(RoutedBatch {
            layout: self.workspace_layout,
            active_view,
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
            let mut end = start + 1;
            while end < events.len() && routes[end] == selected_view {
                end += 1;
            }
            let panel_index = if grid.is_some() {
                selected_view.map_or(Some(self.active_panel), |presented_panel| {
                    panel_routes.current_panel(presented_panel)
                })
            } else {
                Some(0)
            };
            let Some(panel_index) =
                panel_index.filter(|index| panel_routes.contains_current_panel(*index))
            else {
                start = end;
                continue;
            };
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
                selected_view
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
                self.update_active_view(&mut active_view, event);
            }
            start = end;
        }
        self.active_view = active_view;
        Ok(ViewerActionDisposition::Continue { repaint })
    }
}

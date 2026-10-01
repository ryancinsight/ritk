//! Panel ownership while the native event batch is inspected for chrome actions.

use super::super::layout::{NativeViewport, PanelGrid, WorkspaceLayout, MAX_GRID_PANELS};
use super::super::panels::{MaximizedPanel, PanelRouteMap};
use super::super::routing::{is_pointer_event, PanelEventRouter};
use super::super::window_controls::{PanelCloseKind, WindowAction};
use super::super::NativeViewerError;
use crate::presentation::PresentationEvent;

pub(super) struct KeyboardPanelState {
    layout: WorkspaceLayout,
    batch_layout: WorkspaceLayout,
    active_panel: usize,
    maximized_panel: Option<MaximizedPanel>,
    panel_routes: PanelRouteMap,
    panel_series: [Option<usize>; MAX_GRID_PANELS],
    displayed_series: [Option<usize>; MAX_GRID_PANELS],
    browser_active_index: Option<usize>,
    input_router: PanelEventRouter,
}

impl KeyboardPanelState {
    pub(super) fn new(
        layout: WorkspaceLayout,
        active_panel: usize,
        active_view: Option<usize>,
        active_axis: usize,
        maximized_panel: Option<MaximizedPanel>,
        displayed_series: [Option<usize>; MAX_GRID_PANELS],
        browser_active_index: Option<usize>,
    ) -> Result<Self, NativeViewerError> {
        let panel_series = displayed_series;
        if let Some(maximized) = maximized_panel {
            if maximized.panel_index >= panel_series.len() {
                return Err(NativeViewerError::new(
                    "maximized series panel is outside the supported panel range",
                ));
            }
        }
        Ok(Self {
            layout,
            batch_layout: layout,
            active_panel,
            maximized_panel,
            panel_routes: PanelRouteMap::new(layout),
            panel_series,
            displayed_series,
            browser_active_index,
            input_router: PanelEventRouter::new(active_panel, active_view, active_axis),
        })
    }

    pub(super) const fn layout(&self) -> WorkspaceLayout {
        self.layout
    }

    pub(super) const fn active_panel(&self) -> usize {
        self.active_panel
    }

    pub(super) const fn maximized(&self) -> bool {
        self.maximized_panel.is_some()
    }

    pub(super) fn navigation_series(&self) -> [Option<usize>; MAX_GRID_PANELS] {
        let mut series = self.displayed_series;
        if let Some(active_index) = self.browser_active_index {
            for panel_series in &mut series {
                if panel_series.is_none() {
                    *panel_series = Some(active_index);
                }
            }
        }
        series
    }

    pub(super) fn retain_event(&mut self, event: &PresentationEvent, viewports: &[NativeViewport]) {
        let selected_view = self.input_router.route(event, self.batch_layout, viewports);
        if is_pointer_event(event) && self.batch_layout.is_grid() {
            let routed_panel = selected_view.map_or(Some(self.active_panel), |presented_panel| {
                self.panel_routes.current_panel(presented_panel)
            });
            if let Some(panel_index) = routed_panel
                .filter(|index| self.panel_routes.contains_current_panel(*index))
                .filter(|index| {
                    self.layout
                        .grid()
                        .is_some_and(|grid| *index < grid.panel_count())
                })
            {
                self.active_panel = panel_index;
            }
        }
    }

    pub(super) fn apply_action(
        &mut self,
        action: WindowAction,
        is_keyboard: bool,
    ) -> Result<(), NativeViewerError> {
        let action = match (is_keyboard, action) {
            (true, WindowAction::BrowseSeries { series_index, .. }) => {
                Some(WindowAction::BrowseSeries {
                    series_index,
                    panel_index: self.active_panel,
                })
            }
            _ => self.panel_routes.route_action(action),
        };
        let Some(action) = action else {
            return Ok(());
        };
        let next_browser_active_index = match action {
            WindowAction::BrowseSeries {
                series_index,
                panel_index,
            } if self.maximized_panel.is_none() || panel_index == 0 => Some(series_index),
            WindowAction::SelectSeries(series_index)
            | WindowAction::OpenSeriesInNextPanel(series_index)
            | WindowAction::AssignSeries { series_index, .. } => Some(series_index),
            _ => None,
        };
        let maximized_before = self.maximized_panel.map(|panel| panel.panel_index);
        let active_before = self.active_panel;
        let layout_before = self.layout;
        let panel_count_before = self
            .maximized_panel
            .map_or(layout_before, |panel| panel.layout)
            .grid()
            .map_or(1, PanelGrid::panel_count);
        let reset_browser_to_primary = panel_count_before > 1
            && matches!(
                action,
                WindowAction::CloseActivePanel
                    | WindowAction::ClosePanel {
                        kind: PanelCloseKind::Close,
                        ..
                    }
            );
        let prior_state = (
            self.layout,
            self.active_panel,
            self.maximized_panel
                .map(|panel| (panel.layout, panel.panel_index)),
            self.panel_series,
            self.displayed_series,
        );
        self.update_state(action)?;
        if let Some(index) = next_browser_active_index.or_else(|| {
            reset_browser_to_primary
                .then_some(self.panel_series[0])
                .flatten()
        }) {
            self.browser_active_index = Some(index);
        }
        let changed = prior_state
            != (
                self.layout,
                self.active_panel,
                self.maximized_panel
                    .map(|panel| (panel.layout, panel.panel_index)),
                self.panel_series,
                self.displayed_series,
            );
        self.panel_routes.apply_action(
            action,
            maximized_before,
            self.maximized_panel.map(|panel| panel.panel_index),
            active_before,
            panel_count_before,
            changed,
        );
        Ok(())
    }

    fn update_state(&mut self, action: WindowAction) -> Result<(), NativeViewerError> {
        match action {
            WindowAction::ActivateNextPanel => self.activate_panel(true),
            WindowAction::ActivatePreviousPanel => self.activate_panel(false),
            WindowAction::BrowseSeries {
                series_index,
                panel_index,
            } => {
                self.set_series(panel_index, series_index)?;
                if self.maximized_panel.is_none() || panel_index == 0 {
                    self.active_panel = panel_index;
                }
                Ok(())
            }
            WindowAction::SelectSeries(series_index) => self.select_series(series_index),
            WindowAction::AssignSeries {
                series_index,
                panel_index,
            } => {
                self.set_series(panel_index, series_index)?;
                self.active_panel = panel_index;
                Ok(())
            }
            WindowAction::OpenSeriesInNextPanel(series_index) => {
                self.restore_maximized_panel();
                let panel_count = self.layout.grid().map_or(1, PanelGrid::panel_count);
                let panel_index = self
                    .panel_series
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find_map(|(index, series)| series.is_none().then_some(index))
                    .unwrap_or(panel_count);
                if panel_index >= self.panel_series.len() {
                    return Ok(());
                }
                if panel_index >= panel_count {
                    let grid = PanelGrid::containing_panel(panel_index).ok_or_else(|| {
                        NativeViewerError::new("assigned series exceeds the supported panel range")
                    })?;
                    self.layout = WorkspaceLayout::Panels(grid);
                }
                self.set_series(panel_index, series_index)?;
                self.active_panel = panel_index;
                Ok(())
            }
            WindowAction::ClosePanel { index, kind } => self.close_panel(index, kind),
            WindowAction::CloseActivePanel => {
                let panel_index = if self.maximized_panel.is_some() {
                    0
                } else {
                    self.active_panel
                };
                self.close_panel(panel_index, PanelCloseKind::Close)
            }
            WindowAction::ToggleActivePanel => self.toggle_active_panel(),
            WindowAction::MaximizePanel(panel_index) => self.toggle_maximized_panel(panel_index),
            WindowAction::SetLayout(layout) => {
                self.restore_maximized_panel();
                self.layout = layout;
                self.active_panel = 0;
                self.displayed_series = self.panel_series;
                Ok(())
            }
            WindowAction::CloseAllPanels => {
                self.maximized_panel = None;
                self.layout = WorkspaceLayout::Orthogonal;
                self.active_panel = 0;
                self.panel_series.fill(None);
                self.displayed_series.fill(None);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn activate_panel(&mut self, next: bool) -> Result<(), NativeViewerError> {
        if let Some(maximized) = self.maximized_panel {
            let count = maximized.layout.grid().map_or(1, PanelGrid::panel_count);
            if count > 1 {
                let panel_index = if next {
                    (maximized.panel_index + 1) % count
                } else {
                    maximized.panel_index.checked_sub(1).unwrap_or(count - 1)
                };
                self.restore_maximized_panel();
                self.panel_series.swap(0, panel_index);
                self.maximized_panel = Some(MaximizedPanel {
                    layout: self.layout,
                    panel_index,
                });
                self.layout = one_panel_layout();
                self.displayed_series = self.panel_series;
            }
            self.active_panel = 0;
        } else {
            let count = self.layout.grid().map_or(1, PanelGrid::panel_count);
            if count > 1 {
                self.active_panel = if next {
                    (self.active_panel + 1) % count
                } else {
                    self.active_panel.checked_sub(1).unwrap_or(count - 1)
                };
            }
        }
        Ok(())
    }

    fn toggle_active_panel(&mut self) -> Result<(), NativeViewerError> {
        if self.maximized_panel.is_some() {
            self.restore_maximized_panel();
            return Ok(());
        }
        let Some(grid) = self.layout.grid() else {
            return Ok(());
        };
        if grid.panel_count() <= 1 || self.active_panel >= self.panel_series.len() {
            return Ok(());
        }
        let panel_index = self.active_panel;
        self.panel_series.swap(0, panel_index);
        self.maximized_panel = Some(MaximizedPanel {
            layout: self.layout,
            panel_index,
        });
        self.layout = one_panel_layout();
        self.active_panel = 0;
        self.displayed_series = self.panel_series;
        Ok(())
    }

    fn toggle_maximized_panel(&mut self, panel_index: usize) -> Result<(), NativeViewerError> {
        if let Some(maximized) = self.maximized_panel {
            let target_panel = if panel_index == 0 {
                maximized.panel_index
            } else if panel_index == maximized.panel_index {
                0
            } else {
                panel_index
            };
            self.restore_maximized_panel();
            if target_panel == maximized.panel_index {
                return Ok(());
            }
            return self.maximize_panel(target_panel);
        }
        self.maximize_panel(panel_index)
    }

    fn maximize_panel(&mut self, panel_index: usize) -> Result<(), NativeViewerError> {
        let Some(grid) = self.layout.grid() else {
            return Ok(());
        };
        if panel_index >= grid.panel_count() || grid.panel_count() <= 1 {
            return Ok(());
        }
        self.panel_series.swap(0, panel_index);
        self.maximized_panel = Some(MaximizedPanel {
            layout: self.layout,
            panel_index,
        });
        self.layout = one_panel_layout();
        self.active_panel = 0;
        self.displayed_series = self.panel_series;
        Ok(())
    }

    fn set_series(
        &mut self,
        panel_index: usize,
        series_index: usize,
    ) -> Result<(), NativeViewerError> {
        let series = self
            .panel_series
            .get_mut(panel_index)
            .ok_or_else(|| NativeViewerError::new("assigned panel is outside the series grid"))?;
        *series = Some(series_index);
        self.displayed_series[panel_index] = Some(series_index);
        Ok(())
    }

    fn close_panel(
        &mut self,
        visible_panel: usize,
        kind: PanelCloseKind,
    ) -> Result<(), NativeViewerError> {
        let maximized = self.maximized_panel.take();
        let layout = maximized.map_or(self.layout, |panel| panel.layout);
        let panel_count = layout.grid().map_or(1, PanelGrid::panel_count);
        let panel_index = maximized.map_or(visible_panel, |panel| {
            if panel.panel_index > 0 {
                self.panel_series.swap(0, panel.panel_index);
            }
            if visible_panel == 0 {
                panel.panel_index
            } else if visible_panel == panel.panel_index {
                0
            } else {
                visible_panel
            }
        });
        if panel_index >= panel_count {
            return Ok(());
        }
        self.layout = layout;
        self.displayed_series = self.panel_series;
        if kind == PanelCloseKind::Clear {
            self.panel_series[panel_index] = None;
            self.displayed_series[panel_index] = None;
            self.active_panel = panel_index;
            return Ok(());
        }
        if panel_count == 1 {
            self.panel_series.fill(None);
            self.displayed_series.fill(None);
            self.active_panel = 0;
            return Ok(());
        }
        let closed_index = if panel_index == 0 {
            self.panel_series.swap(0, 1);
            1
        } else {
            panel_index
        };
        self.panel_series
            .copy_within(closed_index + 1..panel_count, closed_index);
        self.panel_series[panel_count - 1] = None;
        let remaining_count = panel_count - 1;
        let grid = PanelGrid::containing_panel(remaining_count - 1).ok_or_else(|| {
            NativeViewerError::new("remaining series panels exceed the supported grid")
        })?;
        self.layout = WorkspaceLayout::Panels(grid);
        self.displayed_series = self.panel_series;
        self.active_panel = panel_index.min(remaining_count - 1);
        Ok(())
    }

    fn restore_maximized_panel(&mut self) {
        let Some(maximized) = self.maximized_panel.take() else {
            return;
        };
        if maximized.panel_index > 0 {
            self.panel_series.swap(0, maximized.panel_index);
        }
        self.layout = maximized.layout;
        self.displayed_series = self.panel_series;
        self.active_panel = maximized.panel_index;
    }

    fn select_series(&mut self, series_index: usize) -> Result<(), NativeViewerError> {
        if self.maximized_panel.is_some() && self.panel_series[0] != Some(series_index) {
            self.restore_maximized_panel();
        }
        if let Some(panel_index) = self
            .panel_series
            .iter()
            .position(|series| *series == Some(series_index))
        {
            let panel_count = self.layout.grid().map_or(1, PanelGrid::panel_count);
            if panel_index >= panel_count {
                let grid = PanelGrid::containing_panel(panel_index).ok_or_else(|| {
                    NativeViewerError::new("selected series exceeds the supported panel range")
                })?;
                self.layout = WorkspaceLayout::Panels(grid);
            }
            self.active_panel = panel_index;
            return Ok(());
        }
        self.set_series(self.active_panel, series_index)
    }
}

fn one_panel_layout() -> WorkspaceLayout {
    WorkspaceLayout::Panels(PanelGrid::new(1, 1).expect("invariant: one-panel layout is valid"))
}

#[cfg(test)]
#[path = "panel_state/tests.rs"]
mod tests;

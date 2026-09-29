//! Multi-series panel lifecycle transitions.

use super::compare::ComparePanel;
use super::frame;
use super::layout::{PanelGrid, WorkspaceLayout, MAX_GRID_PANELS};
use super::session::NativeViewerSession;
use super::window_controls::{PanelCloseKind, WindowAction};
use crate::app::SnapApp;
use anyhow::{anyhow, Result};

/// Translates panel positions captured at batch start to their current owners.
pub(super) struct PanelRouteMap {
    original_to_current: [Option<usize>; MAX_GRID_PANELS],
}

impl PanelRouteMap {
    pub(super) fn new(layout: WorkspaceLayout) -> Self {
        let mut original_to_current = [None; MAX_GRID_PANELS];
        for (index, route) in original_to_current
            .iter_mut()
            .enumerate()
            .take(layout.grid().map_or(1, PanelGrid::panel_count))
        {
            *route = Some(index);
        }
        Self {
            original_to_current,
        }
    }

    pub(super) fn current_panel(&self, presented_panel: usize) -> Option<usize> {
        self.original_to_current
            .get(presented_panel)
            .copied()
            .flatten()
    }

    /// Returns whether a routed owner still has retained panel state.
    pub(super) fn contains_current_panel(&self, panel_index: usize) -> bool {
        self.original_to_current.contains(&Some(panel_index))
    }

    pub(super) fn route_action(&self, action: WindowAction) -> Option<WindowAction> {
        match action {
            WindowAction::MaximizePanel(index) => {
                self.current_panel(index).map(WindowAction::MaximizePanel)
            }
            WindowAction::ClosePanel { index, kind } => self
                .current_panel(index)
                .map(|index| WindowAction::ClosePanel { index, kind }),
            WindowAction::AssignSeries {
                series_index,
                panel_index,
            } => self
                .current_panel(panel_index)
                .map(|panel_index| WindowAction::AssignSeries {
                    series_index,
                    panel_index,
                }),
            WindowAction::BrowseSeries {
                series_index,
                panel_index,
            } => self
                .current_panel(panel_index)
                .map(|panel_index| WindowAction::BrowseSeries {
                    series_index,
                    panel_index,
                }),
            _ => Some(action),
        }
    }

    pub(super) fn apply_action(
        &mut self,
        action: WindowAction,
        maximized_before: Option<usize>,
        maximized_after: Option<usize>,
        active_before: usize,
        panel_count_before: usize,
        changed: bool,
    ) {
        if !changed {
            return;
        }

        match (maximized_before, maximized_after) {
            (Some(previous), Some(next)) if previous != next => {
                self.swap(0, previous);
                self.swap(0, next);
            }
            (Some(panel), None) | (None, Some(panel)) => self.swap(0, panel),
            _ => {}
        }

        let close_index = match action {
            WindowAction::ClosePanel {
                index,
                kind: PanelCloseKind::Close,
            } => Some(maximized_before.map_or(index, |maximized| {
                if index == 0 {
                    maximized
                } else if index == maximized {
                    0
                } else {
                    index
                }
            })),
            WindowAction::CloseActivePanel => Some(maximized_before.unwrap_or(active_before)),
            WindowAction::CloseAllPanels => {
                self.original_to_current.fill(None);
                return;
            }
            _ => None,
        };
        let Some(index) = close_index.filter(|_| panel_count_before > 1) else {
            return;
        };
        let closed_primary = index == 0;
        for panel_route in &mut self.original_to_current {
            *panel_route = panel_route.and_then(|current| {
                if closed_primary {
                    match current {
                        0 => None,
                        1 => Some(0),
                        value => value.checked_sub(1),
                    }
                } else if current == index {
                    None
                } else if current > index {
                    current.checked_sub(1)
                } else {
                    Some(current)
                }
            });
        }
    }

    fn swap(&mut self, first: usize, second: usize) {
        for route in &mut self.original_to_current {
            *route = route.map(|current| {
                if current == first {
                    second
                } else if current == second {
                    first
                } else {
                    current
                }
            });
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MaximizedPanel {
    pub(super) layout: WorkspaceLayout,
    pub(super) panel_index: usize,
}

impl NativeViewerSession {
    pub(super) fn toggle_panel_maximize(&mut self, panel_index: usize) -> Result<bool> {
        if let Some(maximized) = self.maximized_panel {
            let target_panel = restored_panel_index(maximized, panel_index);
            self.restore_maximized_panel()?;
            return if target_panel == maximized.panel_index {
                Ok(true)
            } else {
                self.toggle_panel_maximize(target_panel)
            };
        }
        let Some(grid) = self.workspace_layout.grid() else {
            return Ok(false);
        };
        if panel_index >= grid.panel_count() || grid.panel_count() == 1 {
            return Ok(false);
        }
        if panel_index > 0 {
            self.swap_panel_with_primary(panel_index)?;
        }
        self.maximized_panel = Some(MaximizedPanel {
            layout: self.workspace_layout,
            panel_index,
        });
        self.workspace_layout = WorkspaceLayout::Panels(
            PanelGrid::new(1, 1).ok_or_else(|| anyhow!("one-panel layout is invalid"))?,
        );
        self.active_panel = 0;
        self.active_view = None;
        Ok(true)
    }

    pub(super) fn restore_maximized_panel(&mut self) -> Result<bool> {
        let Some(maximized) = self.maximized_panel.take() else {
            return Ok(false);
        };
        if maximized.panel_index > 0 {
            self.swap_panel_with_primary(maximized.panel_index)?;
        }
        self.workspace_layout = maximized.layout;
        let panel_count = maximized.layout.grid().map_or(1, PanelGrid::panel_count);
        self.active_panel = maximized.panel_index.min(panel_count.saturating_sub(1));
        self.active_view = None;
        Ok(true)
    }

    pub(super) fn close_active_panel(&mut self) -> Result<bool> {
        let panel_index = if self.maximized_panel.is_some() {
            0
        } else {
            self.active_panel
        };
        self.close_panel(panel_index)
    }

    pub(super) fn close_panel(&mut self, visible_index: usize) -> Result<bool> {
        let panel_index = self.maximized_panel.map_or(visible_index, |panel| {
            restored_panel_index(panel, visible_index)
        });
        self.restore_maximized_panel()?;
        let Some(grid) = self.workspace_layout.grid() else {
            self.clear_primary()?;
            return Ok(true);
        };
        let current_count = grid.panel_count();
        if panel_index >= current_count {
            return Ok(false);
        }
        if current_count == 1 {
            self.clear_primary()?;
            self.active_panel = 0;
            return Ok(true);
        }
        if panel_index == 0 {
            self.swap_panel_with_primary(1)?;
            self.compare_panels.remove(0);
        } else {
            if panel_index > self.compare_panels.len() {
                return Err(anyhow!("closed series panel has no retained state"));
            }
            self.compare_panels.remove(panel_index - 1);
        }
        self.compare_panels
            .try_push(ComparePanel::empty()?)
            .map_err(|_| anyhow!("series panel capacity is full after close"))?;
        let remaining_count = current_count.saturating_sub(1);
        let remaining_grid = PanelGrid::containing_panel(remaining_count.saturating_sub(1))
            .ok_or_else(|| anyhow!("remaining series panels exceed the supported grid"))?;
        self.workspace_layout = WorkspaceLayout::Panels(remaining_grid);
        self.active_panel = panel_index.min(remaining_count.saturating_sub(1));
        self.active_view = None;
        if let (Some(browser), Some(index)) =
            (self.series_browser.as_mut(), self.primary_series_index)
        {
            browser.set_active(index);
        }
        Ok(true)
    }

    pub(super) fn clear_panel(&mut self, visible_index: usize) -> Result<bool> {
        let panel_index = self.maximized_panel.map_or(visible_index, |panel| {
            restored_panel_index(panel, visible_index)
        });
        self.restore_maximized_panel()?;
        if self.workspace_layout.is_grid() && panel_index > 0 {
            let panel = self
                .compare_panels
                .get_mut(panel_index - 1)
                .ok_or_else(|| anyhow!("cleared series panel has no retained state"))?;
            panel.clear()?;
        } else {
            self.clear_primary()?;
        }
        self.active_panel = panel_index.min(
            self.workspace_layout
                .grid()
                .map_or(1, PanelGrid::panel_count)
                .saturating_sub(1),
        );
        self.active_view = None;
        Ok(true)
    }

    pub(super) fn close_all_panels(&mut self) -> Result<bool> {
        self.maximized_panel = None;
        self.workspace_layout = WorkspaceLayout::Orthogonal;
        self.active_panel = 0;
        self.active_view = None;
        self.compare_panels.clear();
        self.clear_primary()?;
        Ok(true)
    }

    pub(super) fn activate_next_panel(&mut self) -> Result<bool> {
        self.activate_panel_by(1)
    }

    pub(super) fn activate_previous_panel(&mut self) -> Result<bool> {
        self.activate_panel_by(-1)
    }

    fn activate_panel_by(&mut self, step: i32) -> Result<bool> {
        if let Some(maximized) = self.maximized_panel {
            let Some(grid) = maximized.layout.grid() else {
                return Ok(false);
            };
            let count = grid.panel_count();
            let next = if step > 0 {
                (maximized.panel_index + 1) % count
            } else {
                maximized.panel_index.checked_sub(1).unwrap_or(count - 1)
            };
            if next == maximized.panel_index {
                return Ok(false);
            }
            self.restore_maximized_panel()?;
            self.active_panel = next;
            return self.toggle_panel_maximize(next);
        }
        let Some(grid) = self.workspace_layout.grid() else {
            return Ok(false);
        };
        let count = grid.panel_count();
        if count <= 1 {
            return Ok(false);
        }
        self.active_panel = if step > 0 {
            (self.active_panel + 1) % count
        } else {
            self.active_panel.checked_sub(1).unwrap_or(count - 1)
        };
        self.active_view = None;
        Ok(true)
    }

    fn swap_panel_with_primary(&mut self, panel_index: usize) -> Result<()> {
        let panel = self
            .compare_panels
            .get_mut(panel_index.saturating_sub(1))
            .filter(|_| panel_index > 0)
            .ok_or_else(|| anyhow!("series panel state is missing"))?;
        panel.swap_primary(
            &mut self.app,
            &mut self.views[0],
            &mut self.render_scratch[0],
            &mut self.primary_series_index,
        );
        Ok(())
    }

    fn clear_primary(&mut self) -> Result<()> {
        self.app = SnapApp::default();
        self.views = frame::empty_orthogonal_views()?;
        self.render_scratch = std::array::from_fn(|_| Default::default());
        self.projection = None;
        self.projection_scratch = Default::default();
        self.primary_series_index = None;
        Ok(())
    }
}

fn restored_panel_index(maximized: MaximizedPanel, current_panel: usize) -> usize {
    if current_panel == 0 {
        maximized.panel_index
    } else if current_panel == maximized.panel_index {
        0
    } else {
        current_panel
    }
}

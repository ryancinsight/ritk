//! Apply native viewer menu, toolbar and panel actions.

use super::window_controls::{PanelCloseKind, WindowAction};
use super::{NativeViewerError, NativeViewerSession};
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;

impl NativeViewerSession {
    pub(super) fn apply_window_action(
        &mut self,
        action: WindowAction,
    ) -> std::result::Result<bool, NativeViewerError> {
        match action {
            WindowAction::OpenMenu(_) => Ok(true),
            WindowAction::OpenStudy => match self.open_study_from_dialog() {
                Ok(reopened) => Ok(reopened),
                Err(error) => {
                    self.active_app_mut().status_message =
                        format!("DICOM reopen failed; current study remains displayed: {error:#}");
                    Ok(true)
                }
            },
            WindowAction::OpenSeriesPicker => {
                self.restore_maximized_panel()
                    .map_err(NativeViewerError::from)?;
                self.window_chrome
                    .open_series_picker(self.series_browser.as_ref())
                    .map_err(NativeViewerError::from)
            }
            WindowAction::LoadSelectedSeries => {
                let Some(selected) = self.window_chrome.take_selected_series() else {
                    return Ok(false);
                };
                match self.open_selected_series(selected.as_slice()) {
                    Ok(changed) => Ok(changed),
                    Err(error) => {
                        self.active_app_mut().status_message = format!(
                            "Selected series could not be opened; current panels remain: {error:#}"
                        );
                        Ok(true)
                    }
                }
            }
            WindowAction::SelectSeries(index) => match self.select_series(index) {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message = format!(
                        "Series could not be opened; the active image remains displayed: {error:#}"
                    );
                    Ok(true)
                }
            },
            WindowAction::BrowseSeries {
                series_index,
                panel_index,
            } => match self.browse_series(series_index, panel_index) {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message = format!(
                        "Series could not be opened in the active panel; its image remains displayed: {error:#}"
                    );
                    Ok(true)
                }
            },
            WindowAction::OpenSeriesInNextPanel(index) => {
                match self.assign_series_to_next_panel(index) {
                    Ok(changed) => Ok(changed),
                    Err(error) => {
                        self.active_app_mut().status_message =
                            format!("Series could not be opened in another panel: {error:#}");
                        Ok(true)
                    }
                }
            }
            WindowAction::MaximizePanel(index) => self
                .toggle_panel_maximize(index)
                .map_err(NativeViewerError::from),
            WindowAction::ClosePanel { index, kind } => match kind {
                PanelCloseKind::Close => self.close_panel(index).map_err(NativeViewerError::from),
                PanelCloseKind::Clear => self.clear_panel(index).map_err(NativeViewerError::from),
            },
            WindowAction::ToggleActivePanel => self
                .toggle_panel_maximize(self.active_panel)
                .map_err(NativeViewerError::from),
            WindowAction::CloseActivePanel => {
                self.close_active_panel().map_err(NativeViewerError::from)
            }
            WindowAction::CloseAllPanels => {
                self.close_all_panels().map_err(NativeViewerError::from)
            }
            WindowAction::ActivateNextPanel => {
                self.activate_next_panel().map_err(NativeViewerError::from)
            }
            WindowAction::ActivatePreviousPanel => self
                .activate_previous_panel()
                .map_err(NativeViewerError::from),
            WindowAction::AssignSeries {
                series_index,
                panel_index,
            } => match self.assign_series_to_panel(series_index, panel_index) {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message = format!(
                        "Series could not be assigned; the panel remains displayed: {error:#}"
                    );
                    Ok(true)
                }
            },
            WindowAction::ToggleSeriesPreview => {
                self.window_chrome.toggle_series_preview();
                Ok(true)
            }
            WindowAction::SetLayout(layout) => self
                .set_workspace_layout(layout)
                .map_err(NativeViewerError::from),
            WindowAction::Exit => Ok(true),
            WindowAction::SelectTool(tool) => {
                if self.workspace_layout.is_grid() && tool == ToolKind::Crosshair {
                    return Ok(false);
                }
                let app = self.active_app_mut();
                app.active_tool = tool;
                app.tool_state = ToolState::Idle;
                Ok(true)
            }
            WindowAction::ToggleCrosshair => {
                if self.workspace_layout.is_grid() {
                    return Ok(false);
                }
                let app = self.active_app_mut();
                app.show_crosshair = !app.show_crosshair;
                Ok(true)
            }
            WindowAction::ToggleCine => {
                self.active_app_mut().toggle_cine();
                Ok(true)
            }
            WindowAction::ResetView => {
                let app = self.active_app_mut();
                app.zoom = 1.0;
                app.pan_offset = ViewportOffset::new(0.0, 0.0);
                app.tool_state = ToolState::Idle;
                Ok(true)
            }
        }
    }
}

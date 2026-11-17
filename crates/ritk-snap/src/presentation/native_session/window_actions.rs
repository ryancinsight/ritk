//! Apply native viewer menu, toolbar and panel actions.

use super::window_controls::{PanelCloseKind, WindowAction};
use super::{NativeViewerError, NativeViewerSession};
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;

impl NativeViewerSession {
    /// Zoom multiplier that undoes the active panel's fit scale (1:1 pixels).
    fn one_to_one_zoom(&self) -> Option<f32> {
        let view = self.views.get(self.active_panel)?;
        let viewport = self.viewports.get(self.active_panel)?;
        crate::ui::zoom::one_to_one_zoom(
            [viewport.panel_width, viewport.panel_height],
            [view.frame.width(), view.frame.height()],
            view.frame.display_spacing().values(),
        )
    }

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
            WindowAction::Anonymize(scope, level) => match self.anonymize_from_dialog(scope, level)
            {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message =
                        super::deidentify::failure_status(&error);
                    Ok(true)
                }
            },
            WindowAction::RepairStudy => match self.repair_study_from_dialog() {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message =
                        format!("Repaired copy failed: {error:#}");
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
            WindowAction::SelectSlice {
                panel_index,
                axis,
                index,
            } => {
                let active_changed = self.active_panel != panel_index;
                let app = self.app_for_panel_mut(panel_index)?;
                let axis_changed = app.axis != axis;
                app.axis = axis;
                let (previous, _) = app.axis_slice_info(axis);
                if let Some(index) = index {
                    app.set_slice_for_axis(axis, index);
                }
                let slice_changed = previous != app.axis_slice_info(axis).0;
                self.active_panel = panel_index;
                Ok(active_changed || axis_changed || slice_changed)
            }
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
            WindowAction::CancelPendingLoads => Ok(self.cancel_series_loads()),
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
                self.active_app_mut().select_tool(tool);
                Ok(true)
            }
            WindowAction::AssignPointerTool { button, tool } => {
                if let Err(error) = self.active_app_mut().assign_pointer_tool(button, tool) {
                    self.active_app_mut().status_message = format!(
                        "Pointer button reassignment failed; the default layout remains: {error:#}"
                    );
                }
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
            WindowAction::ToggleOverlay => {
                let app = self.active_app_mut();
                app.show_overlay = !app.show_overlay;
                Ok(true)
            }
            WindowAction::ToggleAnonymizedOverlay => {
                self.anonymize_overlay = !self.anonymize_overlay;
                Ok(true)
            }
            WindowAction::ToggleCine => {
                self.active_app_mut().toggle_cine();
                Ok(true)
            }
            WindowAction::ToggleInvert => {
                let app = self.active_app_mut();
                app.invert_display = !app.invert_display;
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::SetColormap(map) => {
                let app = self.active_app_mut();
                let changed = app.colormap != map;
                app.colormap = map;
                if changed {
                    app.bump_visual_revision();
                }
                Ok(changed)
            }
            WindowAction::RotateCw => {
                let app = self.active_app_mut();
                app.view_transform = app.view_transform.rotate_cw();
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::RotateCcw => {
                let app = self.active_app_mut();
                app.view_transform = app.view_transform.rotate_ccw();
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::FlipHorizontal => {
                let app = self.active_app_mut();
                app.view_transform = app.view_transform.toggle_flip_h();
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::FlipVertical => {
                let app = self.active_app_mut();
                app.view_transform = app.view_transform.toggle_flip_v();
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::ResetOrientation => {
                let app = self.active_app_mut();
                app.view_transform = app.view_transform.reset();
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::ZoomFit => {
                let app = self.active_app_mut();
                app.zoom = crate::ui::zoom::FIT_ZOOM;
                app.pan_offset = crate::tools::interaction::ViewportOffset::new(
                    crate::ui::zoom::FIT_PAN_OFFSET[0],
                    crate::ui::zoom::FIT_PAN_OFFSET[1],
                );
                app.bump_visual_revision();
                Ok(true)
            }
            WindowAction::ApplyCustomPreset { index } => {
                match self.active_app_mut().apply_custom_preset(index) {
                    Ok(changed) => Ok(changed),
                    Err(error) => {
                        self.active_app_mut().status_message = format!(
                            "Custom window preset could not be applied; the active image remains displayed: {error:#}"
                        );
                        Ok(true)
                    }
                }
            }
            WindowAction::SaveCustomPreset { index } => {
                match self.active_app_mut().save_custom_preset(index) {
                    Ok(_) => Ok(true),
                    Err(error) => {
                        self.active_app_mut().status_message = format!(
                            "Custom window preset could not be saved; no slot changed: {error:#}"
                        );
                        Ok(true)
                    }
                }
            }
            WindowAction::ZoomOneToOne => match self.one_to_one_zoom() {
                Some(zoom) => {
                    let app = self.active_app_mut();
                    app.zoom = zoom;
                    app.bump_visual_revision();
                    Ok(true)
                }
                None => {
                    self.active_app_mut().status_message =
                        "1:1 zoom needs a finite active panel extent.".to_owned();
                    Ok(true)
                }
            },
            WindowAction::ApplyWindowPreset { index } => {
                match self.active_app_mut().apply_browser_window_preset(index) {
                    Ok(changed) => Ok(changed),
                    Err(error) => {
                        self.active_app_mut().status_message = format!(
                            "Window preset could not be applied; the active image remains displayed: {error:#}"
                        );
                        Ok(true)
                    }
                }
            }
            WindowAction::ResetView => {
                let app = self.active_app_mut();
                app.zoom = 1.0;
                app.pan_offset = ViewportOffset::new(0.0, 0.0);
                app.tool_state = ToolState::Idle;
                app.pointer.abandon();
                Ok(true)
            }
            #[cfg(windows)]
            WindowAction::OpenCalibration => {
                self.window_chrome.open_calibration();
                Ok(true)
            }
            #[cfg(windows)]
            WindowAction::ApplyCalibration => {
                let Some(known_mm) = self.window_chrome.take_calibration_value() else {
                    return Ok(false);
                };
                self.active_app_mut()
                    .calibrate_last_active_length(known_mm)
                    .map_err(|error| NativeViewerError::new(error.to_string()))?;
                self.active_app_mut().status_message =
                    format!("Length calibration set to {known_mm:.3} mm");
                Ok(true)
            }
            #[cfg(windows)]
            WindowAction::ResetCalibration => {
                let changed = self
                    .active_app_mut()
                    .reset_active_length_calibration()
                    .map_err(|error| NativeViewerError::new(error.to_string()))?;
                if changed {
                    self.active_app_mut().status_message =
                        "Length calibration reset to DICOM spacing".to_owned();
                }
                Ok(changed)
            }
        }
    }
}

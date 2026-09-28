//! Native Métis event reduction and lifecycle for the RITK session.

use super::WindowAction;
use super::{record_state, NativeViewerError, NativeViewerSession, VIRTUAL_KEY_OPEN_STUDY};
use crate::app::action_adapter::ViewerActionDisposition;
use crate::presentation::{translate_native_events, PresentationEvent};
use crate::tools::interaction::{ToolState, ViewportOffset};
use metis_platform::native::{NativeApplication, NativeFlow, WindowEvent};
use metis_platform::Framebuffer;
use std::sync::atomic::Ordering;

impl NativeViewerSession {
    pub(super) fn tick_cine_at(&mut self, timestamp: f64) -> bool {
        let primary_advanced = matches!(
            self.app.tick_cine_at(timestamp),
            crate::app::CineTick::Advanced(_)
        );
        let mut secondary_advanced = false;
        if let Some(grid) = self.workspace_layout.grid() {
            for panel in self
                .compare_panels
                .iter_mut()
                .take(grid.panel_count().saturating_sub(1))
            {
                secondary_advanced |= matches!(
                    panel.app.tick_cine_at(timestamp),
                    crate::app::CineTick::Advanced(_)
                );
            }
        }
        primary_advanced || secondary_advanced
    }

    fn apply_viewer_event_segment(
        &mut self,
        events: &[PresentationEvent],
    ) -> std::result::Result<ViewerActionDisposition, NativeViewerError> {
        self.apply_events(events)
    }

    fn apply_window_action(
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
            WindowAction::SelectSeries(index) => match self.select_series(index) {
                Ok(changed) => Ok(changed),
                Err(error) => {
                    self.active_app_mut().status_message = format!(
                        "Series could not be opened; the active image remains displayed: {error:#}"
                    );
                    Ok(true)
                }
            },
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
                let app = self.active_app_mut();
                app.active_tool = tool;
                app.tool_state = ToolState::Idle;
                Ok(true)
            }
            WindowAction::ToggleCrosshair => {
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

impl NativeApplication for NativeViewerSession {
    type Error = NativeViewerError;

    fn framebuffer(&self) -> &Framebuffer {
        self.observation
            .presented_frames
            .fetch_add(1, Ordering::Relaxed);
        &self.framebuffer
    }

    fn handle_events(
        &mut self,
        events: &[WindowEvent],
    ) -> std::result::Result<NativeFlow, NativeViewerError> {
        let translated = translate_native_events(events).map_err(|error| {
            NativeViewerError::new(format!("translate Métis native events: {error}"))
        })?;
        self.observation
            .event_batches
            .fetch_add(1, Ordering::Relaxed);
        self.observation
            .translated_events
            .fetch_add(translated.len(), Ordering::Relaxed);
        if translated.is_empty() && self.capture_after_idle {
            self.record_terminal_frame(false)
                .map_err(NativeViewerError::from)?;
            return Ok(NativeFlow::Exit);
        }

        let mut resize = None;
        let mut dpi = None;
        let mut terminal = false;
        let mut destroyed = false;
        for event in translated.iter() {
            match event {
                PresentationEvent::Resized { width, height } => resize = Some((*width, *height)),
                PresentationEvent::DpiChanged { dpi: value } => dpi = Some(*value),
                PresentationEvent::CloseRequested => terminal = true,
                PresentationEvent::Destroyed => {
                    terminal = true;
                    destroyed = true;
                }
                _ => {}
            }
        }
        if dpi == Some(0) {
            return Err(NativeViewerError::new("native display DPI must be nonzero"));
        }

        let resized = resize.is_some_and(|(width, height)| width > 0 && height > 0);
        if let Some((width, height)) = resize {
            self.surface_width = width;
            self.surface_height = height;
            self.minimized = width == 0 || height == 0;
            self.observation
                .surface_width
                .store(width, Ordering::Relaxed);
            self.observation
                .surface_height
                .store(height, Ordering::Relaxed);
            self.observation
                .minimized
                .store(self.minimized, Ordering::Relaxed);
        }
        if let Some(value) = dpi {
            self.dpi = value;
            self.observation.dpi.store(value, Ordering::Relaxed);
        }

        // Establish the new viewport before reducing pointer events in the
        // same provider batch. Métis can coalesce a resize with input, and
        // those coordinates must use the new client rectangle.
        let geometry_refreshed = if resized {
            self.refresh_frame().map_err(NativeViewerError::from)?;
            true
        } else {
            false
        };
        let mut study_reopened = false;
        let mut open_shortcut_seen = false;
        for event in translated.iter() {
            let is_open_shortcut = matches!(
                event,
                PresentationEvent::KeyDown {
                    virtual_key: VIRTUAL_KEY_OPEN_STUDY,
                    repeated: false,
                    modifiers,
                } if modifiers.ctrl()
            );
            if !terminal && is_open_shortcut && !open_shortcut_seen {
                open_shortcut_seen = true;
                match self.open_study_from_dialog() {
                    Ok(reopened) => study_reopened |= reopened,
                    Err(error) => {
                        self.active_app_mut().status_message = format!(
                            "DICOM reopen failed; current study remains displayed: {error:#}"
                        );
                        study_reopened = true;
                    }
                }
            }
        }
        let mut chrome_repaint = false;
        let mut chrome_actions = Vec::new();
        chrome_actions
            .try_reserve(translated.len())
            .map_err(|_| NativeViewerError::new("reserve native control actions"))?;
        let mut viewer_events = Vec::from(translated);
        let mut retained_event_count = 0;
        let mut chrome_error = None;
        let width = self.surface_width;
        let height = self.surface_height;
        let workspace_layout = self.workspace_layout;
        let active_panel = self.active_panel;
        let (window_chrome, primary_app, compare_panels, browser, viewports) = (
            &mut self.window_chrome,
            &self.app,
            &self.compare_panels,
            &mut self.series_browser,
            &self.viewports,
        );
        let app = if workspace_layout.is_grid() && active_panel > 0 {
            &compare_panels
                .get(active_panel - 1)
                .ok_or_else(|| NativeViewerError::new("series panel is not initialized"))?
                .app
        } else {
            primary_app
        };
        viewer_events.retain(|event| {
            match window_chrome.handle_event(
                event,
                width,
                height,
                app,
                browser,
                workspace_layout,
                viewports,
            ) {
                Ok(chrome_event) => {
                    chrome_repaint |= chrome_event.repaint;
                    if !chrome_event.consumed {
                        retained_event_count += 1;
                    }
                    if let Some(action) = chrome_event.action {
                        chrome_actions.push((retained_event_count, action));
                    }
                    !chrome_event.consumed
                }
                Err(error) => {
                    chrome_error = Some(error);
                    false
                }
            }
        });
        if let Some(error) = chrome_error {
            return Err(NativeViewerError::from(error));
        }
        let mut chrome_exit = false;
        let mut viewer_exit = false;
        let mut viewer_repaint = false;
        let mut viewer_event_cursor = 0;
        for (event_end, action) in chrome_actions {
            if event_end > viewer_event_cursor {
                match self
                    .apply_viewer_event_segment(&viewer_events[viewer_event_cursor..event_end])?
                {
                    ViewerActionDisposition::Continue { repaint } => {
                        viewer_repaint |= repaint;
                    }
                    ViewerActionDisposition::Exit => {
                        viewer_exit = true;
                        break;
                    }
                }
                viewer_event_cursor = event_end;
            }
            if action == WindowAction::Exit {
                chrome_exit = true;
                break;
            }
            chrome_repaint |= self.apply_window_action(action)?;
        }
        if !chrome_exit && !viewer_exit && viewer_event_cursor < viewer_events.len() {
            match self.apply_viewer_event_segment(&viewer_events[viewer_event_cursor..])? {
                ViewerActionDisposition::Continue { repaint } => {
                    viewer_repaint |= repaint;
                }
                ViewerActionDisposition::Exit => viewer_exit = true,
            }
        }
        // Leave the shortcut in the shared action stream. `0x4f` has no viewer
        // action, while retaining the original bounded batch avoids a second
        // allocation and keeps pointer/focus ordering intact.
        let disposition = if chrome_exit || viewer_exit {
            ViewerActionDisposition::Exit
        } else {
            ViewerActionDisposition::Continue {
                repaint: viewer_repaint,
            }
        };

        let repaint = matches!(
            disposition,
            crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: true }
        );
        let cine_repaint = if matches!(
            disposition,
            crate::app::action_adapter::ViewerActionDisposition::Continue { .. }
        ) {
            let timestamp = self.elapsed_seconds();
            self.tick_cine_at(timestamp)
        } else {
            false
        };
        let frame_changed = repaint || cine_repaint || chrome_repaint || study_reopened;
        if terminal
            || matches!(
                disposition,
                crate::app::action_adapter::ViewerActionDisposition::Exit
            )
            || chrome_exit
        {
            if !self.minimized && frame_changed && !geometry_refreshed {
                self.refresh_frame().map_err(NativeViewerError::from)?;
            } else if !geometry_refreshed {
                record_state(
                    &self.observation,
                    self.active_app(),
                    self.dpi,
                    self.minimized,
                )
                .map_err(NativeViewerError::from)?;
            }
            self.record_terminal_frame(destroyed)
                .map_err(NativeViewerError::from)?;
            return Ok(NativeFlow::Exit);
        }

        if !self.minimized && frame_changed && !geometry_refreshed {
            self.refresh_frame().map_err(NativeViewerError::from)?;
        } else if !geometry_refreshed {
            record_state(
                &self.observation,
                self.active_app(),
                self.dpi,
                self.minimized,
            )
            .map_err(NativeViewerError::from)?;
        }
        Ok(NativeFlow::Continue {
            repaint: !self.minimized && (geometry_refreshed || frame_changed),
        })
    }
}

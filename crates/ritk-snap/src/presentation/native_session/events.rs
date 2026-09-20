//! Native Métis event reduction and lifecycle for the RITK session.

use super::{record_state, NativeViewerError, NativeViewerSession, VIRTUAL_KEY_OPEN_STUDY};
use crate::presentation::{translate_native_events, PresentationEvent};
use metis_platform::native::{NativeApplication, NativeFlow, WindowEvent};
use metis_platform::Framebuffer;
use std::sync::atomic::Ordering;

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
                        self.app.status_message = format!(
                            "DICOM reopen failed; current study remains displayed: {error:#}"
                        );
                        study_reopened = true;
                    }
                }
            }
        }
        // Leave the shortcut in the shared action stream. `0x4f` has no viewer
        // action, while retaining the original bounded batch avoids a second
        // allocation and keeps pointer/focus ordering intact.
        let mut selection_changed = false;
        let mut selection_confirmed = false;
        if !terminal {
            for event in translated.iter() {
                let PresentationEvent::KeyDown {
                    virtual_key,
                    repeated,
                    ..
                } = event
                else {
                    continue;
                };
                let (changed, confirmed) = self
                    .reduce_selection_key(*virtual_key, *repeated)
                    .map_err(NativeViewerError::from)?;
                selection_changed |= changed;
                selection_confirmed |= confirmed;
            }
        }
        let selection_visible = self.selection.is_some();
        let disposition = if selection_visible {
            crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: false }
        } else {
            self.apply_events(&translated)?
        };

        let repaint = matches!(
            disposition,
            crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: true }
        );
        let cine_repaint = if matches!(
            disposition,
            crate::app::action_adapter::ViewerActionDisposition::Continue { .. }
        ) {
            matches!(
                self.app.tick_cine_at(self.elapsed_seconds()),
                crate::app::CineTick::Advanced(_)
            )
        } else {
            false
        };
        let frame_changed =
            repaint || cine_repaint || study_reopened || selection_changed || selection_confirmed;
        if terminal
            || matches!(
                disposition,
                crate::app::action_adapter::ViewerActionDisposition::Exit
            )
        {
            if !self.minimized && frame_changed && !geometry_refreshed {
                self.refresh_frame().map_err(NativeViewerError::from)?;
            } else if !geometry_refreshed {
                record_state(&self.observation, &self.app, self.dpi, self.minimized)
                    .map_err(NativeViewerError::from)?;
            }
            self.record_terminal_frame(destroyed)
                .map_err(NativeViewerError::from)?;
            return Ok(NativeFlow::Exit);
        }

        if !self.minimized && frame_changed && !geometry_refreshed {
            self.refresh_frame().map_err(NativeViewerError::from)?;
        } else if !geometry_refreshed {
            record_state(&self.observation, &self.app, self.dpi, self.minimized)
                .map_err(NativeViewerError::from)?;
        }
        Ok(NativeFlow::Continue {
            repaint: !self.minimized && (geometry_refreshed || frame_changed),
        })
    }
}

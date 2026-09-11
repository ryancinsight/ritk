//! Métis browser canvas viewer workflow.
//!
//! This module is the first browser migration slice beyond the eframe shell:
//! RITK consumes the bounded byte handoff from Métis, applies its existing
//! dropped-input classifier and presents the selected RITK frame through the
//! borrowed canvas seam. DICOM parsing and viewer state stay in [`SnapApp`].

use super::SnapApp;
use crate::presentation::{PresentationFrame, WebCanvasPresenter};
use crate::ui::decide_dropped_input_action;
use moirai_pal::wasm::{spawn_local_with_handle, LocalTaskHandle, WebTimer};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use wasm_bindgen::JsValue;

const FRAME_INTERVAL: Duration = Duration::from_millis(16);

thread_local! {
    static VIEWER_TASK: RefCell<Option<LocalTaskHandle>> = const { RefCell::new(None) };
}

struct BrowserViewer {
    app: SnapApp,
    presenter: WebCanvasPresenter,
    frame: Option<PresentationFrame>,
}

impl BrowserViewer {
    fn new(presenter: WebCanvasPresenter) -> Self {
        Self {
            app: SnapApp::default(),
            presenter,
            frame: None,
        }
    }

    fn tick(&mut self) -> std::io::Result<()> {
        let dropped = super::browser_input::take_dropped_files();
        if !dropped.is_empty() {
            let action = decide_dropped_input_action(&dropped);
            self.app.apply_dropped_input_action(action);
            self.frame = None;
        }

        if self.frame.is_none() {
            self.frame = self
                .app
                .render_browser_frame()
                .map_err(|error| std::io::Error::other(error.to_string()))?;
        }

        if let Some(frame) = self.frame.as_ref() {
            self.presenter.present(frame)?;
        }
        Ok(())
    }
}

/// Starts the RITK DICOM byte-drop workflow on a Métis-owned browser canvas.
pub(crate) fn start_web_canvas(canvas_id: String) -> Result<(), JsValue> {
    stop_web_canvas();
    metis_web::metis_start();
    let presenter = match WebCanvasPresenter::from_canvas_id(&canvas_id) {
        Ok(presenter) => presenter,
        Err(error) => {
            metis_web::metis_stop();
            return Err(JsValue::from_str(&error.to_string()));
        }
    };
    let viewer = Rc::new(RefCell::new(BrowserViewer::new(presenter)));
    let task_viewer = Rc::clone(&viewer);
    let handle = spawn_local_with_handle(async move {
        loop {
            if let Err(error) = task_viewer.borrow_mut().tick() {
                tracing::error!(%error, "RITK browser canvas workflow stopped");
                break;
            }
            let timer = match WebTimer::new(FRAME_INTERVAL) {
                Ok(timer) => timer,
                Err(error) => {
                    tracing::error!(%error, "RITK browser canvas timer stopped");
                    break;
                }
            };
            if let Err(error) = timer.await {
                tracing::error!(%error, "RITK browser canvas timer stopped");
                break;
            }
        }
    });
    VIEWER_TASK.with_borrow_mut(|slot| slot.replace(handle));
    Ok(())
}

/// Stops the RITK browser canvas workflow and releases its timer task.
pub(crate) fn stop_web_canvas() {
    VIEWER_TASK.with_borrow_mut(|slot| slot.take());
    metis_web::metis_stop();
}

impl SnapApp {
    fn render_browser_frame(&self) -> anyhow::Result<Option<PresentationFrame>> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(None);
        };
        let window_center = self
            .viewer_state
            .window_center
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_CENTER);
        let window_width = self
            .viewer_state
            .window_width
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_WIDTH)
            .max(1.0);
        let window_level =
            crate::render::WindowLevel::new(f64::from(window_center), f64::from(window_width));
        let slice_index = match self.axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        PresentationFrame::from_slice(volume, self.axis, slice_index, window_level, self.colormap)
            .map(Some)
    }
}

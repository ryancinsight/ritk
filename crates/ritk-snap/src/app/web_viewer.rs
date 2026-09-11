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
    surface: BrowserSurface,
}

impl BrowserViewer {
    fn new(surface: BrowserSurface) -> Self {
        Self {
            app: SnapApp::default(),
            surface,
        }
    }

    fn tick(&mut self) -> std::io::Result<()> {
        let dropped = super::browser_input::take_dropped_files();
        if !dropped.is_empty() {
            let action = decide_dropped_input_action(&dropped);
            self.app.apply_dropped_input_action(action);
            self.surface.clear();
        }
        self.surface.render_and_present(&self.app)
    }
}

enum BrowserSurface {
    Single {
        presenter: WebCanvasPresenter,
        frame: Option<PresentationFrame>,
    },
    Orthogonal {
        presenters: [WebCanvasPresenter; 3],
        frames: Option<[PresentationFrame; 3]>,
    },
}

impl BrowserSurface {
    fn clear(&mut self) {
        match self {
            Self::Single { frame, .. } => *frame = None,
            Self::Orthogonal { frames, .. } => *frames = None,
        }
    }

    fn render_and_present(&mut self, app: &SnapApp) -> std::io::Result<()> {
        match self {
            Self::Single { presenter, frame } => {
                if frame.is_none() {
                    *frame = app
                        .render_browser_frame()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frame) = frame.as_ref() {
                    presenter.present(frame)?;
                }
            }
            Self::Orthogonal { presenters, frames } => {
                if frames.is_none() {
                    *frames = app
                        .render_browser_frames()
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                }
                if let Some(frames) = frames.as_ref() {
                    for (presenter, frame) in presenters.iter().zip(frames) {
                        presenter.present(frame)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn launch_browser_viewer(viewer: BrowserViewer) -> Result<(), JsValue> {
    let viewer = Rc::new(RefCell::new(viewer));
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

/// Starts the RITK DICOM byte-drop workflow on a Métis-owned browser canvas.
pub(crate) fn start_web_canvas(canvas_id: String) -> Result<(), JsValue> {
    stop_web_canvas();
    let presenter = match WebCanvasPresenter::from_canvas_id(&canvas_id) {
        Ok(presenter) => presenter,
        Err(error) => return Err(JsValue::from_str(&error.to_string())),
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Single {
        presenter,
        frame: None,
    }))
}

/// Starts the RITK DICOM byte-drop workflow on three Métis-owned canvases.
pub(crate) fn start_web_orthogonal_canvases(canvas_ids: [String; 3]) -> Result<(), JsValue> {
    stop_web_canvas();
    let [axial_id, coronal_id, sagittal_id] = canvas_ids;
    let presenters = match (
        WebCanvasPresenter::from_canvas_id(&axial_id),
        WebCanvasPresenter::from_canvas_id(&coronal_id),
        WebCanvasPresenter::from_canvas_id(&sagittal_id),
    ) {
        (Ok(axial), Ok(coronal), Ok(sagittal)) => [axial, coronal, sagittal],
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
            return Err(JsValue::from_str(&error.to_string()));
        }
    };
    metis_web::metis_start();
    launch_browser_viewer(BrowserViewer::new(BrowserSurface::Orthogonal {
        presenters,
        frames: None,
    }))
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
        let window_level = self.browser_window_level();
        let slice_index = match self.axis {
            0 => self.viewer_state.slice_index,
            1 => self.coronal_slice,
            _ => self.sagittal_slice,
        };
        PresentationFrame::from_slice(volume, self.axis, slice_index, window_level, self.colormap)
            .map(Some)
    }

    fn render_browser_frames(&self) -> anyhow::Result<Option<[PresentationFrame; 3]>> {
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(None);
        };
        let indices = [0_usize, 1, 2].map(|axis| self.axis_slice_info(axis).0);
        PresentationFrame::from_orthogonal_slices(
            volume,
            indices,
            self.browser_window_level(),
            self.colormap,
        )
        .map(Some)
    }

    fn browser_window_level(&self) -> crate::render::WindowLevel {
        let window_center = self
            .viewer_state
            .window_center
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_CENTER);
        let window_width = self
            .viewer_state
            .window_width
            .unwrap_or(crate::viewer::DEFAULT_WINDOW_WIDTH)
            .max(1.0);
        crate::render::WindowLevel::new(f64::from(window_center), f64::from(window_width))
    }
}
